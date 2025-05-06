from copy import deepcopy

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from pyrregular.models.nodes import standardize


def fill_time_index(arr, nan_delta=1):
    T = deepcopy(arr)

    # time delta
    a_diff = T[:, :, 1:] - T[:, :, :-1]

    # mean time delta
    delta_mean = np.nanmean(a_diff, axis=2, keepdims=True)

    nans = np.isnan(delta_mean)
    if np.any(
        nans
    ):  # this can happen if there is only 1 valid value in the time series
        delta_mean[nans] = nan_delta

    # last timestep
    last_valid_t = np.nanmax(T, axis=2, keepdims=True)

    # find where the nans are
    nan_mask = np.isnan(T)

    # where nans are there is an increasing value from 1 to the last nan
    nan_indices = np.cumsum(nan_mask, axis=2)

    replacement = (last_valid_t + delta_mean * nan_indices)[nan_mask]
    T[nan_mask] = replacement
    return T


def fill_all_nans_signals(X):
    X_ = deepcopy(X)
    mask = np.sum(np.isnan(X_), axis=2) >= X_.shape[2] - 2
    X_[mask, :] = 0
    return X_


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(self, data_size, hidden_size, width_size, depth, *, key):
        """
        data_size: number of input channels (control channels).
        hidden_size: dimension of the hidden state y(t).
        """
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,  # helps control explosion
            key=key,
        )

    def __call__(self, t, y, args):
        # y.shape = (hidden_size,)
        # We produce a matrix of shape (hidden_size, data_size).
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    initial: eqx.nn.MLP
    func: Func
    readout: eqx.nn.Linear  # maps hidden_size -> n_classes

    def __init__(self, data_size, hidden_size, width_size, depth, n_classes, *, key):
        """
        data_size: number of input channels (including time if you use it as a channel).
        hidden_size: dimension of hidden state y(t).
        n_classes: number of classes for multiclass classification.
        """
        key1, key2, key3 = jr.split(key, 3)
        self.initial = eqx.nn.MLP(
            in_size=data_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key1,
        )
        self.func = Func(
            data_size=data_size,
            hidden_size=hidden_size,
            width_size=width_size,
            depth=depth,
            key=key2,
        )
        self.readout = eqx.nn.Linear(
            in_features=hidden_size, out_features=n_classes, key=key3
        )

    def __call__(
        self,
        ts,
        coeffs,
        dt0=1e-2,
        evolving_out=False,
        solver=diffrax.Euler(),
        max_steps=4094,
    ):
        # Create a continuous path from the cubic Hermite coefficients
        control = diffrax.CubicInterpolation(ts, coeffs)

        # Convert to ODE form
        term = diffrax.ControlTerm(self.func, control).to_ode()

        # Initial hidden state
        y0 = self.initial(control.evaluate(ts[0]))

        # Decide how much to save
        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        # Solve the CDE
        solution = diffrax.diffeqsolve(
            term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=y0,
            saveat=saveat,
            throw=False,
            max_steps=max_steps,
        )

        if evolving_out:
            hidden_trajectories = solution.ys  # shape (len(ts), hidden_size)
            logits = jax.vmap(self.readout)(hidden_trajectories)
            return logits
        else:
            final_hidden = solution.ys[-1]  # shape (hidden_size,)
            logits = self.readout(final_hidden)
            return logits


class NeuralCDEClassifier(BaseEstimator, ClassifierMixin):
    """
    A minimal scikit-learn API wrapper for a Neural CDE multiclass classifier.

    X.shape = (n_instances, n_signals, n_timesteps)
      - X[:, :-1, :] => data channels
      - X[:, -1, :]  => the time array for each instance

    y.shape = (n_instances,) with integer labels in {0, 1, ..., n_classes-1}.
    """

    def __init__(
        self,
        hidden_size=8,
        width_size=32,
        depth=1,
        max_iter=50,
        lr=1e-3,
        seed=0,
        print_step=10,
        validate=True,
        patience_lr=10,
        patience_es=20,
        lr_reduce_factor=0.5,
        solver=diffrax.Euler(),
        max_steps=4096,
        reset_loss_after_lr_reduction=False,
    ):
        self.hidden_size = hidden_size
        self.width_size = width_size
        self.depth = depth
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed
        self.print_step = print_step
        self.validate = validate
        self.patience_lr = patience_lr
        self.patience_es = patience_es
        self.lr_reduce_factor = lr_reduce_factor
        self.solver = solver
        self.max_steps = max_steps
        self.reset_loss_after_lr_reduction = reset_loss_after_lr_reduction

        # These will be set after fit:
        self.model_ = None
        self.n_classes_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        # 1) Split into train/val
        if self.validate:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=self.seed, stratify=y
                )
            except ValueError:
                # Fallback if y can't be stratified
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=self.seed
                )
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        self.dt0 = np.nanmin(X[:, -1, 1:] - X[:, -1, :-1])

        # 2) Basic shape checks (on training data)
        _, n_signals_train, _ = X_train.shape
        self.n_classes_ = int(jnp.max(y)) + 1

        # 3) Separate time from data channels (train)
        data_channels_train = fill_all_nans_signals(standardize(X_train[:, :-1, :]))
        times_train = X_train[:, -1, :]
        times_train = fill_time_index(times_train[:, None, :], nan_delta=self.dt0)[
            :, 0, :
        ]
        data_size = n_signals_train - 1

        # 4) Build cubic Hermite coefficients for each instance
        def _make_coeff(xi, ti):
            # xi: shape (data_size, n_timesteps)
            # we need (n_timesteps, data_size) for diffrax
            return diffrax.backward_hermite_coefficients(
                ti, xi.T, fill_forward_nans_at_end=True, replace_nans_at_start=0
            )

        coeffs_train = jax.vmap(_make_coeff)(data_channels_train, times_train)

        # Now do the same for validation data
        data_channels_val = fill_all_nans_signals(standardize(X_val[:, :-1, :]))
        times_val = X_val[:, -1, :]
        times_val = fill_time_index(times_val[:, None, :], nan_delta=self.dt0)[:, 0, :]
        coeffs_val = jax.vmap(_make_coeff)(data_channels_val, times_val)

        # 5) Instantiate the NeuralCDE model
        key = jr.PRNGKey(self.seed)
        self.model_ = NeuralCDE(
            data_size=data_size,
            hidden_size=self.hidden_size,
            width_size=self.width_size,
            depth=self.depth,
            n_classes=self.n_classes_,
            key=key,
        )

        # 6) Set up optimizer
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(eqx.filter(self.model_, eqx.is_array))

        # 7) Define the multiclass cross-entropy for training
        def _train_loss_fn(model):
            def _single_loss(coeff_i, ti, yi):
                logits = model(
                    ti,
                    coeff_i,
                    evolving_out=False,
                    dt0=self.dt0,
                    solver=self.solver,
                    max_steps=self.max_steps,
                )
                logprobs = jnn.log_softmax(logits, axis=-1)
                return -logprobs[yi]

            losses = jax.vmap(_single_loss)(coeffs_train, times_train, y_train)
            return jnp.mean(losses)

        # Validation loss
        def _val_loss_fn(model):
            def _single_loss(coeff_i, ti, yi):
                logits = model(
                    ti,
                    coeff_i,
                    evolving_out=False,
                    dt0=self.dt0,
                    solver=self.solver,
                    max_steps=self.max_steps,
                )
                logprobs = jnn.log_softmax(logits, axis=-1)
                return -logprobs[yi]

            losses = jax.vmap(_single_loss)(coeffs_val, times_val, y_val)
            return jnp.mean(losses)

        # Single gradient step
        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def _step(model):
            return _train_loss_fn(model)

        # 8) Early stopping setup
        best_val_loss = float("inf")
        stagnation_counter_lr = 0  # Counter for learning rate reduction
        stagnation_counter_es = 0  # Counter for early stopping
        best_model_path = "best_model.eqx"  # File path to store the best model

        current_lr = self.lr  # Initial learning rate

        # Training loop
        for step in range(self.max_iter):
            # Compute training loss and gradients
            train_loss, grads = _step(self.model_)
            updates, opt_state = optax.adam(current_lr).update(grads, opt_state)
            self.model_ = eqx.apply_updates(self.model_, updates)

            # Compute validation loss
            val_loss = _val_loss_fn(self.model_)

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stagnation_counter_lr = 0  # Reset learning rate counter
                stagnation_counter_es = 0  # Reset early stopping counter
                eqx.tree_serialise_leaves(
                    best_model_path, self.model_
                )  # Save best model
            else:
                stagnation_counter_lr += 1
                stagnation_counter_es += 1

            # Learning rate reduction
            if stagnation_counter_lr >= self.patience_lr:
                current_lr *= self.lr_reduce_factor  # Reduce learning rate
                print(f"Reducing learning rate to {current_lr:.5f}")
                stagnation_counter_lr = 0  # Reset counter for learning rate
                if self.reset_loss_after_lr_reduction:
                    best_val_loss = val_loss

            # Early stopping
            if stagnation_counter_es >= self.patience_es:
                print("Early stopping triggered!")
                break

            # Display progress
            if step % self.print_step == 0 or step == self.max_iter - 1:
                print(
                    f"Step {step} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
                )

        # Reload the best model
        self.model_ = eqx.tree_deserialise_leaves(best_model_path, self.model_)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for each instance in X.
        Return shape: (n_instances, n_classes).
        """
        if not self.is_fitted_:
            raise ValueError("NeuralCDEClassifier is not fitted yet.")

        data_channels = fill_all_nans_signals(standardize(X[:, :-1, :]))
        times = X[:, -1, :]
        times = fill_time_index(times[:, None, :], nan_delta=self.dt0)[:, 0, :]

        def _make_coeff(xi, ti):
            return diffrax.backward_hermite_coefficients(
                ti, xi.T, fill_forward_nans_at_end=True, replace_nans_at_start=0
            )

        coeffs = jax.vmap(_make_coeff)(data_channels, times)

        def _single_forward(coeff_i, ti):
            logits = self.model_(
                ti,
                coeff_i,
                evolving_out=False,
                dt0=self.dt0,
                solver=self.solver,
                max_steps=self.max_steps,
            )
            return jnn.softmax(logits, axis=-1)

        return np.array(jax.vmap(_single_forward)(coeffs, times))

    def predict(self, X):
        """
        Predict the most likely class for each instance in X.
        Return shape: (n_instances,).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        preds = self.predict(X)
        return np.mean(preds == y).item()


ncde_pipeline = NeuralCDEClassifier(
    hidden_size=8,
    width_size=32,
    depth=1,
    max_iter=1000,
    lr=1e-2,
    seed=0,
    print_step=10,
    validate=True,
    patience_lr=50,
    patience_es=200,
    lr_reduce_factor=0.5,
    solver=diffrax.Euler(),
    max_steps=100,
)
