"""TimesNet.
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
"""

from pypots.classification.timesnet import TimesNet

from pyrregular.wrappers.pypots_wrapper import PyPOTSWrapper


class TimesNetWrapper(PyPOTSWrapper):
    def __init__(self, model, model_params, random_state=None):
        super().__init__(model, model_params, random_state)

    def _fit(self, X, y):
        self.model = self.model(
            n_steps=self.n_steps_,
            n_features=self.n_features_,
            n_classes=self.n_classes_,
            **self.model_params
        )
        X_train, X_val = self._split(X, y)
        self.model.fit(train_set=X_train, val_set=X_val)


timesnet_pipeline = TimesNetWrapper(
    model=TimesNet,
    model_params={
        "n_layers": 2,
        "top_k": 3,
        "d_model": 64,
        "d_ffn": 128,
        "n_kernels": 3,
        "batch_size": 32,
        "epochs": 1000,
        "patience": 50,
        "num_workers": 0,
        "device": None,
    },
)
"""This pipeline applies TimesNet."""
