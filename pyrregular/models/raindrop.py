from pypots.classification.raindrop import Raindrop

from pyrregular.wrappers.pypots_wrapper import PyPOTSWrapper


class RaindropWrapper(PyPOTSWrapper):
    def __init__(self, model, model_params, random_state=None):
        super().__init__(model, model_params, random_state)

    def _fit(self, X, y):
        self.model = self.model(
            n_steps=self.n_steps_,
            n_features=self.n_features_,
            n_classes=self.n_classes_,
            d_model=self.n_features_ * 4,
            **self.model_params
        )
        X_train, X_val = self._split(X, y)
        self.model.fit(train_set=X_train, val_set=X_val)


raindrop_pipeline = RaindropWrapper(
    model=Raindrop,
    model_params={
        "n_layers": 2,
        "d_ffn": 256,
        "n_heads": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "epochs": 1000,
        "patience": 50,
        "num_workers": 0,
        "device": None,
    },
)
