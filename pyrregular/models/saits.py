from pypots.classification.saits import SAITS

from pyrregular.wrappers.pypots_wrapper import PyPOTSWrapper


class SAITSWrapper(PyPOTSWrapper):
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


saits_pipeline = SAITSWrapper(
    model=SAITS,
    model_params={
        "n_layers": 2,
        "d_model": 256,
        "n_heads": 4,
        "d_k": 64,
        "d_v": 64,
        "d_ffn": 128,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "batch_size": 32,
        "epochs": 1000,
        "patience": 50,
        "num_workers": 0,
        "device": None,
    },
)
