"""BRITS.
Bidirectional Recurrent Imputation for Time Series.
"""

from pypots.classification.brits import BRITS

from pyrregular.wrappers.pypots_wrapper import PyPOTSWrapper


class BritsWrapper(PyPOTSWrapper):

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


brits_pipeline = BritsWrapper(
    model=BRITS,
    model_params={
        "rnn_hidden_size": 256,
        "batch_size": 32,
        "epochs": 1000,
        "patience": 50,
        "num_workers": 0,
        "device": None,
    },
)
"""This pipeline applies BRITS"""
