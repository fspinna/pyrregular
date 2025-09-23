import torch
from torch.utils.data import TensorDataset


class IrregularDataset(TensorDataset):
    def __init__(self, X, **kwargs):
        super().__init__(torch.Tensor(X.irr.to_dense(**kwargs)[0]))
