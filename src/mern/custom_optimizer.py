from typing import Iterable
import torch
import torch.optim as optim

def create_rmsprop_optimizer(params: Iterable[torch.Tensor], lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8, weight_decay: float = 0, momentum: float = 0, centered: bool = False) -> torch.optim.Optimizer:
    """
    Creates an RMSProp optimizer with the specified parameters.

    Parameters:
    -----------
    params : Iterable[torch.Tensor]
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, optional
        Learning rate.
    alpha : float, optional
        Smoothing constant.
    eps : float, optional
        Term added to the denominator to improve numerical stability.
    weight_decay : float, optional
        Weight decay (L2 penalty).
    momentum : float, optional
        Momentum factor.
    centered : bool, optional
        If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.

    Returns:
    --------
    torch.optim.Optimizer
        An RMSProp optimizer instance.
    """
    return optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered) 