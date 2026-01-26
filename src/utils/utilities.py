# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch
from prettytable import PrettyTable
import numpy as np


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
            self.save = False

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            self.logger.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).",
            )
        self.val_score_min = val_score
        self.save = True


@torch.jit.script
def ifft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """
    Performs an inverse Fast Fourier Transform (iFFT) on the input tensor along specified dimensions.
    Args:
        scan (torch.Tensor): The input tensor to be transformed. Must have 2, 3, 4, or 5 dimensions.
        dim (tuple[int, int], optional): The dimensions along which to perform the iFFT. Defaults to (-2, -1).
    Returns:
        torch.Tensor: The transformed tensor after applying iFFT.
    Raises:
        ValueError: If the input tensor does not have 2, 3, 4, or 5 dimensions.
    """

    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )
    scan = torch.fft.ifftshift(scan, dim=dim)
    scan = torch.fft.ifft2(scan, dim=dim)
    scan = torch.fft.fftshift(scan, dim=dim)

    return scan


@torch.jit.script
def fft(scan: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """
    Performs an Fast Fourier Transform (FFT) on the input tensor along specified dimensions.
    Args:
        scan (torch.Tensor): The input tensor to be transformed. Must have 2, 3, 4, or 5 dimensions.
        dim (tuple[int, int], optional): The dimensions along which to perform the FFT. Defaults to (-2, -1).
    Returns:
        torch.Tensor: The transformed tensor after applying FFT.
    Raises:
        ValueError: If the input tensor does not have 2, 3, 4, or 5 dimensions.
    """

    if scan.ndim not in [2, 3, 4, 5]:
        raise ValueError(
            f"Dimension of input needs to be 2, 3, 4 or 5 (B x C x H x W x D), but got {scan.ndim}!"
        )

    scan = torch.fft.ifftshift(scan, dim=dim)
    scan = torch.fft.fft2(scan, dim=dim)
    scan = torch.fft.fftshift(scan, dim=dim)

    return scan


def count_parameters(model: torch.nn.Module) -> tuple[PrettyTable, int]:
    """Counts the model parameters.

    Args:
        model (torch.nn.Module): a torch model

    Returns:
        int: number of model parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params
