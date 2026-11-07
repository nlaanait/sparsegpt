"""Revised note §3.4: centered covariance or a trailing ones column."""

from __future__ import annotations

import torch
import torch.nn as nn
import transformers

from sparsifier.prune.sparsegpt import SparseGPT, _weight_matrix


class InPassBiasSparseGPT(SparseGPT):
    """SparseGPT whose mask already accounts for a planned bias correction.

    ``mode="centered"`` replaces H with the centered second moment.
    ``mode="ones_column"`` appends a constant-one input, excludes it from
    mask selection, and writes the last column into ``layer.bias``.
    """

    def __init__(self, layer: nn.Module, mode: str = "centered") -> None:
        if mode not in {"centered", "ones_column"}:
            raise ValueError("mode must be 'centered' or 'ones_column'")
        self.mode = mode
        self.real_layer = layer
        if mode == "ones_column":
            if getattr(layer, "bias", None) is None:
                raise TypeError("ones_column mode requires a bias parameter")
            weight = _weight_matrix(layer)
            dout, din = weight.shape
            aug = nn.Linear(din + 1, dout, bias=False, device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                aug.weight[:, :din] = weight
                aug.weight[:, din] = layer.bias.detach().to(dtype=weight.dtype)
            super().__init__(aug)
            self._din = din
        else:
            super().__init__(layer)
            self._din = self.columns
        self._sum_phi = torch.zeros(self._din, device=self.device, dtype=torch.float32)
        self._n_tokens = 0
        self._centered = False

    def add_batch(self, inputs: torch.Tensor, out=None, blocksize: int = 1024) -> None:
        if self.mode == "ones_column":
            ones = torch.ones((*inputs.shape[:-1], 1), device=inputs.device, dtype=inputs.dtype)
            super().add_batch(torch.cat((inputs, ones), dim=-1), out, blocksize)
            return
        super().add_batch(inputs, out, blocksize)
        flat = inputs.reshape(-1, inputs.shape[-1]).to(device=self.device, dtype=torch.float32)
        self._sum_phi += flat.sum(dim=0)
        self._n_tokens += int(flat.shape[0])

    def _center_hessian(self) -> None:
        if self._centered:
            return
        if self._n_tokens < 1 or self.nsamples < 1:
            raise RuntimeError("Collect activations before centered pruning")
        mu = self._sum_phi / self._n_tokens
        self.H = self.H - (2.0 * self._n_tokens / self.nsamples) * torch.outer(mu, mu)
        self._centered = True

    def fasterprune(self, *args, **kwargs) -> None:
        if self.mode == "centered":
            self._center_hessian()
            super().fasterprune(*args, **kwargs)
            return
        kwargs = dict(kwargs)
        kwargs["protected_trailing"] = 1
        super().fasterprune(*args, **kwargs)
        pruned = _weight_matrix(self.layer)
        real_w = pruned[:, : self._din]
        bias = pruned[:, self._din]
        if isinstance(self.real_layer, transformers.Conv1D):
            real_w = real_w.t()
        self.real_layer.weight.data = real_w.reshape(self.real_layer.weight.shape).to(
            dtype=self.real_layer.weight.dtype
        )
        self.real_layer.bias.data = bias.to(dtype=self.real_layer.bias.dtype)
