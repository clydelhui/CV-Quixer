"""Parameter shift rule for CV quantum circuits.

The parameter shift rule (PSR) estimates gradients without backpropagating
through the circuit simulation. For a gate parameter θ and circuit expectation
value E(θ):

    ∂E/∂θ ≈ r · [E(θ + s) − E(θ − s)]

where s is the shift value and r = 1 / (2 sin(s)). For most Gaussian gates
the exact shift is s = π/2, r = 1/2.

This is useful when:
  1. You want to match the gradient estimation method of real quantum hardware
     (which cannot backpropagate through physical circuit execution).
  2. A gate matrix is not analytically differentiable in PyTorch
     (rare with our implementation, but possible for custom gates).

The implementation uses a torch.autograd.Function so that PSR integrates
transparently with the rest of the PyTorch training loop — loss.backward()
works identically from the caller's perspective regardless of grad mode.

Reference:
    Mitarai et al., "Quantum circuit learning" (2018)
    https://arxiv.org/abs/1803.00745

    Schuld et al., "Evaluating analytic gradients on quantum hardware" (2019)
    https://arxiv.org/abs/1811.11184
"""

from __future__ import annotations

from typing import Callable

import torch


class ParameterShiftFunction(torch.autograd.Function):
    """Custom autograd Function implementing the parameter shift rule.

    Decoupled from individual gate types — the caller provides a circuit_fn
    closure that maps a flat parameter tensor to expectation values. Adding
    new gate types never requires changes here.

    Usage:
        result = ParameterShiftFunction.apply(circuit_fn, params, shift)

    Args (via .apply):
        circuit_fn: Callable[[torch.Tensor], torch.Tensor]
                    Maps a flat parameter tensor of shape (N_params,) to a
                    tensor of expectation values. Must be a pure function
                    (no side effects, re-runnable with shifted params).
        params:     Flat tensor of all circuit parameters, shape (N_params,).
                    Must have requires_grad=True.
        shift:      Shift value s (scalar float). Default π/2 (= 1.5708...).
    """

    @staticmethod
    def forward(
        ctx,
        circuit_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        shift: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(params)
        ctx.circuit_fn = circuit_fn
        ctx.shift = shift
        with torch.no_grad():
            return circuit_fn(params)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None]:
        (params,) = ctx.saved_tensors
        circuit_fn = ctx.circuit_fn
        shift = ctx.shift
        r = 0.5 / torch.sin(torch.tensor(shift))   # r = 1 / (2 sin(s))

        n_params = params.shape[0]
        param_grads = torch.zeros_like(params)

        with torch.no_grad():
            for i in range(n_params):
                params_plus = params.clone()
                params_plus[i] += shift
                params_minus = params.clone()
                params_minus[i] -= shift

                e_plus = circuit_fn(params_plus)
                e_minus = circuit_fn(params_minus)

                # Chain rule: scalar gradient for this parameter
                # grad_output: upstream gradient from the loss
                param_grads[i] = (r * (e_plus - e_minus) * grad_output).sum()

        # Return gradients for: circuit_fn (None — not a tensor), params, shift (None)
        return None, param_grads, None
