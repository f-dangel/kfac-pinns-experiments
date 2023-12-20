"""Demonstrate Gramian computation on a small toy MLP."""

from functools import partial
from typing import List

from einops import einsum
from torch import Tensor, allclose, manual_seed, rand, zeros, zeros_like
from torch.autograd import grad
from torch.nn import Linear, Module, Sequential, Sigmoid

from kfac_pinns_exp.autodiff_utils import autograd_gramian
from kfac_pinns_exp.hooks_gram_grads_linear import (
    from_grad_input,
    from_hess_input,
    from_output,
)
from kfac_pinns_exp.manual_differentiation import (
    manual_backward,
    manual_forward,
    manual_hessian_backward,
)


def manual_laplace_autograd_gramian(
    layers: List[Module], X: Tensor, layer_idx: int, param_name: str
) -> Tensor:
    """Compute the Gramian of the Laplacian of a parameter.

    The Laplacian is computed manually, the gradients for the Gramian are
    computed via autograd.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer_idx: Index of the layer that contains the parameter we want to use to
            compute the Gramian.
        param_name: Name of the parameter we want to use to compute the Gramian.

    Returns:
        Gramian of the neural network's Laplacian w.r.t. the parameter. Has shape
        `[*p.shape, *p.shape]` where `p` denotes the parameter.
    """
    param = getattr(layers[layer_idx], param_name)
    gramian_flat = zeros(
        param.numel(), param.numel(), device=param.device, dtype=param.dtype
    )

    for x_n in X.split(1):
        # manually compute the Laplacian
        activations = manual_forward(layers, x_n)
        gradients = manual_backward(layers, activations)
        hessians = manual_hessian_backward(layers, activations, gradients)
        laplacian = einsum(hessians[0], "batch d d ->")

        # compute the Laplacian's gradient with autodiff
        grad_laplacian = grad(laplacian, param, allow_unused=True)[0]

        # The last layer's bias is not used in HBP, hence its  contribution to the
        # Gramian is zero. Autograd will return None for this bias.
        if grad_laplacian is None:
            grad_laplacian = zeros_like(param)

        # form the Gramian
        grad_laplacian_flat = grad_laplacian.detach().flatten()
        gramian_flat += einsum(grad_laplacian_flat, grad_laplacian_flat, "i,j->i j")

    return gramian_flat.reshape(*param.shape, *param.shape)


def manual_gramian(
    layers: List[Module], X: Tensor, layer_idx: int, param_name: str
) -> Tensor:
    """Compute the Gramian of the Laplacian of a parameter.

    The Gramian is computed manually.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer_idx: Index of the layer that contains the parameter we want to use to
            compute the Gramian.
        param_name: Name of the parameter we want to use to compute the Gramian.

    Returns:
        Gramian of the neural network's Laplacian w.r.t. the parameter. Has shape
        `[*p.shape, *p.shape]` where `p` denotes the parameter.

    Raises:
        NotImplementedError: If the parameter does not live in a linear layer.
    """
    if not isinstance(layers[layer_idx], Linear):
        raise NotImplementedError(
            f"Gramian only supports Linear layers. Got {layers[layer_idx]}."
        )

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)
    laplacian = einsum(hessians[0], "batch d d ->")

    # extract quantities required for the Gramian of a layer's parameter
    layer_input = activations[layer_idx]
    layer_output = activations[layer_idx + 1]
    layer_grad_input = gradients[layer_idx]
    layer_grad_output = gradients[layer_idx + 1]
    layer_hess_input = hessians[layer_idx]
    layer_hess_output = hessians[layer_idx + 1]

    # compute the gradients w.r.t. all children of the parameter
    # NOTE The first layer's grad_input and the last layer's output is not used in the
    # Laplacian, hence `autograd` will return `None` for them
    d_laplacian_d_hess_input, d_laplacian_d_grad_input, d_laplacian_d_output = grad(
        laplacian, (layer_hess_input, layer_grad_input, layer_output), allow_unused=True
    )

    # use zero tensors for gradients of unused tensors
    if d_laplacian_d_output is None:
        d_laplacian_d_output = zeros_like(layer_output)
    if d_laplacian_d_grad_input is None:
        d_laplacian_d_grad_input = zeros_like(layer_grad_input)

    # compute the Gramian
    assert isinstance(layers[layer_idx], Linear)
    param = getattr(layers[layer_idx], param_name)

    if param_name == "bias":
        # only the layer's output contributes
        return einsum(
            d_laplacian_d_output.detach(),
            d_laplacian_d_output.detach(),
            "batch d_out1, batch d_out2 -> d_out1 d_out2",
        )

    assert param_name == "weight"

    # { ∂(Δu(xᵢ)) / ∂W | i = 1, ..., batch_size }
    grads = zeros(X.shape[0], *param.shape, device=param.device, dtype=param.dtype)

    # 1) contribution from layer's output (forward pass)
    grads += einsum(
        d_laplacian_d_output, layer_input, "batch d_out, batch d_in -> batch d_out d_in"
    ).detach()

    # 2) contribution from layer's input gradient (backward pass)
    grads += einsum(
        layer_grad_output,
        d_laplacian_d_grad_input,
        "batch d_out, batch d_in -> batch d_out d_in",
    ).detach()

    # 3) contribution from layers' input Hessian (Hessian backward pass)
    grads += (
        2
        * einsum(
            d_laplacian_d_hess_input,
            param,
            layer_hess_output,
            "batch d_in1 d_in2, d_out1 d_in1, batch d_out1 d_out2-> batch d_out2 d_in2",
        ).detach()
    )

    return einsum(
        grads,
        grads,
        "batch d_out1 d_in1, batch d_out2 d_in2 -> d_out1 d_in1 d_out2 d_in2",
    )


def manual_hook_gramian(
    layers: List[Module], X: Tensor, layer_idx: int, param_name: str
) -> Tensor:
    """Compute the Gramian of the Laplacian of a parameter. Use tensor hooks.

    Args:
        layers: List of layers in the model.
        X: Input data. First dimension is batch dimension.
        layer_idx: Index of the layer that contains the parameter we want to use to
            compute the Gramian.
        param_name: Name of the parameter we want to use to compute the Gramian.

    Returns:
        Gramian of the neural network's Laplacian w.r.t. the parameter. Has shape
        `[*p.shape, *p.shape]` where `p` denotes the parameter.

    Raises:
        NotImplementedError: If the parameter does not live in a linear layer.
    """
    if not isinstance(layers[layer_idx], Linear):
        raise NotImplementedError(
            f"Gramian only supports Linear layers. Got {layers[layer_idx]}."
        )

    activations = manual_forward(layers, X)
    gradients = manual_backward(layers, activations)
    hessians = manual_hessian_backward(layers, activations, gradients)
    laplacian = einsum(hessians[0], "batch d d ->")

    # extract quantities required for the Gramian of a layer's parameter
    layer_input = activations[layer_idx]
    layer_output = activations[layer_idx + 1]
    layer_grad_input = gradients[layer_idx]
    layer_grad_output = gradients[layer_idx + 1]
    layer_hess_input = hessians[layer_idx]
    layer_hess_output = hessians[layer_idx + 1]

    # install hooks that accumulate the gradients for the Gramian in `gram_grads`
    # { ∂(Δu(xᵢ)) / ∂W | i = 1, ..., batch_size }
    assert isinstance(layers[layer_idx], Linear)
    param = getattr(layers[layer_idx], param_name)
    gram_grads = zeros(X.shape[0], *param.shape, device=param.device, dtype=param.dtype)

    if param_name in {"bias", "weight"}:
        layer_output.register_hook(
            partial(
                from_output,
                layer_input=layer_input,
                param_name=param_name,
                accumulator=gram_grads,
            )
        )

    if param_name == "weight":
        # layer's output, grad_input, and hess_input contribute
        layer_grad_input.register_hook(
            partial(
                from_grad_input,
                layer_grad_output=layer_grad_output,
                accumulator=gram_grads,
            )
        )
        layer_hess_input.register_hook(
            partial(
                from_hess_input,
                layer_hess_output=layer_hess_output,
                weight=param,
                accumulator=gram_grads,
            )
        )

    # backpropagate
    laplacian.backward()

    # form the Gramian
    if param_name == "bias":
        gramian = einsum(gram_grads, gram_grads, "batch d1, batch d2 -> d1 d2")
    else:
        assert param_name == "weight"
        gramian = einsum(
            gram_grads,
            gram_grads,
            "batch d_out1 d_in1, batch d_out2 d_in2 -> d_out1 d_in1 d_out2 d_in2",
        )
    return gramian


def main():
    """Compute the Gramian for one weight matrix in a toy MLP.

    We compare four approaches:

    Approach | Laplacian | Gramian
    ---------|-----------|---------
    1        | autograd  | autograd
    2        | manual    | autograd
    3        | manual    | manual
    4        | manual    | hooks
    """
    # setup
    manual_seed(0)
    batch_size = 10
    X = rand(batch_size, 5)
    layers = [
        Linear(5, 4),
        Sigmoid(),
        Linear(4, 3),
        Sigmoid(),
        Linear(3, 2),
        Sigmoid(),
        Linear(2, 1),
    ]
    model = Sequential(*layers)

    for layer_idx in [0, 2, 4, 6]:
        print(f"Layer {layer_idx} ({layers[layer_idx]})")
        for name in ["weight", "bias"]:
            print(f"\tParameter {name!r}")

            # 1) Laplacian and Gramian via autodiff (functorch)
            param_name = f"{layer_idx}.{name}"
            gramian1 = autograd_gramian(model, X, param_name)

            # 2) manual Laplacian, gradients for Gramian via autograd
            gramian2 = manual_laplace_autograd_gramian(layers, X, layer_idx, name)
            same_1_2 = allclose(gramian1, gramian2)
            print(f"\t\tsame(manual+auto, auto)? {same_1_2}")
            assert same_1_2

            # 3) manual Laplacian and Gramian
            gramian3 = manual_gramian(layers, X, layer_idx, name)
            same_1_3 = allclose(gramian1, gramian3)
            print(f"\t\tsame(manual+auto, manual)? {same_1_3}")
            assert same_1_3

            # 4) via hooks
            gramian4 = manual_hook_gramian(layers, X, layer_idx, name)
            same_1_4 = allclose(gramian1, gramian4)
            print(f"\t\tsame(manual+auto, hook)? {same_1_4}")
            assert same_1_4


if __name__ == "__main__":
    main()
