"""Script to measure time and memory of Laplacians."""

from argparse import ArgumentParser
from os import makedirs, path
from time import time
from typing import Callable, Union

from memory_profiler import memory_usage
from torch import Tensor, allclose, cuda, device, float64, manual_seed, rand
from torch.func import hessian, vmap
from torch.nn import Module, Sequential

from kfac_pinns_exp.forward_laplacian import manual_forward_laplacian
from kfac_pinns_exp.train import SUPPORTED_MODELS, set_up_layers

HEREDIR = path.dirname(path.abspath(__file__))
DEFAULTDIR = path.join(HEREDIR, "results", "raw")
IMPLEMENTATIONS = {"backward", "forward"}
DTYPE = float64
EQUATION = "poisson"
METRICS = {"time", "peakmem"}


def autograd_input_laplacian(
    model: Union[Module, Callable[[Tensor], Tensor]],
    X: Tensor,
) -> Tensor:
    """Compute the batched Laplacian of the model w.r.t. its input.

    Args:
        model: The model whose Laplacian will be computed. Must produce batched scalars
            as output. Can either be an `nn.Module` or a tensor-to-tensor function.
        X: The input to the model. First dimension is the batch dimension.
            Must be differentiable.

    Returns:
        The Laplacianof the model w.r.t. X. Has shape `[batch_size, 1]`.
    """

    def f(x: Tensor) -> Tensor:
        """Forward pass on an un-batched input.

        Args:
            x: Un-batched 1d input.

        Returns:
            Un-batched scalar output.

        Raises:
            ValueError: If the input or output have incorrect shapes.
        """
        if x.ndim != 1:
            raise ValueError(f"Input must be 1d. Got {x.ndim}d.")

        output = model(x).squeeze()

        if output.ndim != 0:
            raise ValueError(f"Output must be 0d. Got {output.ndim}d.")

        return output

    def laplace_f(x: Tensor) -> Tensor:
        """Compute the Laplacian of an un-batched input.

        Args:
            x: Un-batched 1d input.

        Returns:
            Un-batched scalar output of shape `[1]`.
        """
        hess_f_x = hessian(f)(x)
        return hess_f_x.trace().unsqueeze(-1)

    return vmap(laplace_f)(X)


def maybe_synchronize(dev: device):
    """Synchronize the device if it is a CUDA device.

    Args:
        dev: The device to synchronize.
    """
    if "cuda" in str(dev):
        cuda.synchronize()


def get_raw_savepath(
    input_dimension: int,
    implementation: str,
    model: str,
    batch_size: int,
    device: str,
    seed: int,
    metric: str,
    datadir: str,
) -> str:
    """Get the save path for the profiled case.

    Args:
        input_dimension: The input dimension.
        implementation: The used implementation.
        model: The used model.
        batch_size: The used batch size.
        device: The used device.
        seed: The used seed.
        metric: The used metric.
        datadir: The used data directory.

    Returns:
        The save path for the Laplacian case.
    """
    return path.join(
        datadir,
        "_".join(
            [
                model,
                implementation,
                device,
                metric,
                f"d_in_{input_dimension}",
                f"N_{batch_size}",
                f"seed_{seed}",
            ]
        )
        + ".csv",
    )


if __name__ == "__main__":
    parser = ArgumentParser("Parse parameters for profiling Laplacians")
    parser.add_argument(
        "--input_dimension",
        type=int,
        help="Input dimension of the neural network",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        help="Neural network architecture whose Laplacian is evaluated",
        required=True,
    )
    parser.add_argument(
        "--implementation",
        type=str,
        choices=IMPLEMENTATIONS,
        help="Implementation of the Laplacian",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices={"cuda", "cpu"},
        help="Device to use",
        required=True,
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=METRICS,
        help="Metric to measure",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--seed", type=int, help="Random seed (optional)", default=0)
    parser.add_argument(
        "--savedir", type=str, help="Directory to save results", default=DEFAULTDIR
    )

    args = parser.parse_args()
    makedirs(args.savedir, exist_ok=True)

    manual_seed(args.seed)  # make deterministic
    DEV = device(args.device)

    X = rand(args.batch_size, args.input_dimension, device=DEV, dtype=DTYPE)

    layers = set_up_layers(args.model, EQUATION, args.input_dimension)
    layers = [layer.to(DEV, DTYPE) for layer in layers]
    model = Sequential(*layers).to(DEV, DTYPE)
    maybe_synchronize(DEV)

    def laplacian(implementation: str) -> Tensor:
        """Compute the Laplacian.

        Args:
            implementation: The implementation to use.

        Returns:
            The Laplacian of shape `(batch_size, 1)`.

        Raises:
            NotImplementedError: If the implementation is not supported.
        """
        if implementation == "backward":
            lap = autograd_input_laplacian(model, X)
        elif implementation == "forward":
            lap = manual_forward_laplacian(layers, X)[-1]["laplacian"]
        else:
            raise NotImplementedError

        return lap

    # warm-up
    laplacian(args.implementation)

    num_steps = {"time": 3, "peakmem": 3}[args.metric]

    def f():
        """Execute forward Laplacian computation multiple times."""
        for _ in range(num_steps):
            laplacian(args.implementation)

    description = (
        f"[{args.model}, {args.implementation}, X.shape={tuple(X.shape)}, "
        + f"device={args.device}, seed={args.seed}]"
    )
    savepath = get_raw_savepath(
        args.input_dimension,
        args.implementation,
        args.model,
        args.batch_size,
        args.device,
        args.seed,
        args.metric,
        args.savedir,
    )

    if args.metric == "time":
        t_start = time()
        f()
        maybe_synchronize(DEV)
        t_end = time()
        t_step_seconds = (t_end - t_start) / num_steps

        print(f"{description} Time taken: {t_step_seconds:.2e} s / iter")
        with open(savepath, "w") as f_result:
            f_result.write(f"{t_step_seconds}")

    elif args.metric == "peakmem":
        if "cuda" in str(DEV):
            f()
            peakmem_bytes = cuda.max_memory_allocated()
        else:
            peakmem_bytes = memory_usage(f, interval=1e-4, max_usage=True) * 2**20
        peakmem_gib = peakmem_bytes / 2**30

        print(f"{description} Memory usage: {peakmem_gib:.2e} GiB")
        with open(savepath, "w") as f_result:
            f_result.write(f"{peakmem_gib}")

    else:
        raise NotImplementedError

    # check equivalence of both implementations
    backward_lap = laplacian("backward")
    forward_lap = laplacian("forward")
    assert allclose(backward_lap, forward_lap, rtol=1e-4, atol=1e-7)
