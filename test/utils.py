"""Utility functions."""

from torch import Tensor, allclose, cuda, device, isclose


def report_nonclose(
    tensor1: Tensor,
    tensor2: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    verbose: bool = True,
):
    """Compare two PyTorch tensors.

    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        rtol: Relative tolerance. Default: `1e-5`.
        atol: Absolute tolerance. Default: `1e-8`.
        equal_nan: Whether comparing two NaNs is considered as `True`. Default: `False`.
        verbose: Whether to print the mismatches. Default: `True`.

    Raises:
        ValueError: If the two tensors don't match.
    """
    kwargs = {"rtol": rtol, "atol": atol, "equal_nan": equal_nan}
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensor shape mismatch: {tensor1.shape} vs. {tensor2.shape}.",
        )

    if not allclose(tensor1, tensor2, **kwargs):
        if verbose:
            correct = 0
            for idx, (t1, t2) in enumerate(zip(tensor1.flatten(), tensor2.flatten())):
                if not isclose(t1, t2, **kwargs):
                    print(f"{idx}: {t1} ≠ {t2} (ratio {t1 / t2:.5f})")
                else:
                    correct += 1
            print(f"Matches: {correct} / {tensor1.numel()}")
            print(f"Max: {tensor1.max():.5f}, {tensor2.max():.5f}")
            print(f"Min: {tensor1.min():.5f}, {tensor2.min():.5f}")
        raise ValueError("Compared arrays don't match.")


DEVICES = [device("cuda"), device("cpu")] if cuda.is_available() else [device("cpu")]
DEVICE_IDS = [f"device={dev}" for dev in DEVICES]
