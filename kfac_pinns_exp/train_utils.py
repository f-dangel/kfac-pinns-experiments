"""Utility functions and classes for the training script."""

from math import log10
from typing import Callable, Tuple

from torch import Tensor, device, dtype, logspace


class LoggingTrigger:
    """Class to trigger logging."""

    def __init__(self, num_steps: int, max_logs: int, num_seconds: float):
        """Initialize the trigger.

        Args:
            num_steps: The number of steps to train.
            max_logs: The maximum number of logs to create.
            num_seconds: The number of seconds to train. If non-zero, `num_steps` and
                `max_logs` are ignored.
        """
        if num_seconds == 0.0:
            logged_steps = {
                int(s) for s in logspace(0, log10(num_steps - 1), max_logs - 1).int()
            } | {0, num_steps - 1}

            def should_log(step: int) -> bool:
                """Function to determine whether to log a step given a step limit.

                Args:
                    step: Current training step.

                Returns:
                    Whether to log the step.
                """
                return step in logged_steps

        else:
            self.next_log = 1

            def should_log(step: int) -> bool:
                """Function to determine whether to log a step given a time limit.

                Args:
                    step: Current training step.

                Returns:
                    Whether to log the step.
                """
                if step in {0, 1}:
                    return True
                elif step >= self.next_log:
                    self.next_log *= 1.1
                    return True
                return False

        self.should_log = should_log


class KillTrigger:
    """Class to kill training."""

    def __init__(self, num_steps: int, num_seconds: float):
        """Initialize the trigger.

        Args:
            num_steps: The number of steps to train.
            num_seconds: The number of seconds to train. If `0`, train for `num_steps`,
                otherwise ignore `num_steps` and train for `num_seconds` seconds.
        """

        def should_kill(step: int, seconds_elapsed: float) -> bool:
            """Function to determine whether training should be killed.

            Args:
                step: Current training step.
                seconds_elapsed: Time elapsed in seconds since training started.

            Returns:
                Whether to kill training.
            """
            if num_seconds == 0.0:
                return step >= num_steps - 1
            else:
                return seconds_elapsed >= num_seconds

        self.should_kill = should_kill


class DataLoader:
    """Produces the same batch for a specified number of steps."""

    def __init__(
        self,
        data_func: Callable[[], Tuple[Tensor, Tensor]],
        dev: device,
        dt: dtype,
        frequency: int,
    ):
        """Save data-generating function, device and data type.

        Args:
            data_func: Function that generates tuples of batched inputs and labels.
            dev: The device to load the data to.
            dt: The data type to load the data to.
            frequency: The frequency at which a batch is updated. `0` means always the
                same batch is used
        """
        self.data_func = data_func
        self.dev = dev
        self.dt = dt
        self.frequency = frequency

    def __iter__(self):
        """Create a data iterator.

        Returns:
            Itself.
        """
        self.step = 0
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        """Return the next batch.

        Returns:
            Next batch from a new function evaluation.
        """
        create_new = (self.frequency == 0 and self.step == 0) or (
            self.frequency != 0 and self.step % self.frequency == 0
        )
        if create_new:
            X, y = self.data_func()
            self.X, self.y = X.to(self.dev, self.dt), y.to(self.dev, self.dt)
        self.step += 1
        return (self.X, self.y)
