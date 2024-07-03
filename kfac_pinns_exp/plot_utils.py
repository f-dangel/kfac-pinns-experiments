"""Plotting utility functions."""

from os import remove
from typing import List

from kfac_pinns_exp.utils import run_verbose


def create_animation(frames: List[str], savepath: str):
    """Create an animation from a list of frames.

    Args:
        frames: List of filenames to `.pdf` frames.
        savepath: Path under which the animation will be saved to. Ends with `.gif`.
    """
    savepath_pdf = savepath.replace(".gif", ".pdf")

    # unite the pdfs (NOTE: `pdfunite` requires `poppler` library)
    cmd = ["pdfunite", *frames, savepath_pdf]
    run_verbose(cmd)
    # delete the individual frames
    for f in frames:
        remove(f)

    # create the animation (NOTE: `convert` requires `imagemagick` library)
    cmd = [
        "convert",
        "-verbose",
        "-delay",
        "20",
        "-loop",
        "0",
        "-density",
        "300",
        savepath_pdf,
        savepath,
    ]
    run_verbose(cmd)
