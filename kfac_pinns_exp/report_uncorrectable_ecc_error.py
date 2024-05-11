"""Report runs which failed with an uncorrectable ECC error.

This serves to figure out runs that crashed due to unexpected error.
"""

from glob import glob
from os import path

HEREDIR = path.dirname(path.abspath(__file__))

# find all .out files recursively
error_files = []
out_files = glob(path.join(HEREDIR, "**", "*.out"), recursive=True)
for out_file in out_files:
    with open(out_file, "r") as f:
        content = "\n".join(f.readlines())
        if "uncorrectable ECC error encountered" in content:
            error_files.append(out_file)

print(
    f"Found {len(error_files)} runs with uncorrectable ECC errors "
    + f"in {len(out_files)} .out files:"
)
for error_file in error_files:
    print(f"\t{error_file}")
