from __future__ import annotations

import argparse

from heatindex.utils import mat_variable_names


def main() -> None:
    p = argparse.ArgumentParser(description="List top-level variables in a MATLAB .mat file.")
    p.add_argument("mat_file")
    args = p.parse_args()

    for name in mat_variable_names(args.mat_file):
        print(name)


if __name__ == "__main__":
    main()
