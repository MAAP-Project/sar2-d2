import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sar2d2 import sar2d2  # type: ignore


def existing_file(path: str) -> Path | object:
    return p if (p := Path(path)).is_file() else argparse.FileType()(path)


def parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector"
    )
    parser.add_argument(
        "--calibration-file",
        metavar="PATH",
        type=existing_file,
        help="path to calibration file",
        required=True,
    )
    parser.add_argument(
        "--bbox",
        type=float,
        help="lat/lon bounding box (example: --bbox -118.068 34.222 -118.058 34.228)",
        nargs=4,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        metavar="PATH",
        type=Path,
        help="path to output directory to write to",
        required=True,
    )

    return parser.parse_args(args)


def main(args: Sequence[str]) -> None:
    ns = parse_args(args)
    sar2d2(ns.calibration_file, ns.bbox, ns.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
