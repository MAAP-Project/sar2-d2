from pathlib import Path
import isce3

BBox = tuple[float, float, float, float]


def sar2d2(calibration_file: Path, bbox: BBox, output_dir: Path):
    print(f"{calibration_file=}")
    print(f"{bbox=}")
    print(f"{output_dir=}")
