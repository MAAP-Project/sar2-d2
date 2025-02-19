#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Feb 18 17:16:46 2025

Author: Juyoung Song
"""

import argparse
import glob
import os
from typing import Any, List, Tuple

import h5py
import numpy as np
import xarray as xr
from osgeo import gdal, osr


def get_args() -> argparse.Namespace:
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory containing NiSAR image files in GUNW.',
        dest='directory'
    )
    parser.add_argument(
        '--exportfilename',
        type=str,
        default='exportchangedetection.tif',
        help='Export filename.',
        dest='exportfilename'
    )
    parser.add_argument(
        '--valuethreshold',
        type=float,
        default=0.15,
        help='Sdiff screening threshold.',
        dest='valuethreshold'
    )
    return parser.parse_args()


def read_gunw_h5(filename: str, pol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, np.ndarray, np.ndarray]:
    """
    Read key datasets from a GUNW HDF5 file.
    
    Parameters:
        filename: Path to the HDF5 file.
        pol: Polarization string (e.g., 'HH').
    
    Returns:
        Tuple containing:
          - ifg: Unwrapped interferometric phase.
          - coh: Coherence magnitude.
          - iono: Ionosphere phase screen.
          - conn: Connected components.
          - mask: Data mask.
          - ref_zero_dop_end_time: Reference zero Doppler end time.
          - sec_zero_dop_end_time: Secondary zero Doppler end time.
          - x_path: x-coordinate array.
          - y_path: y-coordinate array.
    """
    with h5py.File(filename, 'r') as src:
        freq_group_path = '/science/LSAR/GUNW/grids/frequencyA/'
        pol_group_path = f'{freq_group_path}/unwrappedInterferogram/{pol}/'
        ifg_path = f'{pol_group_path}/unwrappedPhase'
        coh_path = f'{pol_group_path}/coherenceMagnitude'
        iono_path = f'{pol_group_path}/ionospherePhaseScreen'
        conn_path = f'{pol_group_path}/connectedComponents'
        mask_path = f'{freq_group_path}/unwrappedInterferogram/mask'
        x_path_key = f'{pol_group_path}/xCoordinates'
        y_path_key = f'{pol_group_path}/yCoordinates'

        ifg = np.array(src[ifg_path], dtype='float32')
        coh = np.array(src[coh_path], dtype='float32')
        iono = np.array(src[iono_path], dtype='float32')
        conn = np.array(src[conn_path], dtype='float32')
        mask = np.array(src[mask_path], dtype='float32')
        x_path = np.array(src[x_path_key], dtype='float32')
        y_path = np.array(src[y_path_key], dtype='float32')

        ref_zero_dop_end_path = '/science/LSAR/identification/referenceZeroDopplerEndTime'
        ref_zero_dop_end_time = src[ref_zero_dop_end_path][()]
        if isinstance(ref_zero_dop_end_time, bytes):
            ref_zero_dop_end_time = ref_zero_dop_end_time.decode('utf-8')

        sec_zero_dop_end_path = '/science/LSAR/identification/secondaryZeroDopplerEndTime'
        sec_zero_dop_end_time = src[sec_zero_dop_end_path][()]
        if isinstance(sec_zero_dop_end_time, bytes):
            sec_zero_dop_end_time = sec_zero_dop_end_time.decode('utf-8')
            
    return ifg, coh, iono, conn, mask, ref_zero_dop_end_time, sec_zero_dop_end_time, x_path, y_path


def pixel_window_sdiff_all_overlap(
    data: xr.DataArray,
    timearray: List[Any],
    x1: int,
    y1: int,
    window_size: int = 10,
    threshold: float = 0.15
) -> np.ndarray:
    """
    Compute sdiff values for a single pixel's time series in overlapping windows.
    Each window shifts by one time step.
    
    Parameters:
        data: xarray.DataArray with dims ('time', 'x', 'y')
        timearray: List of time coordinate values.
        x1: X-index of the pixel.
        y1: Y-index of the pixel.
        window_size: Number of time points per overlapping window.
        threshold: Screening threshold for the cumulative sum.
    
    Returns:
        1D NumPy array of sdiff values (dtype float32).
    """
    pixel_series: np.ndarray = data[:, x1, y1].values
    time_len: int = pixel_series.size

    if time_len < window_size:
        return np.array([], dtype=np.float32)

    num_windows: int = time_len - window_size + 1
    sdiff_results: np.ndarray = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        start = i
        end = i + window_size
        window = pixel_series[start:end]
        window_median = np.median(window)
        window_demean = window - window_median
        cumsum_vals = np.cumsum(window_demean)
        if cumsum_vals[-2] < cumsum_vals[-1] + threshold:
            sdiff = 0
        else:
            sdiff = cumsum_vals.max() - cumsum_vals.min()
        sdiff_results[i] = sdiff

    return sdiff_results


if __name__ == '__main__':
    args = get_args()
    import tifffile as tiff  # For potential TIFF handling if needed

    imagesRUNW: List[np.ndarray] = []
    imagesRCOH: List[np.ndarray] = []
    refTimeArray: List[Any] = []
    secTimeArray: List[Any] = []

    h5_files: List[str] = glob.glob(os.path.join(args.directory, "*.h5")) if args.directory else glob.glob("*.h5")
    
    # Loop over HDF5 files and load data
    for filename in h5_files:
        try:
            runw, rcoh, _, _, _, refTime, secTime, xp, yp = read_gunw_h5(filename, pol='HH')
            imagesRUNW.append(runw)
            imagesRCOH.append(rcoh)
            refTimeArray.append(refTime)
            secTimeArray.append(secTime)
            print(f"Loaded {filename} with reference time: {refTime}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    # Sort coherence images by reference time
    time_image_pairs = list(zip(refTimeArray, imagesRCOH))
    time_image_pairs_sorted = sorted(time_image_pairs, key=lambda pair: pair[0])
    refTimeArray_sorted, imagesRCOH_sorted = zip(*time_image_pairs_sorted)
    refTimeArray_sorted = list(refTimeArray_sorted)
    imagesRCOH_sorted = list(imagesRCOH_sorted)
    
    # Stack coherence images into a 3D array with shape (time, y, x)
    stacked_images = np.stack(imagesRCOH_sorted, axis=0)
    xr_data: xr.DataArray = xr.DataArray(
        stacked_images,
        dims=['time', 'y', 'x'],
        coords={'time': refTimeArray_sorted}
    )
    nt, xbin, ybin = xr_data.shape

    # Sdiff estimation using overlapping windows (window size set to 10)
    windowsize: int = 10
    Sdiff: np.ndarray = np.zeros((xbin, ybin, nt - windowsize + 1), dtype=np.float32)
    for j in range(ybin):
        print(f"Processing row {j+1} of {ybin} for Sdiff estimation")
        for i in range(xbin):
            Sdiff[i, j, :] = pixel_window_sdiff_all_overlap(xr_data, refTimeArray_sorted, i, j, window_size=windowsize, threshold=args.valuethreshold)
    
    # Compute Sdifffinal (max Sdiff) and Sdiffidx (window index of max Sdiff) for each pixel
    Sdifffinal: np.ndarray = np.zeros((xbin, ybin), dtype=np.float32)
    Sdiffidx: np.ndarray = np.zeros((xbin, ybin), dtype=np.int16)
    for j in range(ybin):
        print(f"Processing row {j+1} of {ybin} for final Sdiff")
        for i in range(xbin):
            temp: np.ndarray = Sdiff[i, j, :]
            Sdifffinal[i, j] = np.max(temp)
            Sdiffidx[i, j] = int(np.argmax(temp))
    
    # --- Export GeoTIFFs using native coordinate system ---
    # Here we use xp and yp directly (assumed to be in the native CRS).
    # Create a meshgrid from xp and yp.
    X, Y = np.meshgrid(xp, yp)  # shape: (len(yp), len(xp))
    # Assume uniform spacing:
    pixel_width: float = xp[1] - xp[0]
    pixel_height: float = yp[1] - yp[0]
    # For geotransform, use the upper-left corner:
    gt: Tuple[float, float, float, float, float, float] = (xp[0], pixel_width, 0, yp[-1], 0, -pixel_height)
    print("Geotransform:", gt)
    
    # Define raster dimensions for export.
    num_cols: int = len(xp)  # number of columns (width)
    num_rows: int = len(yp)  # number of rows (height)
    
    # Export Sdifffinal as a GeoTIFF (Float32)
    output_file_final: str = args.exportfilename.replace('.tif', '_Sdifffinal.tif')
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file_final, num_cols, num_rows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    # Use the native CRS code if available; if not, update accordingly.
    # Here we leave the projection undefined or you can set it using srs.ImportFromEPSG(...)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.WriteArray(Sdifffinal)
    band.SetNoDataValue(0)
    band.FlushCache()
    dataset = None
    print("GeoTIFF written to:", output_file_final)
    
    # Export Sdiffidx as a GeoTIFF (Int16)
    output_file_idx: str = args.exportfilename.replace('.tif', '_Sdiffidx.tif')
    dataset_idx = driver.Create(output_file_idx, num_cols, num_rows, 1, gdal.GDT_Int16)
    dataset_idx.SetGeoTransform(gt)
    dataset_idx.SetProjection(srs.ExportToWkt())
    band_idx = dataset_idx.GetRasterBand(1)
    band_idx.WriteArray(Sdiffidx)
    band_idx.SetNoDataValue(-9999)
    band_idx.FlushCache()
    dataset_idx = None
    print("GeoTIFF written to:", output_file_idx)
