import os
import logging
import warnings
from dataclasses import dataclass
from collections import Counter

from abc import ABC, abstractmethod
from collections.abc import Iterator
import h5py
import mimetypes
import numpy as np
from osgeo import osr, gdal
from osgeo.gdal import Dataset
from pathlib import Path
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from typing import Any
import math
import rasterio
from dataclasses import dataclass
import numpy as np
from osgeo import gdal, osr, ogr
from pyproj import Transformer
from scipy.signal import convolve2d
import tempfile


def requires_reprojection(geogrid_mosaic,
                          rtc_image: str,
                          nlooks_image: str = None) -> bool:
    '''
    Check if the reprojection is required to mosaic input raster

    Parameters
    ----------
    geogrid_mosaic: isce3.product.GeoGridParameters
        Mosaic geogrid
    rtc_image: str
        Path to the geocoded RTC image
    nlooks_image: str (optional)
        Path to the nlooks raster

    Returns
    flag_requires_reprojection: bool
        True if reprojection is necessary to mosaic inputs
        False if the images are aligned, so that no reprojection is necessary.
    '''

    # Accepted error in the coordinates as floating number
    maxerr_coord = 1.0e-6

    raster_rtc_image = gdal.Open(rtc_image, gdal.GA_ReadOnly)
    if nlooks_image is not None:
        raster_nlooks = gdal.Open(nlooks_image, gdal.GA_ReadOnly)

    # Compare geotransforms of RTC image and nlooks (if provided)
    if (nlooks_image is not None and
            raster_rtc_image.GetGeoTransform() !=
            raster_nlooks.GetGeoTransform()):
        error_str = (f'ERROR geolocations of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    # Compare dimension - between RTC imagery and corresponding nlooks
    if (nlooks_image is not None and
        (raster_rtc_image.RasterXSize != raster_nlooks.RasterXSize or
            raster_rtc_image.RasterYSize != raster_nlooks.RasterYSize)):
        error_str = (f'ERROR dimensions of {raster_rtc_image} and'
                     f' {raster_nlooks} do not match')
        raise ValueError(error_str)

    rasters_to_check = [raster_rtc_image]
    if nlooks_image is not None:
        rasters_to_check += [raster_nlooks]

    srs_mosaic = osr.SpatialReference()
    srs_mosaic.ImportFromEPSG(geogrid_mosaic.epsg)

    proj_mosaic = osr.SpatialReference(wkt=srs_mosaic.ExportToWkt())
    epsg_mosaic = proj_mosaic.GetAttrValue('AUTHORITY', 1)

    for raster in rasters_to_check:
        x0, dx, _, y0, _, dy = raster.GetGeoTransform()
        projection = raster.GetProjection()

        # check spacing
        if dx != geogrid_mosaic.spacing_x:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if dy != geogrid_mosaic.spacing_y:
            flag_requires_reprojection = True
            return flag_requires_reprojection

        # check projection
        if projection != srs_mosaic.ExportToWkt():
            proj_raster = osr.SpatialReference(wkt=projection)
            epsg_raster = proj_raster.GetAttrValue('AUTHORITY', 1)

            if epsg_raster != epsg_mosaic:
                flag_requires_reprojection = True
                return flag_requires_reprojection

        # check the coordinates
        if (abs((x0 - geogrid_mosaic.start_x) % geogrid_mosaic.spacing_x) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

        if (abs((y0 - geogrid_mosaic.start_y) % geogrid_mosaic.spacing_y) >
                maxerr_coord):
            flag_requires_reprojection = True
            return flag_requires_reprojection

    flag_requires_reprojection = False
    return flag_requires_reprojection

def get_meta_from_tif(tif_file_name):
    """Read metadata from geotiff

    Parameters
    ----------
    input_tif_str: str
        geotiff file path to read the band

    Returns
    -------
    meta_dict: dict
        dictionary containing geotransform, projection, image size,
        utmzone, and epsg code.
    """
    if type(tif_file_name) is list:
        tif_name = tif_file_name[0]
    else:
        tif_name = tif_file_name
    tif_gdal = gdal.Open(tif_name)
    meta_dict = {}
    meta_dict['band_number'] = tif_gdal.RasterCount
    meta_dict['geotransform'] = tif_gdal.GetGeoTransform()
    meta_dict['projection'] = tif_gdal.GetProjection()
    meta_dict['length'] = tif_gdal.RasterYSize
    meta_dict['width'] = tif_gdal.RasterXSize
    proj = osr.SpatialReference(wkt=meta_dict['projection'])
    meta_dict['utmzone'] = proj.GetUTMZone()
    output_epsg = proj.GetAttrValue('AUTHORITY', 1)
    meta_dict['epsg'] = output_epsg
    tif_gdal = None

    return meta_dict



def convert_rounded_coordinates(
        corners,
        from_epsg, to_epsg,
        x_snap=30, y_snap=30):
    """
    Transform and round coordinates from one EPSG coordinate system to another.

    Parameters
    ----------
    corners : list of tuples
        A list of coordinate pairs (x, y) in the source coordinate reference
        system (CRS).
    from_epsg : int
        The EPSG code of the source CRS.
    to_epsg : int
        The EPSG code of the destination CRS.
    x_snap : int, optional
        The grid size in the x-direction to which transformed x-coordinates
        will be rounded. Default is 30.
    y_snap : int, optional
        The grid size in the y-direction to which transformed y-coordinates
        will be rounded. Default is 30.

    Returns
    -------
    transformed_coords : list of tuples
        A list of transformed and rounded coordinate pairs (x, y) in the
        destination CRS.

    Notes
    -----
    This function converts a list of coordinates from one EPSG coordinate
    system to another and then rounds the transformed coordinates to the
    nearest multiples of specified grid sizes (x_snap and y_snap). This is
    useful for aligning coordinates to a regular grid in the destination CRS.
    """
    transformer = Transformer.from_crs(f"epsg:{from_epsg}", f"epsg:{to_epsg}",
                                       always_xy=True)

    rounded_coords = []
    for corner in corners:
        x, y = transformer.transform(corner[0], corner[1])
        rounded_coords.append((np.round(x / x_snap) * x_snap,
                               np.round(y / y_snap) * y_snap))

    return rounded_coords

def change_epsg_tif(input_tif, output_tif, epsg_output,
                    resample_method='nearest',
                    output_nodata='NaN'):
    """Resample the input geotiff image to new EPSG code.

    Parameters
    ----------
    input_tif: str
        geotiff file path to be changed
    output_tif: str
        geotiff file path to be saved
    epsg_output: int
        new EPSG code
    """
    metadata = get_meta_from_tif(input_tif)

    # Get coordinates of the upper left corner
    x_min = metadata['geotransform'][0]
    y_max = metadata['geotransform'][3]

    # Get pixel dimensions
    pixel_x_spacing = metadata['geotransform'][1]
    pixel_y_spacing = metadata['geotransform'][5]

    # Get the number of rows and columns
    cols = metadata['width']
    rows = metadata['length']

    # Calculate coordinates of the lower right corner
    x_max = x_min + (cols * pixel_x_spacing)
    y_min = y_max + (rows * pixel_y_spacing)

    corners = [
        (x_min, y_max),  # Top-left
        (x_max, y_max),  # Top-right
        (x_min, y_min),  # Bottom-left
        (x_max, y_min)  # Bottom-right
    ]

    corner_output = convert_rounded_coordinates(
        corners,
        metadata['epsg'],
        epsg_output,
        x_snap=pixel_x_spacing,
        y_snap=pixel_y_spacing)

    x_coords, y_coords = zip(*corner_output)
    x_min_output, x_max_output = min(x_coords), max(x_coords)
    y_min_output, y_max_output = min(y_coords), max(y_coords)

    opt = gdal.WarpOptions(
        dstSRS=f'EPSG:{epsg_output}',
        resampleAlg=resample_method,
        outputBounds=[
            x_min_output,
            y_min_output,
            x_max_output,
            y_max_output],
        dstNodata=output_nodata,
        xRes=metadata['geotransform'][1],
        yRes=metadata['geotransform'][5],
        format='GTIFF',
        creationOptions=['COMPRESS=DEFLATE',
                         'PREDICTOR=2'])

    gdal.Warp(output_tif, input_tif, options=opt)


def _aggregate_10m_to_30m_conv(image, ratio, normalize_flag):
    """
    Aggregate pixel values in an image to a lower resolution based on
    a specified target label and convolution with a normalization option.
    The output is a downsampled image showing the density of the target
    label within blocks of the original image.

    Parameters
    ----------
    image : ndarray
        The input binary image where the specific label is being targeted.
    ratio : int
        The size and downsample factor represented by the kernel size (e.g., 3 for a 3x3 kernel).
    normalize_flag : bool
        A flag to determine whether to normalize the kernel by its area (True) or use a simple
        summation (False).

    Returns
    -------
    aggregated_data : ndarray
        An aggregated binary image of lower resolution, indicating the density of the target
        label within each block of the original image.
    """
    # Define a 3x3 kernel where all values are 1
    kernel = np.ones((ratio, ratio), dtype=np.int32)

    # Perform the convolution with 'valid' mode to only keep full 3x3 blocks
    aggregated_data = convolve2d(image, kernel, mode='same')

    # Since the convolution is done with stride 1,
    # we need to downsample the result by a factor of ratio
    aggregated_data = aggregated_data[ratio//2::ratio,
                                      ratio//2::ratio]
    if normalize_flag:
        valid_area = image > 0
        pixel_count = convolve2d(valid_area, kernel, mode='same')
        pixel_count = pixel_count[ratio//2::ratio,
                                  ratio//2::ratio]

        aggregated_data = aggregated_data /pixel_count

    return aggregated_data


def _calculate_output_bounds(geotransform,
                             width,
                             length,
                             output_spacing):
    """
    Calculate the bounding box coordinates adjusted to the nearest multiple of the specified output spacing.

    Parameters
    ----------
    geotransform : tuple
        GeoTransform tuple from a GDAL dataset, containing six coefficients.
    width : int
        The number of pixels in x-direction (width) of the dataset.
    length : int
        The number of pixels in y-direction (length) of the dataset.
    output_spacing : float
        Desired spacing for the output, which adjusts the bounding box dimensions.

    Returns
    -------
    list
        Adjusted bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    x_min = geotransform[0]
    x_max = x_min + width * geotransform[1] 

    if geotransform[5] < 0:
        y_max = geotransform[3]
        y_min = y_max + length * geotransform[5]
    else:
        y_min = geotransform[3]
        y_max = y_min + length * geotransform[5]

    x_diff = x_max - x_min
    y_diff = y_max - y_min

    if x_diff % output_spacing != 0:
        x_max = x_min + (x_diff // output_spacing + 1) * output_spacing

    output_spacing = -1 * np.abs(output_spacing)
    if y_diff % output_spacing != 0:
        y_min = y_max + (y_diff // np.abs(output_spacing) + 1) * output_spacing

    return [x_min, y_min, x_max, y_max]

def mosaic_single_output_file(list_rtc_images, list_nlooks, mosaic_filename,
                              mosaic_mode, scratch_dir='', geogrid_in=None,
                              temp_files_list=None, no_data_value=np.nan,
                              verbose=True):
    '''
    Mosaic RTC images saving the output into a single multi-band file

    Parameters
    -----------
        list_rtc: list
            List of the path to the rtc geobursts
        list_nlooks: list
            List of the nlooks raster that corresponds to list_rtc
        mosaic_filename: str
            Path to the output mosaic
        scratch_dir: str (optional)
            Directory for temporary files
        geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and
            projection. The geogrid of the output mosaic will automatically
            determined when it is None
        temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
        verbose : bool
            Flag to enable/disable the verbose mode
    '''
    mosaic_dict = compute_mosaic_array(
        list_rtc_images, list_nlooks, mosaic_mode, scratch_dir=scratch_dir,
        geogrid_in=geogrid_in, temp_files_list=temp_files_list,
        verbose=verbose, no_data_value=no_data_value)

    arr_numerator = mosaic_dict['mosaic_array']
    description_list = mosaic_dict['description_list']
    length = mosaic_dict['length']
    width = mosaic_dict['width']
    num_bands = mosaic_dict['num_bands']
    wkt_projection = mosaic_dict['wkt_projection']
    xmin_mosaic = mosaic_dict['xmin_mosaic']
    ymax_mosaic = mosaic_dict['ymax_mosaic']
    posting_x = mosaic_dict['posting_x']
    posting_y = mosaic_dict['posting_y']

    # Retrieve the datatype information from the first input image
    reference_raster = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
    datatype_mosaic = reference_raster.GetRasterBand(1).DataType
    reference_raster = None

    # Write out the array
    drv_out = gdal.GetDriverByName('Gtiff')
    raster_out = drv_out.Create(mosaic_filename,
                                width, length, num_bands,
                                datatype_mosaic)

    raster_out.SetGeoTransform((xmin_mosaic, posting_x, 0,
                                ymax_mosaic, 0, posting_y))
    raster_out.SetProjection(wkt_projection)

    for i_band in range(num_bands):
        gdal_band = raster_out.GetRasterBand(i_band+1)
        gdal_band.WriteArray(arr_numerator[i_band])
        gdal_band.SetDescription(description_list[i_band])


def compute_mosaic_array(list_rtc_images,
                         list_nlooks,
                         mosaic_mode,
                         scratch_dir='',
                         geogrid_in=None,
                         temp_files_list=None,
                         no_data_value=np.isnan,
                         verbose=True):
    '''
    Mosaic S-1 geobursts and return the mosaic as dictionary

    Parameters
    -----------
       list_rtc: list
           List of the path to the rtc geobursts
       list_nlooks: list
           List of the nlooks raster that corresponds to list_rtc
       mosaic_mode: str
            Mosaic mode. Choices: "average", "first", and "bursts_center"
       scratch_dir: str (optional)
            Directory for temporary files
       geogrid_in: isce3.product.GeoGridParameters, default: None
            Geogrid information to determine the output mosaic's shape and
            projection. The geogrid of the output mosaic will automatically
            determined when it is None
       temp_files_list: list (optional)
            Mutable list of temporary files. If provided,
            paths to the temporary files generated will be
            appended to this list
       verbose: flag (optional)
            Flag to enable (True) or disable (False) the verbose mode
    Returns
        mosaic_dict: dict
            Mosaic dictionary
    '''

    mosaic_mode_choices_list = ['average', 'first', 'bursts_center']
    if mosaic_mode.lower() not in mosaic_mode_choices_list:
        raise ValueError(f'ERROR invalid mosaic mode: {mosaic_mode}.'
                         f' Choices: {", ".join(mosaic_mode_choices_list)}')

    num_raster = len(list_rtc_images)
    description_list = []
    num_bands = None
    posting_x = None
    posting_y = None

    list_geo_transform = np.zeros((num_raster, 6))
    list_dimension = np.zeros((num_raster, 2), dtype=np.int32)

    for i, path_rtc in enumerate(list_rtc_images):

        raster_in = gdal.Open(path_rtc, gdal.GA_ReadOnly)
        list_geo_transform[i, :] = raster_in.GetGeoTransform()
        list_dimension[i, :] = (raster_in.RasterYSize, raster_in.RasterXSize)

        # Check if the number of bands are consistent over the
        # input RTC rasters
        if num_bands is None:
            num_bands = raster_in.RasterCount

        elif num_bands != raster_in.RasterCount:
            raise ValueError(f'ERROR: the file "{os.path.basename(path_rtc)}"'
                             f' has {raster_in.RasterCount} bands. Expected:'
                             f' {num_bands}.')

        if len(description_list) == 0:
            for i_band in range(num_bands):
                description_list.append(
                    raster_in.GetRasterBand(i_band+1).GetDescription())

        # Close GDAL dataset
        raster_in = None

    if geogrid_in is None:
        # determine GeoTransformation, posting, dimension, and projection from
        # the input raster
        for i in range(num_raster):
            if list_geo_transform[:, 1].max() == list_geo_transform[:, 1].min():
                posting_x = list_geo_transform[0, 1]

            if list_geo_transform[:, 5].max() == list_geo_transform[:, 5].min():
                posting_y = list_geo_transform[0, 5]

        # determine the dimension and the upper left corner of the output
        # mosaic
        xmin_mosaic = list_geo_transform[:, 0].min()
        ymax_mosaic = list_geo_transform[:, 3].max()
        xmax_mosaic = (list_geo_transform[:, 0] +
                       list_geo_transform[:, 1]*list_dimension[:, 1]).max()
        ymin_mosaic = (list_geo_transform[:, 3] +
                       list_geo_transform[:, 5]*list_dimension[:, 0]).min()

        dim_mosaic = (int(np.ceil((ymin_mosaic - ymax_mosaic) / posting_y)),
                      int(np.ceil((xmax_mosaic - xmin_mosaic) / posting_x)))

        gdal_ds_raster_in = gdal.Open(list_rtc_images[0], gdal.GA_ReadOnly)
        wkt_projection = gdal_ds_raster_in.GetProjectionRef()
        del gdal_ds_raster_in

    else:
        # Directly bring the geogrid information from the input parameter
        xmin_mosaic = geogrid_in.start_x
        ymax_mosaic = geogrid_in.start_y
        posting_x = geogrid_in.spacing_x
        posting_y = geogrid_in.spacing_y

        dim_mosaic = (geogrid_in.length, geogrid_in.width)

        xmax_mosaic = xmin_mosaic + posting_x * dim_mosaic[1]
        ymin_mosaic = ymax_mosaic + posting_y * dim_mosaic[0]

        srs_mosaic = osr.SpatialReference()
        srs_mosaic.ImportFromEPSG(geogrid_in.epsg)
        wkt_projection = srs_mosaic.ExportToWkt()

    if verbose:
        print('    mosaic geogrid:')
        print('        start X:', xmin_mosaic)
        print('        end X:', xmax_mosaic)
        print('        start Y:', ymax_mosaic)
        print('        end Y:', ymin_mosaic)
        print('        spacing X:', posting_x)
        print('        spacing Y:', posting_y)
        print('        width:', dim_mosaic[1])
        print('        length:', dim_mosaic[0])
        print('        projection:', wkt_projection)
        print('        number of bands: {num_bands}')

    if mosaic_mode.lower() == 'average':
        arr_numerator = np.zeros((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                 dtype=float)
        arr_denominator = np.zeros(dim_mosaic, dtype=float)
    else:
        arr_numerator = np.full((num_bands, dim_mosaic[0], dim_mosaic[1]),
                                np.nan, dtype=float)
        if mosaic_mode.lower() == 'bursts_center':
            arr_distance = np.full(dim_mosaic, np.nan, dtype=float)

    for i, path_rtc in enumerate(list_rtc_images):
        if i < len(list_nlooks):
            path_nlooks = list_nlooks[i]
        else:
            path_nlooks = None

        if verbose:
            print(f'    mosaicking ({i+1}/{num_raster}): '
                  f'{os.path.basename(path_rtc)}')
        if geogrid_in is not None and requires_reprojection(
                geogrid_in, path_rtc, path_nlooks):
            if verbose:
                print('        the image requires reprojection/relocation')

                relocated_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

                print('        reprojecting image to temporary file:',
                      relocated_file)

            if temp_files_list is not None:
                temp_files_list.append(relocated_file)

            warp_creation_options = gdal.WarpOptions(
                creationOptions=['COMPRESS=DEFLATE',
                                 'PREDICTOR=2'])

            gdal.Warp(
                relocated_file, path_rtc,
                format='GTiff',
                dstSRS=wkt_projection,
                outputBounds=[
                    geogrid_in.start_x,
                    geogrid_in.start_y +
                    geogrid_in.length * geogrid_in.spacing_y,
                    geogrid_in.start_x +
                    geogrid_in.width * geogrid_in.spacing_x,
                    geogrid_in.start_y],
                multithread=True,
                xRes=geogrid_in.spacing_x,
                yRes=abs(geogrid_in.spacing_y),
                resampleAlg='average',
                errorThreshold=0,
                dstNodata=np.nan,
                options=warp_creation_options
                )
            path_rtc = relocated_file

            if path_nlooks is not None:
                relocated_file_nlooks = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

                print('        reprojecting number of looks layer to temporary'
                      ' file:', relocated_file_nlooks)

                if temp_files_list is not None:
                    temp_files_list.append(relocated_file_nlooks)

                gdal.Warp(
                    relocated_file_nlooks, path_nlooks,
                    format='GTiff',
                    dstSRS=wkt_projection,
                    outputBounds=[
                        geogrid_in.start_x,
                        geogrid_in.start_y +
                        geogrid_in.length * geogrid_in.spacing_y,
                        geogrid_in.start_x +
                        geogrid_in.width * geogrid_in.spacing_x,
                        geogrid_in.start_y],
                    multithread=True,
                    xRes=geogrid_in.spacing_x,
                    yRes=abs(geogrid_in.spacing_y),
                    resampleAlg='cubic',
                    errorThreshold=0,
                    dstNodata=np.nan)
                path_nlooks = relocated_file_nlooks

            offset_imgx = 0
            offset_imgy = 0
        else:

            # calculate the burst RTC's offset wrt. the output mosaic in
            # the image coordinate
            offset_imgx = int((list_geo_transform[i, 0] - xmin_mosaic) /
                              posting_x + 0.5)
            offset_imgy = int((list_geo_transform[i, 3] - ymax_mosaic) /
                              posting_y + 0.5)

        if verbose:
            print('        image offset (x, y): '
                  f'({offset_imgx}, {offset_imgy})')

        if path_nlooks is not None:
            nlooks_gdal_ds = gdal.Open(path_nlooks, gdal.GA_ReadOnly)
            arr_nlooks = nlooks_gdal_ds.ReadAsArray()
            invalid_ind = np.isnan(arr_nlooks)
            arr_nlooks[invalid_ind] = 0.0
        else:
            arr_nlooks = 1

        rtc_image_gdal_ds = gdal.Open(path_rtc, gdal.GA_ReadOnly)

        for i_band in range(num_bands):

            band_ds = rtc_image_gdal_ds.GetRasterBand(i_band + 1)
            arr_rtc = band_ds.ReadAsArray()

            if i_band == 0:
                length = min(arr_rtc.shape[0], dim_mosaic[0] - offset_imgy)
                width = min(arr_rtc.shape[1], dim_mosaic[1] - offset_imgx)

            if (length != arr_rtc.shape[0] or
                    width != arr_rtc.shape[1]):
                # Image needs to be cropped to fit in the mosaic
                arr_rtc = arr_rtc[0:length, 0:width]

            if mosaic_mode.lower() == 'average':
                # Replace NaN values with 0
                arr_rtc[np.isnan(arr_rtc)] = 0.0

                arr_numerator[i_band,
                              offset_imgy: offset_imgy + length,
                              offset_imgx: offset_imgx + width] += \
                    arr_rtc * arr_nlooks

                if path_nlooks is not None:
                    arr_denominator[
                        offset_imgy: offset_imgy + length,
                        offset_imgx: offset_imgx + width] += arr_nlooks
                else:
                    arr_denominator[
                        offset_imgy: offset_imgy + length,
                        offset_imgx: offset_imgx + width] += np.asarray(
                        arr_rtc > 0, dtype=np.byte)

                continue

            arr_temp = arr_numerator[i_band, offset_imgy: offset_imgy + length,
                                     offset_imgx: offset_imgx + width].copy()
            if not np.isnan(no_data_value):
                arr_temp[arr_temp == no_data_value] = np.nan

            if i_band == 0 and mosaic_mode.lower() == 'first':
                ind = np.isnan(arr_temp)
            elif i_band == 0 and mosaic_mode.lower() == 'bursts_center':
                geotransform = rtc_image_gdal_ds.GetGeoTransform()

                arr_new_distance = _compute_distance_to_burst_center(
                    arr_rtc, geotransform)

                arr_distance_temp = arr_distance[
                    offset_imgy: offset_imgy + length,
                    offset_imgx: offset_imgx + width]
                ind = np.logical_or(np.isnan(arr_distance_temp),
                                    arr_new_distance <= arr_distance_temp)

                arr_distance_temp[ind] = arr_new_distance[ind]
                arr_distance[
                    offset_imgy: offset_imgy + length,
                    offset_imgx: offset_imgx + width] = arr_distance_temp

                del arr_distance_temp

            arr_temp[ind] = arr_rtc[ind]
            arr_numerator[i_band,
                          offset_imgy: offset_imgy + length,
                          offset_imgx: offset_imgx + width] = arr_temp

        rtc_image_gdal_ds = None
        nlooks_gdal_ds = None

    if mosaic_mode.lower() == 'average':
        # Mode: average
        # `arr_numerator` holds the accumulated sum. Now, we divide it
        # by `arr_denominator` to get the average value
        for i_band in range(num_bands):
            valid_ind = arr_denominator > 0
            arr_numerator[i_band][valid_ind] = \
                arr_numerator[i_band][valid_ind] / arr_denominator[valid_ind]

            arr_numerator[i_band][arr_denominator == 0] = np.nan

    mosaic_dict = {
        'mosaic_array': arr_numerator,
        'description_list': description_list,
        'length': dim_mosaic[0],
        'width': dim_mosaic[1],
        'num_bands': num_bands,
        'wkt_projection': wkt_projection,
        'xmin_mosaic': xmin_mosaic,
        'ymax_mosaic': ymax_mosaic,
        'posting_x': posting_x,
        'posting_y': posting_y
    }
    return mosaic_dict


def majority_element(num_list):
    """
    Determine the majority element in a list
    Parameters
    ----------
    num_list : List[int]
        A list of integers where the majority element needs to be determined.

    Returns
    -------
    int:
        The majority element in the list. If no majority exists,
        it may return any element from the list.
    """

    counter = Counter(np.array(num_list))
    most_common = counter.most_common()
    most_freq_elem = most_common[0][0]

    return most_freq_elem



@dataclass
class DSWXGeogrid:
    """
    A dataclass representing the geographical grid configuration
    for an RTC (Radar Terrain Correction) run.

    Attributes:
    -----------
    start_x : float
        The starting x-coordinate of the grid.
    start_y : float
        The starting y-coordinate of the grid.
    end_x : float
        The ending x-coordinate of the grid.
    end_y : float
        The ending y-coordinate of the grid.
    spacing_x : float
        The spacing between points in the x-direction.
    spacing_y : float
        The spacing between points in the y-direction.
    length : int
        The number of points in the y-direction.
    width : int
        The number of points in the x-direction.
    epsg : int
        The EPSG code representing the coordinate reference system of the grid.
    """
    start_x: float = np.nan
    start_y: float = np.nan
    end_x: float = np.nan
    end_y: float = np.nan
    spacing_x: float = np.nan
    spacing_y: float = np.nan
    length: int = np.nan
    width: int = np.nan
    epsg: int = np.nan

    def get_geogrid_from_geotiff(self,
                                 geotiff_path):
        """
        Extract geographical grid parameters from a GeoTIFF file
        and update the dataclass attributes.

        Parameters
        ----------
        geotiff_path : str
            The file path to the GeoTIFF file from which the grid
            parameters are to be extracted.
        """
        tif_gdal = gdal.Open(geotiff_path)
        geotransform = tif_gdal.GetGeoTransform()
        self.start_x = geotransform[0]
        self.spacing_x = geotransform[1]

        self.start_y = geotransform[3]
        self.spacing_y = geotransform[5]

        self.length = tif_gdal.RasterYSize
        self.width = tif_gdal.RasterXSize

        self.end_x = self.start_x + self.width * self.spacing_x
        self.end_y = self.start_y + self.length * self.spacing_y

        projection = tif_gdal.GetProjection()
        proj = osr.SpatialReference(wkt=projection)
        output_epsg = proj.GetAttrValue('AUTHORITY', 1)
        self.epsg = int(output_epsg)
        tif_gdal = None
        del tif_gdal

    @classmethod
    def from_geotiff(cls, geotiff_path):
        """
        Extract geographical grid parameters from a GeoTIFF file
        and update the dataclass attributes.
        Parameters
        ----------
        geotiff_path : str
            The file path to the GeoTIFF file from which the grid
            parameters are to be extracted.
        """
        tif_gdal = gdal.Open(geotiff_path)
        geotransform = tif_gdal.GetGeoTransform()
        start_x, spacing_x, _, start_y, _, spacing_y = geotransform
        length = tif_gdal.RasterYSize
        width = tif_gdal.RasterXSize
        end_x = start_x + width * spacing_x
        end_y = start_y + length * spacing_y
        projection = tif_gdal.GetProjection()
        proj = osr.SpatialReference(wkt=projection)
        output_epsg = proj.GetAttrValue('AUTHORITY', 1)
        epsg = int(output_epsg)
        tif_gdal = None
        del tif_gdal
        return cls(start_x, start_y, end_x, end_y, spacing_x, spacing_y,
                   length, width, epsg)

    def update_geogrid(self, geotiff_path):
        """
        Update the existing geographical grid parameters based on a new
        GeoTIFF file, extending the grid to encompass both the existing
        and new grid areas.
        """
        new_geogrid = DSWXGeogrid.from_geotiff(geotiff_path)

        if self.epsg != new_geogrid.epsg and not np.isnan(self.epsg):
            raise ValueError("EPSG codes of the existing and "
                             "new geogrids do not match.")
        self.start_x = min(filter(lambda x: not np.isnan(x),
                                  [self.start_x, new_geogrid.start_x]))
        self.end_x = max(filter(lambda x: not np.isnan(x),
                                [self.end_x, new_geogrid.end_x]))

        if self.spacing_y > 0 or np.isnan(self.spacing_y):
            self.end_y = max(filter(lambda x: not np.isnan(x),
                                    [self.end_y, new_geogrid.end_y]))
            self.start_y = min(filter(lambda x: not np.isnan(x),
                                      [self.start_y, new_geogrid.start_y]))
        else:
            self.start_y = max(filter(lambda x: not np.isnan(x),
                                      [self.start_y, new_geogrid.start_y]))
            self.end_y = min(filter(lambda x: not np.isnan(x),
                                    [self.end_y, new_geogrid.end_y]))

        self.spacing_x = new_geogrid.spacing_x \
            if not np.isnan(new_geogrid.spacing_x) else self.spacing_x
        self.spacing_y = new_geogrid.spacing_y \
            if not np.isnan(new_geogrid.spacing_y) else self.spacing_y

        if not np.isnan(self.start_x) and not np.isnan(self.end_x) and \
           not np.isnan(self.spacing_x):
            self.width = int((self.end_x - self.start_x) / self.spacing_x)

        if not np.isnan(self.start_y) and not np.isnan(self.end_y) and \
           not np.isnan(self.spacing_y):
            self.length = int((self.end_y - self.start_y) / self.spacing_y)

        self.epsg = new_geogrid.epsg \
            if not np.isnan(new_geogrid.epsg) else self.epsg


class DataReader(ABC):
    def __init__(self, row_blk_size: int, col_blk_size: int):
        self.row_blk_size = row_blk_size
        self.col_blk_size = col_blk_size

    @abstractmethod
    def process_rtc_hdf5(self, input_list: list) -> Any:
        pass


class RTCReader(DataReader):
    def __init__(self, row_blk_size: int, col_blk_size: int):
        super().__init__(row_blk_size, col_blk_size)

    def process_rtc_hdf5(
            self,
            input_list: list,
            scratch_dir: str,
            mosaic_mode: str,
            mosaic_prefix: str,
            resamp_method: str,
            resamp_out_res: float,
            resamp_required: bool,
    ):

        """Read data from input HDF5s in blocks and generate mosaicked output
           Geotiff

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.
        scratch_dir: str
            Directory which stores the temporary files
        mosaic_mode: str
            Mosaic algorithm mode choice in 'average', 'first',
            or 'burst_center'
        mosaic_prefix: str
            Mosaicked output file name prefix
        resamp_required: bool
            Indicates whether resampling (downsampling) needs to be performed 
            on input RTC product in Geotiff.
        resamp_method: str
            Set GDAL.Warp() resampling algorithms based on its built-in options
            Default = 'nearest'
        resamp_out_res: float
            User define output resolution for resampled Geotiff
        """
        # Extract polarizations
        pols_rtc = self.extract_nisar_polarization(input_list)

        # Generate data paths
        data_path = self.generate_nisar_dataset_name(pols_rtc)

        # Generate layover mask path
        layover_mask_name = 'layoverShadowMask'
        layover_path = str(self.generate_nisar_layover_name(layover_mask_name))

        # Collect EPSG
        epsg_array, epsg_same_flag = self.get_nisar_epsg(input_list)

        # Write all RTC HDF5 inputs to intermeidate Geotiff first and re-use
        # existing functions to reproject data and create mosaicked output
        # from intermediate Geotiffs
        (
            geogrid_in,
            input_gtiff_list,
            layover_gtiff_list) = self.write_rtc_geotiff(
                input_list,
                scratch_dir,
                epsg_array,
                data_path,
                layover_path,
            )

        # Choose Resampling methods
        if resamp_required:
            # Apply multi-look technique
            if resamp_method == 'multilook':
                if len(input_gtiff_list) > 0:
                    for idx, input_geotiff in enumerate(input_gtiff_list):
                        self.multi_look_average(
                            input_geotiff,
                            scratch_dir,
                            resamp_out_res,
                            geogrid_in,
                        )
            else:
                # Apply resampling using GDAL.Warp() based on
                # resampling methods
                if len(input_gtiff_list) > 0:
                    for idx, input_geotiff in enumerate(input_gtiff_list):
                        self.resample_rtc(
			    input_geotiff,
			    scratch_dir,
			    resamp_out_res,
			    geogrid_in,
			    resamp_method,
		        )
            if len(layover_gtiff_list) > 0:
                layover_exist = True
                for idx, layover_geotiff in enumerate(layover_gtiff_list):
                    self.resample_rtc(
			layover_geotiff,
			scratch_dir,
			resamp_out_res,
			geogrid_in,
			'nearest',
		    )
            else:
                layover_exist = False
        else:    
            if len(layover_gtiff_list) > 0:
                layover_exist = True
            else:
                layover_exist = False

        # Mosaic intermediate geotiffs
        nlooks_list = []
        self.mosaic_rtc_geotiff(
            input_list,
            data_path,
            scratch_dir,
            geogrid_in,
            nlooks_list,
            mosaic_mode,
            mosaic_prefix,
            layover_exist,)    

    # Class functions
    def write_rtc_geotiff(
            self,
            input_list: list,
            scratch_dir: str,
            epsg_array: np.ndarray,
            data_path: list,
            layover_path: list,
    ):
        """ Create intermediate Geotiffs from a list of input RTCs

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.
        scratch_dir: str
            Directory which stores the temporary files
        epsg_array: array of int
            EPSG of each of the RTC input HDF5
        data_path: list
            RTC dataset path within the HDF5 input file
        layover_path: str
            layoverShadowMask layer dataset path

        Returns
        -------
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        output_gtiff_list: list
            List of RTC Geotiffs derived from input RTC HDF5.
        layover_gtiff_list: list
            List of layoverShadow Mask Geotiffs derived from input RTC HDF5.
        """

        # Reproject geotiff
        most_freq_epsg = majority_element(epsg_array)
        designated_value = np.float32(500)

        # List of written Geotiffs
        output_gtiff_list = []
        layover_gtiff_list = []

        # Create intermediate input Geotiffs
        for input_idx, input_rtc in enumerate(input_list):
            # Extract file names
            output_prefix = self.extract_file_name(input_rtc)

            # Read geotranform data
            geotransform, crs = self.read_geodata_hdf5(input_rtc)

            # Read metadata
            dswx_metadata_dict = self.read_metadata_hdf5(input_rtc)

            # Create Intermediate Geotiffs for each input GCOV file
            for path_idx, dataset_path in enumerate(data_path):
                data_name = Path(dataset_path).name[:2]
                dataset = f'HDF5:{input_rtc}:/{dataset_path}'
                output_gtiff = f'{scratch_dir}/{output_prefix}_{data_name}.tif'
                output_gtiff_list = np.append(output_gtiff_list, output_gtiff)

                h5_ds = gdal.Open(dataset, gdal.GA_ReadOnly)
                num_cols = h5_ds.RasterXSize
                num_rows = h5_ds.RasterYSize

                row_blk_size = self.row_blk_size
                col_blk_size = self.col_blk_size

                self.read_write_rtc(
                    h5_ds,
                    output_gtiff,
                    num_rows,
                    num_cols,
                    row_blk_size,
                    col_blk_size,
                    designated_value,
                    geotransform,
                    crs,
                    dswx_metadata_dict,
                    )

        geogrid_in = DSWXGeogrid()
        # Loop through EPSG (input files)
        for input_idx, input_rtc in enumerate(input_list):
            input_prefix = self.extract_file_name(input_rtc)
            # Check if the RTC has the same EPSG with the reference.
            if epsg_array[input_idx] != most_freq_epsg:
                for idx, dataset_path in enumerate(data_path):
                    data_name = Path(dataset_path).name[:2]
                    input_gtiff = \
                        f'{scratch_dir}/{input_prefix}_{data_name}.tif'
                    temp_gtiff = \
                        f'{scratch_dir}/{input_prefix}_temp_{data_name}.tif'

                    # Change EPSG
                    change_epsg_tif(
                        input_tif=input_gtiff,
                        output_tif=temp_gtiff,
                        epsg_output=most_freq_epsg,
                        output_nodata=255,
                    )

                    # Update geogrid
                    geogrid_in.update_geogrid(output_gtiff)

                    # Replace input file with output temp file
                    os.replace(temp_gtiff, input_gtiff)
            else:
                for idx, dataset_path in enumerate(data_path):
                    data_name = Path(dataset_path).name[:2]
                    output_gtiff = \
                        f'{scratch_dir}/{input_prefix}_{data_name}.tif'

                    # Update geogrid
                    geogrid_in.update_geogrid(output_gtiff)

        # Generate Layover Shadow Mask Intermediate Geotiffs
        for input_idx, input_rtc in enumerate(input_list):
            layover_data = f'HDF5:{input_rtc}:/{layover_path}'
            h5_layover = gdal.Open(layover_data, gdal.GA_ReadOnly)

            # Check if layoverShadowMask layer exists:
            if h5_layover is None:
                warnings.warn(f'\nDataset at {layover_data} does not exist or '
                              'cannot be opened.', RuntimeWarning)
                break

                output_prefix = self.extract_file_name(input_rtc)
                output_layover_gtiff = \
                    f'{scratch_dir}/{output_prefix}_layover.tif'
                layover_gtiff_list = np.append(layover_gtiff_list, output_layover_gtiff)

                num_cols = h5_layover.RasterXSize
                col_blk_size = self.col_blk_size

                self.read_write_rtc(
                    h5_layover,
                    output_layover_gtiff,
                    num_rows,
                    num_cols,
                    row_blk_size,
                    col_blk_size,
                    designated_value,
                    geotransform,
                    crs,
                    dswx_metadata_dict,
                )

                # Change EPSG of layOverMask if necessary
                if epsg_array[input_idx] != most_freq_epsg:
                    input_prefix = self.extract_file_name(input_rtc)
                    input_layover_gtiff = \
                        f'{scratch_dir}/{input_prefix}_layover.tif'
                    temp_layover_gtiff = \
                        f'{scratch_dir}/{input_prefix}_temp_layover.tif'

                    change_epsg_tif(
                        input_tif=input_layover_gtiff,
                        output_tif=temp_layover_gtiff,
                        epsg_output=most_freq_epsg,
                        output_nodata=255,
                    )

                    geogrid_in.update_geogrid(output_layover_gtiff)

                    # Replace input file with output temp file
                    os.replace(temp_layover_gtiff, input_layover_gtiff)
                else:
                    geogrid_in.update_geogrid(output_layover_gtiff)

        return geogrid_in, output_gtiff_list, layover_gtiff_list

    def mosaic_rtc_geotiff(
        self,
        input_list: list,
        data_path: list,
        scratch_dir: str,
        geogrid_in: DSWXGeogrid,
        nlooks_list: list,
        mosaic_mode: str,
        mosaic_prefix: str,
        layover_exist: bool,
    ):
        """ Create mosaicked output Geotiff from a list of input RTCs

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.
        data_path: list
            RTC dataset path within the HDF5 input file
        scratch_dir: str
            Directory which stores the temporary files
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        nlooks_list: list
            List of the nlooks raster that corresponds to list_rtc
        mosaic_mode: str
            Mosaic algorithm mode choice in 'average', 'first',
            or 'burst_center'
        mosaic_prefix: str
            Mosaicked output file name prefix
        layover_exist: bool
            Boolean which indicates if a layoverShadowMask layer
            exists in input RTC
        """
        for idx, dataset_path in enumerate(data_path):
            data_name = Path(dataset_path).name[:2]
            input_gtiff_list = []
            for input_idx, input_rtc in enumerate(input_list):
                input_prefix = self.extract_file_name(input_rtc)
                input_gtiff = f'{scratch_dir}/{input_prefix}_{data_name}.tif'
                input_gtiff_list = np.append(input_gtiff_list, input_gtiff)

            # Mosaic dataset of same polarization into a single Geotiff
            output_mosaic_gtiff = \
                f'{scratch_dir}/{mosaic_prefix}_{data_name}.tif'
            mosaic_single_output_file(
                input_gtiff_list,
                nlooks_list,
                output_mosaic_gtiff,
                mosaic_mode,
                scratch_dir=scratch_dir,
                geogrid_in=geogrid_in,
                temp_files_list=None,
                )

        # Mosaic layover shadow mask
        if layover_exist:
            layover_gtiff_list = []
            for input_idx, input_rtc in enumerate(input_list):
                input_prefix = self.extract_file_name(input_rtc)
                layover_gtiff = f'{scratch_dir}/{input_prefix}_layover.tif'
                layover_gtiff_list = np.append(layover_gtiff_list,
                                               layover_gtiff)

            layover_mosaic_gtiff = f'{scratch_dir}/{mosaic_prefix}_layover.tif'

            mosaic_single_output_file(
                layover_gtiff_list,
                nlooks_list,
                layover_mosaic_gtiff,
                mosaic_mode,
                scratch_dir=scratch_dir,
                geogrid_in=geogrid_in,
                temp_files_list=None,
            )

    def resample_rtc(
        self,
        input_geotiff: str,
        scratch_dir: str,
        output_res: float,
        geogrid_in: DSWXGeogrid,
        resamp_method: str = 'nearest',
    ):
        """Resample input geotfif from their native resolution to desired
        output resolution

        Parameters
        ----------
        input_geotiff: str
            Input geotiff path to be resampled.
        scratch_dir: str
            Directory which stores the temporary files
        output_res: float
            User define output resolution for resampled Geotiff
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        resamp_method: str
            Set GDAL.Warp() resampling algorithms based on its built-in options
            Default = 'nearest
        """

        # Check if the file exists
        if not os.path.exists(input_geotiff):
            raise FileNotFoundError(f"The file '{input_geotiff}' does not exist.")

        full_path = Path(input_geotiff)
        output_geotiff = f'{full_path.parent}/{full_path.stem}_resamp.tif'

        ds_input = gdal.Open(input_geotiff)
        geotransform = ds_input.GetGeoTransform()

        # Set GDAL Warp options
        # Resampling method
        #gdal.GRA_Bilinear, gdal.GRA_NearestNeighbour, gdal.GRA_Cubic, gdal.GRA_Average, etc

        options = gdal.WarpOptions(
            xRes=output_res,
            yRes=output_res,
            resampleAlg=resamp_method)

        ds_output = gdal.Warp(output_geotiff, ds_input, options=options)

        # Update Geogrid in output Geotiff and replace the input Geotiff with it
        geogrid_in.update_geogrid(output_geotiff)
        os.replace(output_geotiff, input_geotiff)

        ds_input = None
        ds_output = None


    def multi_look_average(
        self,
        input_geotiff: str,
        scratch_dir: str,
        output_res: float,
        geogrid_in: DSWXGeogrid,
    ):
        """Apply upsampling and multi-look pixel averaging on input geotfif 
        to obtain Geotiff with desired output resolution

        Parameters
        ----------
        input_geotiff: str
            Input geotiff path to be resampled.
        scratch_dir: str
            Directory which stores the temporary files
        output_res: float
            User define output resolution for multi-looked Geotiff
        geogrid_in: DSWXGeogrid object
            A dataclass object  representing the geographical grid
            configuration for an RTC (Radar Terrain Correction) run.
        """

        ds_input = gdal.Open(input_geotiff)
        geotransform_input = ds_input.GetGeoTransform()

        input_width = ds_input.RasterXSize
        input_length = ds_input.RasterYSize

        input_res_x = np.abs(geotransform_input[1])
        input_res_y = np.abs(geotransform_input[5])

        if input_res_x != input_res_y:
            raise ValueError(
                "x and y resolutions of the input must be the same."
            )
        
        full_path = Path(input_geotiff)
        output_geotiff = f'{full_path.parent}/{full_path.stem}_multi_look.tif'

        # Multi-look parameters
        interm_upsamp_res = math.gcd(int(input_res_x), int(output_res))
        downsamp_ratio = output_res // interm_upsamp_res  # ratio = 3
        normalized_flag = True

        if input_res_x == 20:
            # Perform upsampling to 10 meter resolution
            # Compute upsampled data output bounds
            upsamp_bounds = _calculate_output_bounds(
                geotransform_input,
                input_width,
                input_length,
                interm_upsamp_res, 
            )

            # Perform GDAL.warp() in memory for upsampled data 
            warp_options = gdal.WarpOptions(
                xRes=interm_upsamp_res,
                yRes=-interm_upsamp_res,
                outputBounds=upsamp_bounds,
                resampleAlg='nearest',
                format='MEM'  # Use memory as the output format
            )

            ds_upsamp = gdal.Warp('', ds_input, options=warp_options)
            data_upsamp = ds_upsamp.GetRasterBand(1).ReadAsArray()
            geotransform_upsamp = ds_upsamp.GetGeoTransform()
            projection_upsamp = ds_upsamp.GetProjection()

            # Aggregate pixel values in a image to lower resolution to achieve
            # multi-looking effect
            multi_look_output = _aggregate_10m_to_30m_conv(
                data_upsamp,
                downsamp_ratio,
                normalized_flag,
            ) 

            # Write multi-look averaged data to output geotiff
            self.write_array_to_geotiff(
                ds_upsamp,
                multi_look_output,
                upsamp_bounds,
                output_res,
                output_geotiff,
            )    
        elif input_res_x == 10:
            # Directly average 10m resolution input to 30m resolution output
            ds_array = ds_input.GetRasterBand(1).ReadAsArray()
            multi_look_output = _aggregate_10m_to_30m_conv(
                ds_array,
                downsamp_ratio,
                normalized_flag,
            ) 
            # Write to output geotiff
            self.write_array_to_geotiff(
                ds_input,
                multi_look_output,
                upsamp_bounds,
                output_res,
                output_geotiff,
            )    
        else:
            raise ValueError("Input RTC are expected to have only 10m or 20m resolutions.")


        # Update Geogrid in output Geotiff and replace the input Geotiff with it
        geogrid_in.update_geogrid(output_geotiff)
        os.replace(output_geotiff, input_geotiff)


    def write_array_to_geotiff(
        self,
        ds_input,
        output_data,
        output_bounds,
        output_res,
        output_geotiff,
    ):
        """Create output geotiff using gdal.CreateCopy()

        Parameters
        ----------
        ds_input: gdal dataset
            input dataset opened by GDAL
        output_data: numpy.ndarray
            output_data to be written into output geotiff
        output_bounds: list
            The bounding box coordinates where the output will be clipped.
        output_res: float
            User define output resolution for resampled Geotiff
        downsamp_ratio: int
            downsampling factor for dataset in the input geotiff
        output_geotiff: str
            Output geotiff to be created.
        """

        # Update Geotransformation
        geotransform_input = ds_input.GetGeoTransform()
        geotransform_output = (
            geotransform_input[0],                        
            output_res,
            geotransform_input[2],                        
            geotransform_input[3],                        
            geotransform_input[4],                        
            output_res,
        )  

        # Calculate the output raster size
        output_y_size, output_x_size = output_data.shape

        driver = gdal.GetDriverByName('GTiff')
        ds_output = driver.Create(
            output_geotiff, 
            output_x_size, 
            output_y_size, 
            ds_input.RasterCount, 
            gdal.GDT_Float32
        )

        # Set Geotransform and Projection
        ds_output.SetGeoTransform(geotransform_output)
        ds_output.SetProjection(ds_input.GetProjection())

        # Write output data to raster
        ds_output.GetRasterBand(1).WriteArray(output_data)

        ds_input = None
        ds_output = None


    def extract_file_name(self, input_rtc):
        """Extract file name identifier from input file name

        Parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        file_name: str
            file name identifier
        """

        # Check if the file exists
        if not os.path.exists(input_rtc):
            raise FileNotFoundError(f"The file '{input_rtc}' does not exist.")

        file_name = Path(input_rtc).stem.split('-')[0]

        return file_name

    def extract_nisar_polarization(self, input_list):
        """Extract input RTC dataset polarizations

        Parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.

        Returns
        -------
        polarizations: list of str
            All dataset polarizations listed in the input HDF5 file
        """

        pol_list_path = \
            '/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations'
        polarizations = []
        pols_rtc = []
        for input_idx, input_rtc in enumerate(input_list):
            print(input_rtc)
            # Check if the file exists
            if not os.path.exists(input_rtc):
                raise FileNotFoundError(
                    f"The file '{input_rtc}' does not exist.")
            with h5py.File(input_rtc, 'r') as src_h5:
                pols = np.sort(src_h5[pol_list_path][()])
                if len(polarizations) == 0:
                    polarizations = pols.copy()
                elif not np.all(polarizations == pols):
                    raise ValueError(
                        "Polarizations of multiple RTC files "
                        "are not consistent.")

        for pol_idx, pol in enumerate(polarizations):
            pols_rtc = np.append(pols_rtc, pol.decode('utf-8'))

        return pols_rtc

    def generate_nisar_dataset_name(self, data_name: str | list[str]):
        """Generate dataset paths

        Parameters
        ----------
        data_name: str or list of str
            All dataset polarizations listed in the input HDF5 file

        Returns
        -------
        data_path: np.ndarray of str
            RTC dataset path within the HDF5 input file
        """

        if isinstance(data_name, str):
            data_name = [data_name]

        group = '/science/LSAR/GCOV/grids/frequencyA/'
        data_path = []
        for name_idx, dname in enumerate(data_name):
            data_path = np.append(data_path, f'{group}{dname * 2}')

        return data_path

    def generate_nisar_layover_name(self, layover_name: str):
        """Generate layOverShadowMask dataset path

        Parameters
        ----------
        layover_name: str
            Name of layover and shadow Mask layer in the input HDF5 file

        Returns
        -------
        data_path: str
            RTC dataset path within the HDF5 input file
        """
        group = '/science/LSAR/GCOV/grids/frequencyA/'

        data_path = f'{group}{layover_name}'

        return data_path

    def get_nisar_epsg(self, input_list):
        """extract data from RTC Geo information and store it as a dictionary

        parameters
        ----------
        input_list: list
            The HDF5 file paths of input RTCs to be mosaicked.

        Returns
        -------
        epsg_array: array of int
            EPSG of each of the RTC input HDF5
        epsg_same_flag: bool
            A flag which indicates whether all input EPSG are the same
            if True, all input EPSG are the same and vice versa.
        """
        proj = '/science/LSAR/GCOV/grids/frequencyA/projection'

        epsg_array = np.zeros(len(input_list), dtype=int)
        for input_idx, input_rtc in enumerate(input_list):
            with h5py.File(input_rtc, 'r') as src_h5:
                epsg_array[input_idx] = src_h5[proj][()]

        if (epsg_array == epsg_array[0]).all():
            epsg_same_flag = True
        else:
            epsg_same_flag = False

        return epsg_array, epsg_same_flag

    def read_write_rtc(
        self,
        h5_ds: Dataset,
        output_gtiff,
        num_rows: int,
        num_cols: int,
        row_blk_size: int,
        col_blk_size: int,
        designated_value: np.float32,
        geotransform: Affine,
        crs: str,
        dswx_metadata_dict: dict):
        """Read an level-2 RTC prodcut in HDF5 format and writ it out in
        GeoTiff format in data blocks defined by row_blk_size and col_blk_size.

        Parameters
        ----------
        h5_ds: GDAL Dataset
            GDAL dataset object to be processed
        output_gtiff: str
        Output Geotiff file path and name
            num_rows: int
        The number of rows (height) of the output Geotiff.
            num_cols: int
        The number of columns (width) of the output Geotiff.
            row_blk_size: int
        The number of rows to read each time from the dataset.
            col_blk_size: int
        The number of columns to read each time from the dataset
        designated_value: np.float32
            Identify Inf in the dataset and replace them with
            a designated value
        geotransform: Affine Transformation object
            Transformation matrix which maps pixel locations in (row, col)
            coordinates to (x, y) spatial positions.
        crs: str
            Coordinate Reference System object in EPSG representation
        dswx_metadata_dict: dictionary
            This dictionary metadata extracted from input RTC
        """
        row_blk_size = self.row_blk_size
        col_blk_size = self.col_blk_size

        with rasterio.open(
            output_gtiff,
            'w',
            driver='GTiff',
            height=num_rows,
            width=num_cols,
            count=1,
            dtype='float32',
            crs=crs,
            transform=geotransform,
            compress='DEFLATE',
        ) as dst:
            for idx_y, slice_row in enumerate(slice_gen(num_rows, row_blk_size)):
                row_slice_size = slice_row.stop - slice_row.start
                for idx_x, slice_col in enumerate(slice_gen(num_cols, col_blk_size)):
                    col_slice_size = slice_col.stop - slice_col.start

                    ds_blk = h5_ds.ReadAsArray(
                        slice_col.start,
                        slice_row.start,
                        col_slice_size,
                        row_slice_size,
                    )

                    # Replace Inf values with a designated value: 500
                    ds_blk[np.isinf(ds_blk)] = designated_value
                    ds_blk[ds_blk > designated_value] = designated_value
                    ds_blk[ds_blk == 0] = np.nan

                    dst.write(
                        ds_blk,
                        1,
                        window=Window(
                            slice_col.start,
                            slice_row.start,
                            col_slice_size,
                            row_slice_size
                        )
                    )

            dst.update_tags(**dswx_metadata_dict)

    def read_geodata_hdf5(self, input_rtc):
        """extract data from RTC Geo information and store it as a dictionary

        parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        geotransform: Affine Transformation object
            Transformation matrix which maps pixel locations in (row, col)
            coordinates to (x, y) spatial positions.
        crs: str
            Coordinate Reference System object in EPSG representation
        """
        frequency_a_path = '/science/LSAR/GCOV/grids/frequencyA'
        geo_name_mapping = {
            'xcoord': f'{frequency_a_path}/xCoordinates',
            'ycoord': f'{frequency_a_path}/yCoordinates',
            'xposting': f'{frequency_a_path}/xCoordinateSpacing',
            'yposting': f'{frequency_a_path}/yCoordinateSpacing',
            'proj': f'{frequency_a_path}/projection'
        }

        with h5py.File(input_rtc, 'r') as src_h5:
            xmin = src_h5[f"{geo_name_mapping['xcoord']}"][:][0]
            ymin = src_h5[f"{geo_name_mapping['ycoord']}"][:][0]
            xres = src_h5[f"{geo_name_mapping['xposting']}"][()]
            yres = src_h5[f"{geo_name_mapping['yposting']}"][()]
            epsg = src_h5[f"{geo_name_mapping['proj']}"][()]

        # Geo transformation
        geotransform = Affine.translation(
            xmin - xres/2, ymin - yres/2) * Affine.scale(xres, yres)

        # Coordinate Reference System
        crs = f'EPSG:{epsg}'

        return geotransform, crs

    def read_metadata_hdf5(self, input_rtc):
        """Read NISAR Level-2 GCOV metadata

        Parameters
        ----------
        input_rtc: str
            The HDF5 RTC input file path

        Returns
        -------
        dswx_metadata_dict: dictionary
            RTC metadata dictionary. Will be written into output GeoTIFF.

        """
        id_path = '/science/LSAR/identification'
        meta_path = '/science/LSAR/GCOV/metadata'
        # Metadata Name Dictionary
        dswx_meta_mapping = {
            'RTC_ORBIT_PASS_DIRECTION': f'{id_path}/orbitPassDirection',
            'RTC_LOOK_DIRECTION': f'{id_path}/lookDirection',
            'RTC_PRODUCT_VERSION': f'{id_path}/productVersion',
            'RTC_SENSING_START_TIME': f'{id_path}/zeroDopplerStartTime',
            'RTC_SENSING_END_TIME': f'{id_path}/zeroDopplerEndTime',
            'RTC_FRAME_NUMBER': f'{id_path}/frameNumber',
            'RTC_TRACK_NUMBER': f'{id_path}/trackNumber',
            'RTC_ABSOLUTE_ORBIT_NUMBER': f'{id_path}/absoluteOrbitNumber',
            'RTC_INPUT_L1_SLC_GRANULES':
                f'{meta_path}/processingInformation/inputs/l1SlcGranules',
            }

        with h5py.File(input_rtc, 'r') as src_h5:
            orbit_pass_dir = src_h5[
                dswx_meta_mapping['RTC_ORBIT_PASS_DIRECTION']][()].decode()
            look_dir = src_h5[
                dswx_meta_mapping['RTC_LOOK_DIRECTION']][()].decode()
            prod_ver = src_h5[
                dswx_meta_mapping['RTC_PRODUCT_VERSION']][()].decode()
            zero_dopp_start = src_h5[
                dswx_meta_mapping['RTC_SENSING_START_TIME']][()].decode()
            zero_dopp_end = src_h5[
                dswx_meta_mapping['RTC_SENSING_END_TIME']][()].decode()
            frame_number = src_h5[
                dswx_meta_mapping['RTC_FRAME_NUMBER']][()]
            track_number = src_h5[
                dswx_meta_mapping['RTC_TRACK_NUMBER']][()]
            abs_orbit_number = src_h5[
                dswx_meta_mapping['RTC_ABSOLUTE_ORBIT_NUMBER']][()]
            try:
                input_slc_granules = src_h5[
                    dswx_meta_mapping['RTC_INPUT_L1_SLC_GRANULES']][(0)].decode()
            except:
                print('RTC_INPUT_L1_SLC_GRANULES is not available')
        dswx_metadata_dict = {
            'ORBIT_PASS_DIRECTION': orbit_pass_dir,
            'LOOK_DIRECTION': look_dir,
            'PRODUCT_VERSION': prod_ver,
            'ZERO_DOPPLER_START_TIME': zero_dopp_start,
            'ZERO_DOPPLER_END_TIME': zero_dopp_end,
            'FRAME_NUMBER': frame_number,
            'TRACK_NUMBER': track_number,
            'ABSOLUTE_ORBIT_NUMBER': abs_orbit_number,
        }

        return dswx_metadata_dict


def slice_gen(total_size: int,
              batch_size: int,
              combine_rem: bool = True) -> Iterator[slice]:
    """Generate slices with size defined by batch_size.

    Parameters
    ----------
    total_size: int
        size of data to be manipulated by slice_gen
    batch_size: int
        designated data chunk size in which data is sliced into.
    combine_rem: bool
        Combine the remaining values with the last complete block if 'True'.
        If False, ignore the remaining values
        Default = 'True'

    Yields
    ------
    slice: slice obj
        Iterable slices of data with specified input batch size,
        bounded by start_idx and stop_idx.
    """
    num_complete_blks = total_size // batch_size
    num_total_complete = num_complete_blks * batch_size
    num_rem = total_size - num_total_complete

    if combine_rem and num_rem > 0:
        for start_idx in range(0, num_total_complete - batch_size, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)

        last_blk_start = num_total_complete - batch_size
        last_blk_stop = total_size
        yield slice(last_blk_start, last_blk_stop)
    else:
        for start_idx in range(0, num_total_complete, batch_size):
            stop_idx = start_idx + batch_size
            yield slice(start_idx, stop_idx)


def run(cfg):
    """Generate mosaic workflow with user-defined args stored
    in dictionary runconfig 'cfg'

    Parameters:
    -----------
    cfg: RunConfig
        RunConfig object with user runconfig options
    """

    # Mosaicking parameters
    processing_cfg = cfg.groups.processing

    input_list = cfg.groups.input_file_group.input_file_path

    mosaic_cfg = processing_cfg.mosaic
    mosaic_mode = mosaic_cfg.mosaic_mode
    mosaic_prefix = mosaic_cfg.mosaic_prefix

    resamp_required = mosaic_cfg.resamp_required
    resamp_method = mosaic_cfg.resamp_method
    resamp_out_res = mosaic_cfg.resamp_out_res

    scratch_dir = cfg.groups.product_path_group.scratch_path
    os.makedirs(scratch_dir, exist_ok=True)

    row_blk_size = mosaic_cfg.read_row_blk_size
    col_blk_size = mosaic_cfg.read_col_blk_size

    # Create reader object
    reader = RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )

    # Mosaic input RTC into output Geotiff
    reader.process_rtc_hdf5(
        input_list,
        scratch_dir,
        mosaic_mode,
        mosaic_prefix,
        resamp_method,
        resamp_out_res,
        resamp_required,
    )


