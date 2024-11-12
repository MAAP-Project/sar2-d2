import sys
import os
import logging
import time
import zipfile
from io import TextIOWrapper

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
from osgeo import gdal
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

from dist import (reader_ni,
                  util,
                  filter_SAR,
                  generate_log)
from dist.dist_runconfig import (_get_parser,
                                 RunConfig,)


logger = logging.getLogger('sar2-d2')


def get_rtc_stack_block(filename_list, blocksize, block_ind, scale='db'):
    refind = 0

    # define basic frame from first scene
    rtc_path = filename_list[refind]
    tif_ds = gdal.Open(rtc_path, gdal.GA_ReadOnly)
    refrows = tif_ds.RasterYSize
    refcols = tif_ds.RasterXSize
    tif_ds = None
    del tif_ds

    nblocks = int(np.ceil(refrows / blocksize))
    number_scene = len(filename_list)

    ref_lat_rtc, ref_lon_rtc = read_tif_latlon(filename_list[refind])

    # measure georeferenced offset in range and azimuth
    offset_x = []
    offset_y = []
    for find, fname in enumerate(filename_list):
        lat_rtc, lon_rtc = read_tif_latlon(filename_list[1])

        lon_spacing = np.abs(lon_rtc[1] - lon_rtc[2])
        lat_spacing = np.abs(lat_rtc[1] - lat_rtc[2])
        lon_rtc_diff = lon_rtc[1] - ref_lon_rtc[1]
        lat_rtc_diff = lat_rtc[1] - ref_lat_rtc[1]

        # print(lon_rtc_diff, lon_rtc_diff / lon_spacing)
        offset_x.append((lon_rtc_diff / lon_spacing))
        offset_y.append((lat_rtc_diff / lat_spacing))

    # loop for secondary RTCs
    avg_raster_whole = np.empty([refrows, refcols])
    std_raster_whole = np.empty([refrows, refcols])
    max_raster_whole = np.empty([refrows, refcols])
    avg_raster_whole[:] = np.nan
    std_raster_whole[:] = np.nan
    max_raster_whole[:] = np.nan
    # temp_raster = np.ones([refrows, refcols, number_scene])
    block = block_ind


    row_start = block * blocksize
    row_end = row_start + blocksize

    if (row_end > refrows):
        row_end = refrows
        block_rows_data = row_end - row_start

    else:
        block_rows_data = blocksize

    print("-- reading block: ", block, row_start, row_end, block_rows_data)

    base_ul_x = np.min(ref_lon_rtc)
    base_ul_y = ref_lat_rtc[row_start]
    base_lr_x = np.max(ref_lon_rtc)
    base_lr_y = ref_lat_rtc[row_end]

    target_rtc_set = np.empty([
        block_rows_data,
        refcols,
        number_scene],
        dtype=float)

    for find, fname in enumerate(filename_list):
        print('file reading', find, fname)
        lat_rtc, lon_rtc = read_tif_latlon(fname)

        src_tif = gdal.Open(fname, gdal.GA_ReadOnly)
        # x and y coordinates for reference raster
        target_ul_x_ind = (np.abs(lon_rtc - base_ul_x)).argmin()
        target_ul_y_ind = (np.abs(lat_rtc - base_ul_y)).argmin()

        target_lr_x_ind = (np.abs(lon_rtc - base_lr_x)).argmin()
        target_lr_y_ind = (np.abs(lat_rtc - base_lr_y)).argmin()


        row_sub = target_lr_y_ind - target_ul_y_ind
        col_sub = target_lr_x_ind - target_ul_x_ind

        target_rtc_image = np.empty([row_sub,
                                        col_sub],
                                        dtype=float)
        band = src_tif.GetRasterBand(1)
        target_rtc_image = band.ReadAsArray(int(target_ul_x_ind),
            int(target_ul_y_ind),
            int(col_sub), int(row_sub))

        off_x_start = np.round(
            (lon_rtc[target_ul_x_ind] - base_ul_x)/lon_spacing)
        off_x_end = np.round(
            (lon_rtc[target_lr_x_ind] - base_lr_x)/lon_spacing)
        off_y_start = -np.round(
            (lat_rtc[target_ul_y_ind] - base_ul_y )/lat_spacing)
        off_y_end = -np.round(
            (lat_rtc[target_lr_y_ind] - base_lr_y)/lat_spacing)

        if off_y_start < 0:
            off_y_start = 0
        if off_y_end > 0:
            off_y_end = 0
        if off_x_start < 0:
            off_x_start = 0
        if off_x_end > 0:
            off_x_end = 0

        src_tif = None
        del src_tif
        band = None
        del band
        image_rows, image_cols = np.shape(target_rtc_image)
        if scale == 'db':
            target_rtc_image[target_rtc_image<-30] = np.nan
            target_rtc_set[
                int(off_y_start):int(image_rows + off_y_start),
                int(off_x_start):int(off_x_start + image_cols),
                find] = 10 * np.log10(target_rtc_image)
        else:
            target_rtc_image[target_rtc_image<(10**-10)] = np.nan
            target_rtc_set[
                int(off_y_start):int(image_rows + off_y_start),
                int(off_x_start):int(off_x_start + image_cols),
                find] = target_rtc_image

    return target_rtc_set


def read_tif_latlon(intput_tif_str):
    #  Initialize the Image Size
    ds = gdal.Open(intput_tif_str)


    #get the point to transform, pixel (0,0) in this case
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]

    xres = (maxx - minx) / float(width)
    yres = (maxy - miny) / float(height)
    #get the coordinates in lat long
    lat = np.linspace(maxy, miny, height+1)
    lon = np.linspace(minx, maxx, width+1)

    ds = None
    del ds  # close the dataset (Python object and pointers)

    return lat, lon


# Function to process each row in parallel
def process_row(row, polarizations, filter_option, block_param):
    acqtime = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])
    polarization_stack = []
    actual_pol = []
    for polarization in polarizations:
        file_path = row["geotiff_co"] if polarization in ['HH', 'VV'] else row["geotiff_cross"]
        if file_path is not None:
            if os.path.exists(file_path):
                subset_data = util.get_raster_block(file_path, block_param)

                despeckled = filter_SAR.tv_bregman(subset_data, **filter_option)
                despeckled_db = 10 * np.log10(despeckled)

                # Create an xarray DataArray from the subsetted data
                da = xr.DataArray(despeckled_db, dims=['y', 'x'],
                                coords={'y': np.arange(subset_data.shape[0]),
                                        'x': np.arange(subset_data.shape[1])})
                # Add time dimension to the DataArray
                da = da.expand_dims(time=pd.Index([acqtime], name='time'))
                polarization_stack.append(da)
                actual_pol.append(polarization)
    # Concatenate along 'polarization' dimension
    da_polarized = xr.concat(polarization_stack, dim=pd.Index(actual_pol, name='polarization'))
    return da_polarized


def run_bootstrap(args):
    """Wrapper to call bootstrap_trial with arguments."""
    return bootstrap_trial(*args)


def bootstrap_trial(qmetric_residuals, dmask, Sdiff_random_max):
    """Single bootstrap iteration."""
    # Shuffle time axis for permutation
    permutation = np.random.permutation(len(qmetric_residuals))
    Srandom = qmetric_residuals[permutation].cumsum('time')
    Srandom = Srandom.where(dmask)
    Srandom_max = Srandom.max('time')
    Srandom_min = Srandom.min('time')
    Sdiff_random = Srandom_max - Srandom_min

    # Mask where Sdiff_random is greater than Sdiff_random_max
    gtmask = np.ma.greater(Sdiff_random, Sdiff_random_max).data

    return Sdiff_random, gtmask


def convert_HDF5_to_GTIFF(t, row, reader, scratch_dir,
                          mosaic_mode, resamp_method,
                          resamp_out_res, resamp_required):
    ddd = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])
    formatted_time = ddd.strftime('%Y%m%d')

    # Mosaic input RTC into output Geotiff (uncomment if necessary)
    reader.process_rtc_hdf5(
        input_list=[row['rtc_path']],
        scratch_dir=scratch_dir,
        mosaic_mode=mosaic_mode,
        mosaic_prefix=formatted_time,
        resamp_method=resamp_method,
        resamp_out_res=resamp_out_res,
        resamp_required=resamp_required,
    )

    # Define file paths for HH and HV polarizations
    output_filename_hh = f'{scratch_dir}/{formatted_time}_HH.tif'
    output_filename_hv = f'{scratch_dir}/{formatted_time}_HV.tif'

    # Check if files exist and store results
    result = {'t': t, 'geotiff_co': None, 'geotiff_cross': None}
    if os.path.isfile(output_filename_hh):
        result['geotiff_co'] = output_filename_hh
    if os.path.isfile(output_filename_hv):
        result['geotiff_cross'] = output_filename_hv

    return result


def dist_workflow(cfg):

    import time
    t_all = time.time()

    input_gcov_list = cfg.groups.input_file_group.input_file_path
    scratch_dir = cfg.groups.product_path_group.scratch_path
    output_dir = cfg.groups.product_path_group.sas_output_path
    proc_param = cfg.groups.processing_parameters

    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    block_length = 300
    forest_cross_thres = 0.008 # -20 dB
    forest_cross_thres = proc_param.forest_mask_threshold #dB
    resamp_out_res = cfg.groups.product_path_group.output_spacing

    filter_option = {'lambda_value': proc_param.filter_lambda}

    polarizations = util.extract_nisar_polarization(input_gcov_list)
    date_str_list = []
    data_stack = []\

    for input_h5 in input_gcov_list:
        # Find HDF5 metadata

        rtc_metadata = util.read_metadata_hdf5(input_h5)
        rtc_metadata['rtc_path'] = input_h5
        # tags = src.tags(0)
        # date_str = tags['ZERO_DOPPLER_START_TIME']
        track_number = rtc_metadata['TRACK_NUMBER']
        date_str_list.append(rtc_metadata['ZERO_DOPPLER_START_TIME'])
        data_stack.append(rtc_metadata)

    data_stack_df = pd.DataFrame(data_stack)

    mosaic_mode = 'first'
    mosaic_prefix = 'first'

    resamp_required = True
    resamp_method = 'nearest'

    row_blk_size = 300
    col_blk_size = 300

    # Create reader object
    reader = reader_ni.RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )
    data_stack_df['geotiff_co'] = ""
    data_stack_df['geotiff_cross'] = ""

    # 1) Convert GCOV HDF5 to Geotiff
    # Use ProcessPoolExecutor for parallel processing
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                convert_HDF5_to_GTIFF, t, row, reader, scratch_dir, mosaic_mode, resamp_method, resamp_out_res, resamp_required
            )
            for t, row in data_stack_df.iterrows()
        ]

        # Collect results as they are completed
        for future in futures:
            results.append(future.result())

    # Update data_stack_df with results
    for result in results:
        data_stack_df.at[result['t'], 'geotiff_co'] = result['geotiff_co']
        data_stack_df.at[result['t'], 'geotiff_cross'] = result['geotiff_cross']

    # 2) Resample GeoTiff to have same bounding boxes.
    geogrid_in = reader_ni.DSWXGeogrid()

    for t, row in data_stack_df.iterrows():
        geogrid_in.update_geogrid(row['geotiff_co'])

    for t, row in data_stack_df.iterrows():
        ddd = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])

        # Format the datetime to 'yyyymmdd'
        formatted_time = ddd.strftime('%Y%m%d')
        output_filename = f'{scratch_dir}/{formatted_time}_HH_resampled.tif'
        util.resample_and_crop_with_gdalwarp(
            row['geotiff_co'],
            output_filename,
            geogrid_in.start_x,
            geogrid_in.start_y,
            geogrid_in.end_x,
            geogrid_in.end_y,
            geogrid_in.spacing_x,
            geogrid_in.spacing_y,
            geogrid_in.epsg)
        if os.path.isfile(output_filename):
            data_stack_df.at[t, 'geotiff_co'] = output_filename
        if row['geotiff_cross'] is not None:
            if os.path.exists(row['geotiff_cross']):
                output_filename = f'{scratch_dir}/{formatted_time}_HV_resampled.tif'
                util.resample_and_crop_with_gdalwarp(
                    row['geotiff_cross'],
                    output_filename,
                    geogrid_in.start_x,
                    geogrid_in.start_y,
                    geogrid_in.end_x,
                    geogrid_in.end_y,
                    geogrid_in.spacing_x,
                    geogrid_in.spacing_y,
                    geogrid_in.epsg)
                if os.path.isfile(output_filename):
                    data_stack_df.at[t, 'geotiff_cross'] = output_filename
    image_meta = util.get_meta_from_tif(data_stack_df.iloc[0]['geotiff_co'])

    pad_shape = (0, 0)
    block_params = util.block_param_generator(
        lines_per_block=block_length,
        data_shape=(geogrid_in.length,
                    geogrid_in.width),
        pad_shape=pad_shape)


    for block_ind, block_param in enumerate(block_params):
        print(f'Processing block {block_ind}')
        row_start = block_param.read_start_line
        row_num = block_param.read_start_line + block_param.read_length
        pixel_window = Window(row_start,
                            0,
                            row_num,
                            block_param.data_width)
        da_stack = []
        # Apply despeckle filter for images
        # Use ThreadPoolExecutor for parallel processing of rows
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_row, row, polarizations, filter_option, block_param)
                for _, row in data_stack_df.iterrows()
            ]
            for future in futures:
                da_stack.append(future.result())

        ds = xr.concat(da_stack, dim='time')
        sorted_ds = ds.sortby('time')
        da_min_vh = sorted_ds.min('time').sel(polarization='HH')

        fmask = (da_min_vh > forest_cross_thres).persist()
        if proc_param.debug_mode:
            fmask_path_name = f'{scratch_dir}/forest_mask_binary.tif'

            util.write_raster_block(
                    fmask_path_name,
                    np.squeeze(fmask.values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)

        X = sorted_ds.where(fmask == 1)
        Xmean_maxmin = ( X.max(dim='time') + X.min(dim='time') ) / 2.
        R = X - Xmean_maxmin
        S = R.fillna(0).cumsum(dim='time')

        # mask out invalid values
        S_masked = S.where(S != 0)

        Sdiff = S_masked.max(dim='time', skipna=True) - S_masked.min(dim='time', skipna=True)

        for pol_single in polarizations:
            if proc_param.debug_mode:

                change_path_name = f'{scratch_dir}/cumsum_maximum_{pol_single}.tif'
                print(f'saving {change_path_name}')
                util.write_raster_block(
                    change_path_name,
                    np.squeeze(Sdiff.sel(polarization=pol_single).values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    datatype='float32',
                    cog_flag=True)

            # Loop over each time step in R
            for time_step in R['time']:
                # Extract data for the specific date
                S_date = S.sel(time=time_step).sel(polarization=pol_single)

                # Construct the file name using the time step
                date_str = pd.to_datetime(time_step.values).strftime('%Y%m%d')

                if proc_param.debug_mode:
                    output_path = f'{scratch_dir}/S_{date_str}_{pol_single}.tif'
                    print(output_path)
                    # Save the data to a GeoTIFF file
                    util.write_raster_block(
                        output_path,
                        np.squeeze(S_date.values),
                        block_param=block_param,
                        geotransform=image_meta['geotransform'],
                        projection=image_meta['projection'],
                        datatype='float32',  # or another appropriate datatype
                        cog_flag=True
                    )

            Sdiff = Sdiff.where(Sdiff > 0)
            Sdiff = Sdiff.where(S.max(dim='time')>0)  # Only care for positive values in Smax

            quantile = 85
            #arr = Sdiff.values.flatten()
            if proc_param.threshold is None:
                dthres = np.nanpercentile(Sdiff.sel(polarization=pol_single).values.flatten(), quantile)
                print(f'Setting the threshold for SDiff based on the {quantile}th percentile to: {dthres:.3f}')

            else:
                dthres = proc_param.threshold
                print(f'Setting the threshold for User-defined value: {dthres:.3f}')

            dmask = Sdiff > dthres
            Sdiff = Sdiff.where(dmask)

            if proc_param.debug_mode:
                change_path_name = f'{scratch_dir}/cumsum_maximum_binary_{pol_single}.tif'
                print(f'saving {change_path_name}')
                util.write_raster_block(
                    change_path_name,
                    np.squeeze(dmask.sel(polarization=pol_single).values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)

            # # copy the residual, and apply dmask (threshold binary for cumsum)
            qmetric_residuals = R.copy().sel(polarization=pol_single)
            dmask = dmask.sel(polarization=pol_single)
            qmetric_residuals = qmetric_residuals.where(dmask)

            S_masked = S.sel(polarization=pol_single).where(dmask)
            Smax_masked = S_masked.max('time')
            Smin_masked = S_masked.min('time')
            SDiff_masked = Smax_masked - Smin_masked


            # to keep track of the maxium Sdiff of the bootstrapped sample:
            Sdiff_random_max = qmetric_residuals.mean('time').copy()
            # to compute the Sdiff sums of the bootstrapped sample:
            Sdiff_random_sum = qmetric_residuals.mean('time').copy()
            # to keep track of the count of the bootstrapped sample
            image_shape = (qmetric_residuals.sizes['x'], qmetric_residuals.sizes['y'])

            n_Sdiff_gt_Sdiff_random = np.ma.zeros((image_shape[1] ,image_shape[0]))

            # # Set the number of bootstraps to > 100, for demonstration purposes with start with 10
            n_bootstraps = 20
            print(f'Number of bootstraps in the trial run {n_bootstraps}')

            # Run in parallel
            # Prepare arguments for each iteration
            args_list = [(qmetric_residuals, dmask, Sdiff_random_max) for _ in range(n_bootstraps)]

            # Run in parallel
            with ProcessPoolExecutor() as executor:
                results = executor.map(run_bootstrap, args_list)

            # Aggregate results
            for Sdiff_random, gtmask in results:
                Sdiff_random_sum += Sdiff_random
                Sdiff_random_max.where(gtmask, Sdiff_random.where(gtmask).values)
                n_Sdiff_gt_Sdiff_random[np.ma.greater(Sdiff_random, Sdiff_random_max)] += 1


            CL = n_Sdiff_gt_Sdiff_random/n_bootstraps

            cl_thres=.98
            print(f'Chosen confidence level threshold:   CL={cl_thres*100}%')


            # Apply the threshold to the data stacks
            CL_da = SDiff_masked.copy()
            CL_da.data = CL
            CL_da = CL_da.where(dmask)
            CL_da_mask= CL_da >= cl_thres
            CL_da_invalid = CL_da < cl_thres
            CL_da_masked = CL_da.copy()
            CL_da_masked = CL_da_masked.where(CL_da_mask)

            if proc_param.debug_mode:

                confidence_path_name = f'{output_dir}/confidence_{pol_single}.tif'
                util.write_raster_block(
                    confidence_path_name,
                    np.squeeze(CL_da.values),
                    block_param=block_param,
                    datatype='float32',
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)
                confidence_valid_path_name = f'{output_dir}/confidence_valid_{pol_single}.tif'
                util.write_raster_block(
                    confidence_valid_path_name,
                    np.squeeze(CL_da_masked.values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)
                confidence_invalid_path_name = f'{output_dir}/confidence_invalid_{pol_single}.tif'
                util.write_raster_block(
                    confidence_invalid_path_name,
                    np.squeeze(CL_da_invalid.values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)

                for time_step in S['time']:
                    # S_masked = S.sel(polarization=pol_single).where(dmask)

                    Sdiff_single_date = S.sel(time=time_step) - S.min(dim='time')

                    Sdiff_single_date = Sdiff_single_date.where(Sdiff_single_date > 0)
                    Sdiff_single_date = Sdiff.where(S.max(dim='time')>0)  # Only care for positive values in Smax
                    dmask_single = Sdiff_single_date > dthres
                    Sdiff_single_date = Sdiff_single_date.where(dmask_single)

                    change_path_name = f'{output_dir}/cumsum_single_{time_step.values}_{pol_single}.tif'
                    print(f'saving {change_path_name}')
                    util.write_raster_block(
                        change_path_name,
                        np.squeeze(dmask_single.sel(polarization=pol_single).values),
                        block_param=block_param,
                        geotransform=image_meta['geotransform'],
                        projection=image_meta['projection'],
                        cog_flag=True)

            SDiff_array_masked = S_masked - Smin_masked
            # Apply a mask to ensure only positive values are considered
            SDiff_array_masked = SDiff_array_masked.where(SDiff_array_masked > 0)
            SDiff_array_masked_filled = SDiff_array_masked.fillna(-np.inf)

            # Determine where there is at least one non-NaN value along the time dimension
            valid_data_mask = SDiff_array_masked.notnull().any(dim='time')

            # Initialize arrays to hold max values and year information
            max_change_data = SDiff_array_masked.max(dim='time').where(valid_data_mask, drop=True)  # Get max value along time where valid data exists
            max_time_index = SDiff_array_masked.fillna(-np.inf).argmax(dim='time')  # Find max time index, ignoring NaNs

            # max_time_index = SDiff_array_masked.argmax(dim='time')

            time_of_max = SDiff_array_masked.time[max_time_index]
            year_of_max = time_of_max.dt.year
            year_data_ref = year_of_max - 2005  # Offset year data as required
            # Replace areas where dmask is False with 255 in year_data_ref
            year_data_ref = year_data_ref.where(dmask & CL_da_mask, 0)

            # Convert the result to an integer type since 255 is an integer
            year_data_ref = year_data_ref.astype(int)

            # Prepare cumulative change raster output
            cumulative_change_path_name = f'{output_dir}/cumulative_change_by_year_{pol_single}.tif'

            # year_data_ref = year_data_ref.squeeze()
            # print(np.shape(year_data_ref), 'year_ref')

            util.write_raster_block(
                cumulative_change_path_name,
                year_data_ref.values,  # Stack max and year as separate bands
                block_param=block_param,
                datatype='int8',
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                cog_flag=True
)
            # Collect all results and distinguish them




            # cumulative_change = None  # This will hold the cumulative results for all years
            # candidate_years = np.arange(2006, 2012)
            # starting_marker = 1

            # for year in candidate_years:
            #     print('Year', year)
            #     # Select all time steps for the current year
            #     yearly_scenes = S.sel(time=S['time.year'] == year)

            #     # Initialize an empty mask for the current year
            #     yearly_change = None  # This will hold the results for the current year

            #     for time_step in yearly_scenes['time']:
            #         # Calculate the difference and apply threshold
            #         S_min_single_pol = S.sel(polarization=pol_single).min(dim='time')

            #         Sdiff_single_date = yearly_scenes.sel(time=time_step).sel(polarization=pol_single) - S_min_single_pol
            #         print(Sdiff_single_date)

            #         Sdiff_single_date = Sdiff_single_date.where(Sdiff_single_date > 0)
            #         Sdiff_single_date = Sdiff_single_date.where(S.max(dim='time') > 0)
            #         Sdiff_single_date = Sdiff_single_date.sel(polarization=pol_single)
            #         # Mask for pixels above the threshold
            #         dmask_single = Sdiff_single_date > dthres
            #         Sdiff_single_date = Sdiff_single_date.where(dmask_single)

            #         # Update the yearly change mask
            #         if yearly_change is None:
            #             yearly_change = dmask_single.values
            #             print(np.shape(yearly_change))
            #         else:
            #             yearly_change = yearly_change | dmask_single  # Accumulate changes within the year

            #         # Convert yearly change mask to year-specific marker value
            #         yearly_change = yearly_change * starting_marker
            #         print('yc', np.shape(yearly_change))

            #         # Update the cumulative change mask
            #         if cumulative_change is None:
            #             cumulative_change = yearly_change.copy()
            #             print('cum', np.shape(cumulative_change))

            #         else:
            #             # Only update cumulative_change where yearly_change has non-zero values
            #             cumulative_change = np.where(cumulative_change > 0, cumulative_change, yearly_change)

            #         # Increment the marker for the next year
            #     starting_marker += 1
            #     print(np.shape(cumulative_change))
            #     # Save the final cumulative change result with year-specific values
            #     cumulative_change_path_name = f'{output_dir}/cumulative_change_by_year_{pol_single}.tif'
            #     print(f'Saving cumulative change to {cumulative_change_path_name}')
            #     util.write_raster_block(
            #         cumulative_change_path_name,
            #         np.squeeze(cumulative_change),
            #         block_param=block_param,
            #         geotransform=image_meta['geotransform'],
            #         projection=image_meta['projection'],
            #         cog_flag=True
            #         )

    t_time_end = time.time()
    logger.info(f'total processing time: {t_time_end - t_all} sec')


def main():

    parser = _get_parser()
    args = parser.parse_args()
    cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dist_algorithm', args)
    generate_log.configure_log_file(cfg.groups.log_file)

    dist_workflow(cfg)


if __name__ == '__main__':
    '''run dswx_ni from command line'''
    main()
