import os
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from cum_sum_dist import (reader_ni,
                          util,
                          filter_SAR,
                          generate_log)
from cum_sum_dist.dist_runconfig import (_get_parser,
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


def bootstrap_trial(args):
    """Single bootstrap iteration."""
    qmetric_residuals, dmask = args
    # Shuffle time axis for permutation
    permutation = np.random.permutation(len(qmetric_residuals['time']))
    Srandom = qmetric_residuals.isel(time=permutation).cumsum('time')
    Srandom = Srandom.where(dmask)
    Srandom_max = Srandom.max('time')
    Srandom_min = Srandom.min('time')
    Sdiff_random = Srandom_max - Srandom_min

    return Sdiff_random


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

    t_all = time.time()

    input_gcov_list = cfg.groups.input_file_group.input_file_path
    scratch_dir = cfg.groups.product_path_group.scratch_path
    output_dir = cfg.groups.product_path_group.sas_output_path
    proc_param = cfg.groups.processing_parameters

    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    block_length = 300
    forest_cross_thres = proc_param.forest_mask_threshold #dB
    bootstrap_number = proc_param.bootstrap_number
    resamp_out_res = cfg.groups.product_path_group.output_spacing
    confidence_threshold = proc_param.confidence_threshold
    filter_option = {'lambda_value': proc_param.filter_lambda}

    polarizations = util.extract_nisar_polarization(input_gcov_list)
    date_str_list = []
    data_stack = []

    # Extract metadata from HDF 5
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
                convert_HDF5_to_GTIFF, t, row, reader, scratch_dir,
                mosaic_mode, resamp_method, resamp_out_res, resamp_required
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
        cumsum_result = R.fillna(0).cumsum(dim='time')
        valid_counts = R.count(dim='time')
        # mask out invalid values
        cumsum_result_masked = cumsum_result.where(cumsum_result != 0)

        cumsum_result_diff = cumsum_result_masked.max(dim='time', skipna=True) - cumsum_result_masked.min(dim='time', skipna=True)

        for pol_single in polarizations:
            counts_pol = valid_counts.sel(polarization=pol_single)
            valid_mask = counts_pol == np.max(counts_pol)

            output_path = f'{output_dir}/valid_counts_{pol_single}.tif'
            print(f'Saving valid counts to {output_path}')
            util.write_raster_block(
                output_path,
                np.squeeze(counts_pol.values),
                block_param=block_param,
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                datatype='int16',
                cog_flag=True
            )
            
            da_mean = sorted_ds.mean('time').sel(polarization=pol_single)
            mean_backscatter_path = f'{output_dir}/mean_backscatter_{pol_single}.tif'

            util.write_raster_block(
                mean_backscatter_path,
                np.squeeze(da_mean.values),
                block_param=block_param,
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                datatype='float32',
                cog_flag=True)

            if proc_param.debug_mode:

                change_path_name = f'{scratch_dir}/cumsum_maximum_{pol_single}.tif'
                print(f'saving {change_path_name}')
                util.write_raster_block(
                    change_path_name,
                    np.squeeze(cumsum_result_diff.sel(polarization=pol_single).values),
                    block_param=block_param,
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    datatype='float32',
                    cog_flag=True)

            # Loop over each time step in R
            for time_step in R['time']:
                # Extract data for the specific date
                cumsum_result_date = cumsum_result.sel(time=time_step).sel(polarization=pol_single)

                # Compute the mean value of S_date over non-zero data
                mean_S_date = cumsum_result_date.mean(dim=('x', 'y'), skipna=True).item()
                mean_X_date = X.sel(time=time_step).sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()
                mean_R_date = R.sel(time=time_step).sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()
                
                # Print the mean value
                date_str = pd.to_datetime(time_step.values).strftime('%Y-%m-%d')

                # Check if S_date has any non-zero data
                if np.isnan(mean_X_date):
                    print(f"Mean value of S for date {date_str} and polarization {pol_single}: {mean_S_date} {mean_X_date} {mean_R_date}")
                    print(f"No data for date {date_str} and polarization {pol_single}, skipping.")
                    continue  # Skip to the next time step

                # Construct the file name using the time step
                date_str = pd.to_datetime(time_step.values).strftime('%Y%m%d')

                if proc_param.debug_mode:
                    output_path = f'{scratch_dir}/cumsum_result_{date_str}_{pol_single}.tif'
                    # Save the data to a GeoTIFF file
                    util.write_raster_block(
                        output_path,
                        np.squeeze(cumsum_result_date.values),
                        block_param=block_param,
                        geotransform=image_meta['geotransform'],
                        projection=image_meta['projection'],
                        datatype='float32',  # or another appropriate datatype
                        cog_flag=True
                    )

            cumsum_result_diff = cumsum_result_diff.where(cumsum_result_diff > 0)
            cumsum_result_diff = cumsum_result_diff.where(cumsum_result.max(dim='time')>0)  # Only care for positive values in Smax

            quantile = 85
            #arr = Sdiff.values.flatten()
            if proc_param.threshold is None:
                dthres = np.nanpercentile(cumsum_result_diff.sel(polarization=pol_single).values.flatten(), quantile)
                print(f'Setting the threshold for cumsum_result_Diff based on the {quantile}th percentile to: {dthres:.3f}')

            else:
                dthres = proc_param.threshold
                print(f'Setting the threshold for User-defined value: {dthres:.3f}')

            dmask = cumsum_result_diff > dthres
            cumsum_result_diff = cumsum_result_diff.where(dmask)

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

            cumsum_result_masked = cumsum_result.sel(polarization=pol_single).where(dmask)
            cumsum_result_max_masked = cumsum_result_masked.max('time')
            cumsum_result_min_masked = cumsum_result_masked.min('time')
            cumsum_result_Diff_masked = cumsum_result_max_masked - cumsum_result_min_masked

            # # Set the number of bootstraps to > 100, for demonstration purposes with start with 10
            n_bootstraps = bootstrap_number
            print(f'Number of bootstraps in the trial run {n_bootstraps}')
            # Prepare arguments for each iteration
            args_list = [(qmetric_residuals, dmask) for _ in range(n_bootstraps)]

            # Run in parallel
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(bootstrap_trial, args_list))

            # Initialize variables
            n_cumsum_result_gt_Sdiff_random = xr.zeros_like(cumsum_result_Diff_masked)
            Sdiff_random_list = []

            # Aggregate results
            for Sdiff_random in results:
                Sdiff_random_list.append(Sdiff_random)

                # Update n_Sdiff_gt_Sdiff_random where SDiff_masked > Sdiff_random
                gt_Sdiff_mask = cumsum_result_Diff_masked > Sdiff_random
                n_cumsum_result_gt_Sdiff_random += gt_Sdiff_mask.astype(int)

            CL = n_cumsum_result_gt_Sdiff_random / n_bootstraps

            cl_thres = confidence_threshold
            print(f'Chosen confidence level threshold:   CL={cl_thres*100}%')

            # Apply the threshold to the data stacks
            CL_da = cumsum_result_Diff_masked.copy()
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
                
                confidence_path_name = f'{output_dir}/CL_{pol_single}.tif'
                util.write_raster_block(
                    confidence_path_name,
                    np.squeeze(CL),
                    block_param=block_param,
                    datatype='float32',
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)
                confidence_path_name = f'{output_dir}/test_{pol_single}.tif'
                util.write_raster_block(
                    confidence_path_name,
                    np.squeeze(n_cumsum_result_gt_Sdiff_random),
                    block_param=block_param,
                    datatype='float32',
                    geotransform=image_meta['geotransform'],
                    projection=image_meta['projection'],
                    cog_flag=True)

                for time_step in cumsum_result['time']:
                    # S_masked = cumsum_result.sel(polarization=pol_single).where(dmask)

                    cumsum_result_diff_single_date = cumsum_result.sel(time=time_step) - cumsum_result.min(dim='time')
                    date_str = pd.to_datetime(time_step.values).strftime('%Y-%m-%d')

                    mean_X_date = X.sel(time=time_step).sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()
                    mean_cumsum_result_date = cumsum_result_diff_single_date.sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()
                    # Check if S_date has any non-zero data
                    if np.isnan(mean_X_date):
                        continue

                    cumsum_result_diff_single_date = cumsum_result_diff_single_date.where(cumsum_result_diff_single_date > 0)
                    cumsum_result_diff_single_date = cumsum_result_diff_single_date.where(cumsum_result.max(dim='time') > 0)  # Only care for positive values in Smax
                    mean_cumsum_result_date = cumsum_result_diff_single_date.sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()

                    dmask_single = cumsum_result_diff_single_date > dthres
                    cumsum_result_diff_single_date = cumsum_result_diff_single_date.where(dmask_single)
                    change_path_name = f'{output_dir}/cumsum_single_{date_str}_{pol_single}.tif'
                    mean_cumsum_result_date = cumsum_result_diff_single_date.sel(polarization=pol_single).mean(dim=('x', 'y'), skipna=True).item()

                    util.write_raster_block(
                        change_path_name,
                        np.squeeze(cumsum_result_diff_single_date.sel(polarization=pol_single).values),
                        block_param=block_param,
                        datatype='float32',
                        geotransform=image_meta['geotransform'],
                        projection=image_meta['projection'],
                        cog_flag=True)

                    change_path_name = f'{output_dir}/cumsum_single_binary_{date_str}_{pol_single}.tif'
                    util.write_raster_block(
                        change_path_name,
                        np.squeeze(dmask_single.sel(polarization=pol_single).values),
                        block_param=block_param,
                        geotransform=image_meta['geotransform'],
                        projection=image_meta['projection'],
                        cog_flag=True)

            cumsum_result_Diff_array_masked = cumsum_result_masked - cumsum_result_min_masked
            # Apply a mask to ensure only positive values are considered
            cumsum_result_Diff_array_masked = cumsum_result_Diff_array_masked.where(cumsum_result_Diff_array_masked > 0)
            cumsum_result_Diff_array_masked_filled = cumsum_result_Diff_array_masked.fillna(-np.inf)

            # Determine where there is at least one non-NaN value along the time dimension
            valid_data_mask = cumsum_result_Diff_array_masked.notnull().any(dim='time')

            # Initialize arrays to hold max values and year information
            max_change_data = cumsum_result_Diff_array_masked.max(dim='time').where(valid_data_mask, drop=True)  # Get max value along time where valid data exists
            max_time_index = cumsum_result_Diff_array_masked.fillna(-np.inf).argmax(dim='time')  # Find max time index, ignoring NaNs

            # max_time_index = cumsum_result_Diff_array_masked.argmax(dim='time')

            time_of_max = cumsum_result_Diff_array_masked.time[max_time_index]
            year_of_max = time_of_max.dt.year
            year_data_ref = year_of_max - 2005  # Offset year data as required
            # Replace areas where dmask is False with 255 in year_data_ref
            year_data_ref = year_data_ref.where(dmask & CL_da_mask & valid_mask, 0)

            # Convert the result to an integer type since 255 is an integer
            year_data_ref = year_data_ref.astype(int)

            # Prepare cumulative change raster output
            cumulative_change_path_name = f'{output_dir}/cumulative_change_by_year_{pol_single}.tif'

            util.write_raster_block(
                cumulative_change_path_name,
                year_data_ref.values,  # Stack max and year as separate bands
                block_param=block_param,
                datatype='int8',
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                cog_flag=True
                )

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
