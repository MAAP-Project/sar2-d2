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

from dist import reader_ni, util, filter_SAR
from dist.dist_runconfig import (_get_parser,
                                 RunConfig,)
from dist import generate_log


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


def dist_workflow(cfg):

    import time
    t_all = time.time()

    input_gcov_list = cfg.groups.input_file_group.input_file_path
    scratch_dir = cfg.groups.product_path_group.scratch_path
    os.makedirs(scratch_dir, exist_ok=True)
    block_length = 300
    forest_cross_thres = 0.008 # -20 dB
    forest_cross_thres = -20 #dB
    filter_option = {'lambda_value': 20}

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

    resamp_required = False
    resamp_method = 'nearest'
    resamp_out_res = 20

    row_blk_size = 300
    col_blk_size = 300

    # Create reader object
    reader = reader_ni.RTCReader(
        row_blk_size=row_blk_size,
        col_blk_size=col_blk_size,
    )
    data_stack_df['geotiff_co'] = ""
    data_stack_df['geotiff_cross'] = ""

    for t, row in data_stack_df.iterrows():
        ddd = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])

        # Format the datetime to 'yyyymmdd'
        formatted_time = ddd.strftime('%Y%m%d')

        # Mosaic input RTC into output Geotiff
        # reader.process_rtc_hdf5(
        #     input_list=[row['rtc_path']],
        #     scratch_dir=scratch_dir,
        #     mosaic_mode=mosaic_mode,
        #     mosaic_prefix=formatted_time,
        #     resamp_method=resamp_method,
        #     resamp_out_res=resamp_out_res,
        #     resamp_required=resamp_required,
        # )

        output_filename = f'{scratch_dir}/{formatted_time}_HH.tif'
        if os.path.isfile(output_filename):
            data_stack_df.at[t, 'geotiff_co'] = output_filename
        output_filename = f'{scratch_dir}/{formatted_time}_HV.tif'
        if os.path.isfile(output_filename):
            data_stack_df.at[t, 'geotiff_cross'] = output_filename



    geogrid_in = reader_ni.DSWXGeogrid()

    for t, row in data_stack_df.iterrows():
        geogrid_in.update_geogrid(row['geotiff_co'])

    for t, row in data_stack_df.iterrows():
        ddd = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])

        # Format the datetime to 'yyyymmdd'
        formatted_time = ddd.strftime('%Y%m%d')
        output_filename = f'{scratch_dir}/{formatted_time}_HH_resampled.tif'
        # util.resample_and_crop_with_gdalwarp(
        #     row['geotiff_co'],
        #     output_filename,
        #     geogrid_in.start_x,
        #     geogrid_in.start_y, 
        #     geogrid_in.end_x, 
        #     geogrid_in.end_y, 
        #     geogrid_in.spacing_x, 
        #     geogrid_in.spacing_y, 
        #     geogrid_in.epsg)
        if os.path.isfile(output_filename):
            data_stack_df.at[t, 'geotiff_co'] = output_filename

        output_filename = f'{scratch_dir}/{formatted_time}_HV_resampled.tif'
        # util.resample_and_crop_with_gdalwarp(
        #     row['geotiff_cross'],
        #     output_filename,
        #     geogrid_in.start_x,
        #     geogrid_in.start_y, 
        #     geogrid_in.end_x, 
        #     geogrid_in.end_y, 
        #     geogrid_in.spacing_x, 
        #     geogrid_in.spacing_y, 
        #     geogrid_in.epsg)
        if os.path.isfile(output_filename):
            data_stack_df.at[t, 'geotiff_cross'] = output_filename
    image_meta = util.get_meta_from_tif(data_stack_df.iloc[0]['geotiff_co'])

    pad_shape = (0, 0)
    block_params = util.block_param_generator(
        lines_per_block=block_length,
        data_shape=(geogrid_in.length,
                    geogrid_in.width),
        pad_shape=pad_shape)


    for block_param in block_params:
        row_start = block_param.read_start_line
        row_num = block_param.read_start_line + block_param.read_length
        pixel_window = Window(row_start,
                            0,
                            row_num,
                            block_param.data_width)
        da_stack = []

        for t, row in data_stack_df.iterrows():

            acqtime = pd.to_datetime(row['ZERO_DOPPLER_START_TIME'])
            polarization_stack = []

            for polarization in polarizations:
                if polarization in ['HH', 'VV']:
                    file_path = row["geotiff_co"]
                else:
                    file_path = row["geotiff_cross"]
                
                subset_data = util.get_raster_block(
                    file_path,
                    block_param)
                print('*****',np.nanmean(subset_data))

                despeckled = filter_SAR.tv_bregman(subset_data, **filter_option)
                despeckled_db = 10 * np.log10(despeckled)
                print('----',np.nanmean(despeckled))
                    # Create an xarray DataArray from the subsetted data
                da = xr.DataArray(despeckled_db, dims=['y', 'x'], 
                                  coords={'y': np.arange(subset_data.shape[0]), 
                                          'x': np.arange(subset_data.shape[1])})
                
                # Add time dimension to the DataArray
                da = da.expand_dims(time=pd.Index([acqtime], name='time'))
                polarization_stack.append(da)
                            
            # Concatenate the polarization stack along the 'polarization' dimension
            da_polarized = xr.concat(polarization_stack, dim=pd.Index(polarizations, name='polarization'))
            da_stack.append(da_polarized)

        ds = xr.concat(da_stack, dim='time')
        sorted_ds = ds.sortby('time')
        da_min_vh = sorted_ds.min('time').sel(polarization='HV')

        fmask = (da_min_vh > forest_cross_thres).persist()
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

        Sdiff = S.max(dim='time') - S.min(dim='time')
        Sdiff = S.max(dim='time', skipna=True) - S.min(dim='time', skipna=True)

        for pol_single in polarizations:
            change_path_name = f'{scratch_dir}/cumsum_maximum_{pol_single}.tif'
            util.write_raster_block(
                change_path_name,
                np.squeeze(Sdiff.sel(polarization=pol_single).values),
                block_param=block_param,
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                datatype='float32',
                cog_flag=True)

        Sdiff = Sdiff.where(Sdiff > 0)
        Sdiff = Sdiff.where(S.max(dim='time')>0)  # Only care for positive values in Smax

        quantile = 85
        #arr = Sdiff.values.flatten()
        dthres = np.nanpercentile(Sdiff.values.flatten(), quantile)
        print(f'Setting the threshold for SDiff based on the {quantile}th percentile to: {dthres:.3f}')

        dmask = Sdiff > dthres 
        Sdiff = Sdiff.where(dmask)
        for pol_single in polarizations:
            change_path_name = f'{scratch_dir}/cumsum_maximum_binary_{pol_single}.tif'
            util.write_raster_block(
                change_path_name,
                np.squeeze(dmask.sel(polarization=pol_single).values),
                block_param=block_param,
                geotransform=image_meta['geotransform'],
                projection=image_meta['projection'],
                cog_flag=True)

        # qmetric_residuals = R.copy().sel(polarization=polarization)
        # dmask = dmask.sel(polarization=polarization)
        # qmetric_residuals = qmetric_residuals.where(dmask)

        # S_masked=S.where(dmask)
        # Smax_masked=S_masked.max('time')
        # Smin_masked=S_masked.min('time')
        # SDiff_masked=Smax_masked-Smin_masked


        # # to keep track of the maxium Sdiff of the bootstrapped sample:
        # Sdiff_random_max = qmetric_residuals.mean('time').copy() 
        # # to compute the Sdiff sums of the bootstrapped sample:
        # Sdiff_random_sum = qmetric_residuals.mean('time').copy() 
        # # to keep track of the count of the bootstrapped sample
        # image_shape = (qmetric_residuals.sizes['x'], qmetric_residuals.sizes['y'])

        # n_Sdiff_gt_Sdiff_random=np.ma.zeros((1, image_shape[1] ,image_shape[0]))
        # image_shape = (qmetric_residuals.sizes['x'], qmetric_residuals.sizes['y'])
        # # Set the number of bootstraps to > 100, for demonstration purposes with start with 10
        # n_bootstraps=100
        # if develop_run:
        #     n_bootstraps=1
        # print(f'Number of bootstraps in the trial run {n_bootstraps}')

        # # to compute the Sdiff sums of the bootstrapped sample:
        # for i in range(n_bootstraps):
        #     # For efficiency, we shuffle the time axis index and use that 
        #     #to randomize the masked array
        #     permutation=np.random.permutation(len(qmetric_residuals))
        #     Srandom = qmetric_residuals[permutation].cumsum('time')
        #     Srandom = Srandom.where(dmask)
        #     Srandom_max=Srandom.max('time')
        #     Srandom_min=Srandom.min('time')
        #     Sdiff_random=Srandom_max-Srandom_min
            
        #     Sdiff_random_sum += Sdiff_random
        #     # Mask where Sdiff_random is greater than Sdiff_random_max
        #     gtmask=np.ma.greater(Sdiff_random,Sdiff_random_max).data
        #     Sdiff_random_max.where(gtmask,Sdiff_random.where(np.ma.greater(Sdiff_random,Sdiff_random_max).data).values)
        #     n_Sdiff_gt_Sdiff_random[np.ma.greater(SDiff_masked,Sdiff_random)] += 1

        # CL = n_Sdiff_gt_Sdiff_random/n_bootstraps

        # cl_thres=.98
        # print(f'Chosen confidence level threshold:   CL={cl_thres*100}%')


        # # Apply the threshold to the data stacks 
        # CL_da = SDiff_masked.copy()
        # CL_da.data = CL
        # CL_da = CL_da.where(dmask)
        # CL_da_mask= CL_da >=cl_thres
        # CL_da_invalid=CL_da<cl_thres
        # CL_da_masked = CL_da.copy()
        # CL_da_masked = CL_da_masked.where(CL_da_mask)



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
