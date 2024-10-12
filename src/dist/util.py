import logging
import os
import math
import shutil
import tempfile
import subprocess

import numpy as np 
import h5py
from osgeo import gdal, osr
from rasterio.transform import Affine
from shapely.geometry import LinearRing, Point, Polygon, box

from dataclasses import dataclass


np2gdal_conversion = {
  "byte": 1,
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}


def extract_nisar_polarization(input_list):
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


def read_metadata_hdf5(input_rtc):
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

    pol_list_path = \
        '/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations'
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
        'RTC_POL': f'{pol_list_path}'
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
        rtc_pols = src_h5[dswx_meta_mapping['RTC_POL']][()]
        rtc_decoded_pol = [pol.decode('utf-8') for pol in rtc_pols]
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
        'POLARIZATIONS': rtc_decoded_pol
    }

    return dswx_metadata_dict


def get_gcov_size(filename, path):
    
    with h5py.File(filename) as src:
        h5_ds = src[path]
        shape = h5_ds.shape
        height, width  = shape[-2], shape[-1]
    return width, height



def read_tif_latlon(intput_tif_str):
    #  Initialize the Image Size
    ds = gdal.Open(intput_tif_str)
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    epsg_input = proj.GetAttrValue('AUTHORITY',1)

    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]

    ds = None
    del ds  # close the dataset (Python object and pointers)
    if epsg_input != 4326:
        xcoords = [minx, maxx, maxx, minx]
        ycoords = [miny, miny, maxy, maxy]

        poly_wkt = []  # Initialize as a list

        for xcoord, ycoord in zip(xcoords, ycoords):
            lon, lat = get_lonlat(xcoord, ycoord, int(epsg_input))
            poly_wkt.append((lon, lat))

        poly = Polygon(poly_wkt)
    else:
        poly = box(minx, miny, maxx, maxy)

    return poly


def get_lonlat(xcoord, ycoord, epsg):

    from osgeo import ogr
    from osgeo import osr

    InSR = osr.SpatialReference()
    InSR.ImportFromEPSG(epsg)       # WGS84/Geographic
    OutSR = osr.SpatialReference()
    OutSR.ImportFromEPSG(4326)     # WGS84 UTM Zone 56 South

    Point = ogr.Geometry(ogr.wkbPoint)
    Point.AddPoint(xcoord, ycoord) # use your coordinates here
    Point.AssignSpatialReference(InSR)    # tell the point what coordinates it's in
    Point.TransformTo(OutSR)              # project it to the out spatial reference
    return Point.GetX(), Point.GetY()

def get_rtc_stack_block(filename_list, path, blocksize, block_ind, scale='db'):
    refind = 0

    # define basic frame from first scene
    rtc_path = filename_list[refind]
    refcols, refrows = get_gcov_size(rtc_path, path)

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
        lat_rtc, lon_rtc = read_hdf_latlon(fname, path)

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
        with h5py.File(fname) as src:
            band = src[path]
            
    
        # Extract the subset using slicing
        subset = band[int(target_ul_y_ind):int(target_ul_y_ind) +  int(row_sub),
                         int(target_ul_x_ind):int(target_ul_x_ind) + int(col_sub)]

        # Optionally, convert to a NumPy array if further processing is needed
        target_rtc_image = np.array(subset)
        # target_rtc_image = band.ReadAsArray(int(target_ul_x_ind),
        #     int(target_ul_y_ind),
        #     int(col_sub), int(row_sub))

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


def read_geotiff(input_tif_str, band_ind=None, verbose=True):
    """Read band from geotiff

    Parameters
    ----------
    input_tif_str: str
        geotiff file path to read the band
    band_ind: int
        Index of the band to read, starts from 0

    Returns
    -------
    tifdata: numpy.ndarray
        image from geotiff
    """
    tif = gdal.Open(input_tif_str)
    if band_ind is None:
        tifdata = tif.ReadAsArray()
    else:
        tifdata = tif.GetRasterBand(band_ind + 1).ReadAsArray()

    tif.FlushCache()
    tif = None
    del tif
    if verbose:
        print(f" -- Reading {input_tif_str} ... {tifdata.shape}")
    return tifdata



def read_hdf_latlon(intput_h5_path, path):
    #  Initialize the Image Size

    dataset = f'HDF5:{intput_h5_path}:/{path}'
    h5_ds = gdal.Open(dataset, gdal.GA_ReadOnly)
    with h5py.File(intput_h5_path) as src:
        h5_ds = src[path]
        shape = h5_ds.shape
        width, height = shape[-2], shape[-1]

    gt, crs = read_geodata_hdf5(intput_h5_path)

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

def read_geodata_hdf5(input_rtc):
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


def block_param_generator(lines_per_block, data_shape, pad_shape):
    ''' Generator for block specific parameter class.

    Parameters
    ----------
    lines_per_block: int
        Lines to be processed per block (in batch).
    data_shape: tuple(int, int)
        Length and width of input raster.
    pad_shape: tuple(int, int)
        Padding for the length and width of block to be filtered.

    Returns
    -------
    _: BlockParam
        BlockParam object for current block
    '''
    data_length, data_width = data_shape
    pad_length, pad_width = pad_shape
    half_pad_length = pad_length // 2
    half_pad_width = pad_width // 2

    # Calculate number of blocks to break raster into
    num_blocks = int(np.ceil(data_length / lines_per_block))

    for block in range(num_blocks):
        start_line = block * lines_per_block

        # Discriminate between first, last, and middle blocks
        first_block = block == 0
        last_block = block == num_blocks - 1 or num_blocks == 1
        middle_block = not first_block and not last_block

        # Determine block size; Last block uses leftover lines
        block_length = data_length - start_line \
            if last_block else lines_per_block
        # Determine padding along length. Full padding for middle blocks
        # Half padding for start and end blocks
        read_length_pad = pad_length if middle_block else half_pad_length

        # Determine 1st line of output
        write_start_line = block * lines_per_block

        # Determine 1st dataset line to read. Subtract half padding length
        # to account for additional lines to be read.
        read_start_line = block * lines_per_block - half_pad_length

        # If applicable, save negative start line as deficit
        # to account for later
        read_start_line, start_line_deficit = (
            0, read_start_line) if read_start_line < 0 else (
            read_start_line, 0)

        # Initial guess at number lines to read; accounting
        # for negative start at the end
        read_length = block_length + read_length_pad
        if not first_block:
            read_length -= abs(start_line_deficit)

        # Check for over-reading and adjust lines read as needed
        end_line_deficit = min(
            data_length - read_start_line - read_length, 0)
        read_length -= abs(end_line_deficit)

        # Determine block padding in length
        if first_block:
            # Only the top part of the block should be padded.
            # If end_deficit_line=0 we have a sufficient number
            # of lines to be read in the subsequent block
            top_pad = half_pad_length
            bottom_pad = abs(end_line_deficit)
        elif last_block:
            # Only the bottom part of the block should be padded
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = half_pad_length
        else:
            # Top and bottom should be added taking into account line deficit
            top_pad = abs(
                start_line_deficit) if start_line_deficit < 0 else 0
            bottom_pad = abs(end_line_deficit)

        block_pad = ((top_pad, bottom_pad),
                     (half_pad_width, half_pad_width))

        yield BlockParam(block_length,
                         write_start_line,
                         read_start_line,
                         read_length,
                         block_pad,
                         data_width,
                         data_length)

    return


@dataclass
class BlockParam:
    '''
    Class for block specific parameters
    Facilitate block parameters exchange between functions
    '''
    # Length of current block to filter; padding not included
    block_length: int

    # First line to write to for current block
    write_start_line: int

    # First line to read from dataset for current block
    read_start_line: int

    # Number of lines to read from dataset for current block
    read_length: int

    # Padding to be applied to read in current block. First tuple is padding to
    # be applied to top/bottom (along length). Second tuple is padding to be
    # applied to left/right (along width). Values in second tuple do not change
    # included in class so one less value is passed between functions.
    block_pad: tuple

    # Width of current block. Value does not change per block; included to
    # in class so one less value is to be passed between functions.
    data_width: int

    data_length: int


def get_raster_block(raster_path, block_param):
    ''' Get a block of data from raster.
        Raster can be a HDF5 file or a GDAL-friendly raster

    Parameters
    ----------
    raster_path: str
        raster path where a block is to be read from. String value represents a
        filepath for GDAL rasters.
    block_param: BlockParam
        Object specifying size of block and where to read from raster,
        and amount of padding for the read array

    Returns
    -------
    data_block: np.ndarray
        Block read from raster with shape specified in block_param.
    '''
    # Open input data using GDAL to get raster length
    ds_data = gdal.Open(raster_path, gdal.GA_Update)

    # Number of bands in the raster
    num_bands = ds_data.RasterCount
    # List to store blocks from each band
    data_blocks = []
    for i in range(num_bands):
        band = ds_data.GetRasterBand(i+1)
        data_block = band.ReadAsArray(
            0,
            block_param.read_start_line,
            block_param.data_width,
            block_param.read_length)

        # Pad data_block with zeros according to pad_length/pad_width
        data_block = np.pad(data_block, block_param.block_pad,
                            mode='constant', constant_values=0)

        if data_block.ndim == 1:
            data_block = data_block[np.newaxis, :]
        data_blocks.append(data_block)
    data_blocks = np.array(data_blocks)

    if num_bands == 1:
        data_blocks = np.reshape(data_blocks,
                                 [data_blocks.shape[1],
                                  data_blocks.shape[2]])
    return data_blocks


def write_raster_block(out_raster, data,
                       block_param, geotransform, projection,
                       datatype='byte',
                       cog_flag=False,
                       scratch_dir='.'):
    """
    Write processed data block to the specified raster file.

    Parameters
    ----------
    out_raster : h5py.Dataset or str
        Raster where data needs to be written. String value represents
        filepath for GDAL rasters.
    data : np.ndarray
        Data to be written to the raster.
    block_param : BlockParam
        Specifications for the data block to be written.
    geotransform : tuple
        GeoTransform parameters for the raster.
    projection : str
        Projection string for the raster.
    datatype : str, optional
        Data type of the raster. Defaults to 'byte'.
    cog_flag : bool, optional
        If True, converts the raster to COG format. Defaults to False.
    scratch_dir : str, optional
        Directory for intermediate processing. Defaults to '.'.
    """
    gdal_type = np2gdal_conversion[datatype]

    data = np.array(data, dtype=datatype)
    ndim = data.ndim
    number_band = 1 if ndim < 3 else data.shape[0]

    data_start_without_pad = block_param.write_start_line - \
        block_param.read_start_line + block_param.block_pad[0][0]
    data_end_without_pad = data_start_without_pad + \
        block_param.block_length

    if block_param.write_start_line == 0:
        driver = gdal.GetDriverByName('GTiff')
        ds_data = driver.Create(out_raster,
                                block_param.data_width,
                                block_param.data_length,
                                number_band, gdal_type)
        if not ds_data:
            raise IOError(f"Failed to create raster: {out_raster}")

        ds_data.SetGeoTransform(geotransform)
        ds_data.SetProjection(projection)
    else:
        ds_data = gdal.Open(out_raster, gdal.GA_Update)
        if not ds_data:
            raise IOError(f"Failed to open raster for update: {out_raster}")

    if ndim == 3:
        for im_ind in range(0, number_band):

            ds_data.GetRasterBand(im_ind+1).WriteArray(
                data[im_ind,
                     data_start_without_pad:data_end_without_pad,
                     :],
                xoff=0,
                yoff=block_param.write_start_line)
    elif data.ndim == 2:
        data_towrite = data[data_start_without_pad:data_end_without_pad, :]
        ds_data.GetRasterBand(1).WriteArray(
            data_towrite,
            xoff=0,
            yoff=block_param.write_start_line)
    # data.ndim == 1
    else:
        ds_data.GetRasterBand(1).WriteArray(
            np.reshape(data, [1, len(data)]),
            xoff=0,
            yoff=block_param.write_start_line)
    del ds_data

    # Write COG is cog_flag is True and last block.
    if (block_param.write_start_line + block_param.block_length ==
       block_param.data_length) and cog_flag:
        _save_as_cog(out_raster, scratch_dir)


def _save_as_cog(filename,
                 scratch_dir='.',
                 logger=None,
                 flag_compress=True,
                 ovr_resamp_algorithm=None,
                 compression='DEFLATE',
                 nbits=16):
    """Save (overwrite) a GeoTIFF file as a cloud-optimized GeoTIFF.

    Parameters
    ----------
    filename: str
            GeoTIFF to be saved as a cloud-optimized GeoTIFF
    scratch_dir: str (optional)
            Temporary Directory
    ovr_resamp_algorithm: str (optional)
            Resampling algorithm for overviews.
            Options: "AVERAGE", "AVERAGE_MAGPHASE", "RMS", "BILINEAR",
            "CUBIC", "CUBICSPLINE", "GAUSS", "LANCZOS", "MODE",
            "NEAREST", or "NONE". Defaults to "NEAREST", if integer, and
            "CUBICSPLINE", otherwise.
    compression: str (optional)
            Compression type.
            Optional: "NONE", "LZW", "JPEG", "DEFLATE", "ZSTD", "WEBP",
            "LERC", "LERC_DEFLATE", "LERC_ZSTD", "LZMA"
    """
    if logger is None:
        logger = logging.getLogger('proteus')

    logger.info('        COG step 1: add overviews')
    gdal_ds = gdal.Open(filename, gdal.GA_Update)
    gdal_dtype = gdal_ds.GetRasterBand(1).DataType
    dtype_name = gdal.GetDataTypeName(gdal_dtype).lower()

    overviews_list = [4, 16, 64, 128]

    is_integer = 'byte' in dtype_name or 'int' in dtype_name
    if ovr_resamp_algorithm is None and is_integer:
        ovr_resamp_algorithm = 'NEAREST'
    elif ovr_resamp_algorithm is None:
        ovr_resamp_algorithm = 'CUBICSPLINE'

    gdal_ds.BuildOverviews(ovr_resamp_algorithm, overviews_list,
                           gdal.TermProgress_nocb)

    del gdal_ds  # close the dataset (Python object and pointers)
    external_overview_file = filename + '.ovr'
    if os.path.isfile(external_overview_file):
        os.remove(external_overview_file)

    logger.info('        COG step 2: save as COG')
    temp_file = tempfile.NamedTemporaryFile(
                    dir=scratch_dir, suffix='.tif').name

    # Blocks of 512 x 512 => 256 KiB (UInt8) or 1MiB (Float32)
    tile_size = 512
    gdal_translate_options = ['BIGTIFF=IF_SAFER',
                              'MAX_Z_ERROR=0',
                              'TILED=YES',
                              f'BLOCKXSIZE={tile_size}',
                              f'BLOCKYSIZE={tile_size}',
                              'COPY_SRC_OVERVIEWS=YES']

    if compression:
        gdal_translate_options += [f'COMPRESS={compression}']

    if is_integer:
        gdal_translate_options += ['PREDICTOR=2']
    else:
        gdal_translate_options += ['PREDICTOR=3']

    if nbits is not None:
        gdal_translate_options += [f'NBITS={nbits}']

        # suppress type casting errors
        gdal.SetConfigOption('CPL_LOG', '/dev/null')

    gdal.Translate(temp_file, filename,
                   creationOptions=gdal_translate_options)

    shutil.move(temp_file, filename)

def resample_and_crop_with_gdalwarp(input_file, output_file, start_x, start_y, end_x, end_y, spacing_x, spacing_y, epsg):
    """
    Resample and crop a GeoTIFF using gdalwarp, overwriting the existing file, with EPSG projection.

    :param input_file: Path to the input GeoTIFF file (will be overwritten)
    :param start_x: X-coordinate of the top-left corner (must be less than end_x)
    :param start_y: Y-coordinate of the top-left corner (must be greater than end_y)
    :param end_x: X-coordinate of the bottom-right corner
    :param end_y: Y-coordinate of the bottom-right corner
    :param spacing_x: Pixel size in the X direction
    :param spacing_y: Pixel size in the Y direction
    :param epsg: EPSG code for the coordinate system (e.g., 4326 for WGS84)
    """

    # Ensure that start_x < end_x and start_y > end_y
    if start_x >= end_x:
        raise ValueError("start_x must be less than end_x (minx < maxx).")
    if start_y <= end_y:
        raise ValueError("start_y must be greater than end_y (maxy > miny).")

    # Define the bounding box as a string
    bbox = f"{start_x} {end_y} {end_x} {start_y}"  # xmin ymin xmax ymax
    print(bbox)
    # Call gdalwarp with subprocess to overwrite the file and specify the EPSG
    gdalwarp_command = [
        "gdalwarp",
        "-overwrite",  # Overwrite the output file
        "-te", str(start_x), str(end_y), str(end_x), str(start_y),  # Pass bounding box values as separate arguments
        "-tr", str(spacing_x), str(spacing_y),  # Set the resolution (pixel size in x and y directions)
        "-r", "nearest",  # Resampling method (you can change to nearest, cubic, etc.)
        "-t_srs", f"EPSG:{epsg}",  # Specify the target CRS using EPSG code
        "-of", "GTiff",  # Output format (GeoTIFF)
        input_file,  # Input GeoTIFF file
        output_file,  # Overwrite the input file
    ]

    # Run the gdalwarp command
    print(gdalwarp_command)
    gdalwarp_command_str = ' '.join(gdalwarp_command)
    print(gdalwarp_command_str)
    subprocess.run(gdalwarp_command, check=True)