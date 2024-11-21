import os
import zipfile
import subprocess
import argparse
from ruamel.yaml import YAML
from dataclasses import dataclass

@dataclass
class LatLonBbox:
    left: float
    bottom: float    
    right: float
    top: float

    def get_as_str(self) -> str:
        """
        Return lat/lon bounding box coordinates in a single string,
        with orientation: [left  bottom  right top].
        
        Example: '-156 18.8 -154.7 20.3'
        """
        return f"{self.left} {self.bottom} {self.right} {self.top}"


def pad_bbox(bbox: LatLonBbox, pad: float = 1.0) -> LatLonBbox:
    """
    Get a copy of the input LatLonBbox, but uniformly padded/enlarged.

    Warning: Only correct for North American coordinates.
    """
    return LatLonBbox(
        left=bbox.left - pad,
        bottom=bbox.bottom - pad,
        right=bbox.right + pad,
        top=bbox.top + pad
    )


def compute_union_bbox(ll1: LatLonBbox, ll2: LatLonBbox) -> LatLonBbox:
    """
    Compute union of two LotLonBbox objects in N. America.

    Warning: Only correct for North American coordinates.
    """
    top = max(ll1.top, ll2.top)
    bottom = min(ll1.bottom, ll2.bottom)

    left = max(ll1.left, ll2.left)
    right = min(ll1.right, ll2.right)

    return LatLonBbox(left=left, bottom=bottom, right=right, top=top)
        

def make_dem(bbox: LatLonBbox, dem_path: str) -> None:
    """
    Generate a SRTM DEM Gtiff for the given bounding box.
    
    Parameters
    ----------
    bbox : LatLonBbox
        Lat/lon bounding box to get the DEM for.
    dem_path : str
        Filepath to the generated DEM Gtiff.
        Suggested format: "<out_dir>/dem.tif"
    """
    # Annoyingly, rasterio cannot find the PROJ_DATA directory
    # when running in the NASA MAAP ADE.
    # So, we need to manually set the environment variable, and
    # then run sardem

    # Step 1: Get the path to PROJ_DATA.
    #     From Command Line, use the command: echo $PROJ_DATA
    #     Example outputs:
    #       In conda base environment in MAAP ADE, this produces: /opt/conda/share/proj
    #       In a custom conda environment named 'dem', this produces: '/opt/conda/envs/dem/share/proj'
    result = subprocess.run(['echo $PROJ_DATA'], stdout=subprocess.PIPE, shell=True)
    proj_data_path = result.stdout.decode('utf-8').strip()
    
    os.environ['PROJ_DATA'] = proj_data_path
    
    # Step 2: Run sardem
    os.system(f"sardem --bbox {bbox.get_as_str()} --data-source COP -o {dem_path} --output-format GTiff")

    # Warning: in a Jupyter notebook on NASA MAAP ADE, Steps 1 and 2 must be combined:
    #     !PROJ_DATA={proj_data_path} sardem --bbox {bbox} --data-source COP -o {dem_file} --output-format GTiff


def unzip_file(input_zip_path, output_dir):
    """
    Unzips the specified .zip file into the given output directory.

    Example usage:
    output_files = unzip_file('path/to/your/input.zip', 'path/to/output/directory')
    print("Extracted files:", output_files)

    Parameters
    ----------
    input_zip_path : str
        The path to the input .zip file.
    output_dir : str
        The path to the output directory.
    
    Returns
    -------
    unzipped_dir : str
        The path to the unzipped directory.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the .zip file
    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Get the basename of the input zip file without the .zip extension
    base_name = os.path.splitext(os.path.basename(input_zip_path))[0]
 
    unzipped_dir = os.path.join(output_dir, base_name)
    
    return unzipped_dir

def alos1_to_l0b(input_alos1_path, output_l0b_path):
    """
    Repackages the specified ALOS1 granule into a NISAR-formatted L0B HDF5.

    Parameters
    ----------
    input_alos1_path : str
        The path to the input .zip file.
    output_l0b_path : str
        The path to the output L0B file. Should use the `.h5` extension.
    
    Returns
    -------
    return_code : int
        The return code from the shell command. 0 indicates success.
    """
    # Construct the command
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    command = [
               'python',
               # '-m',
               os.path.join(current_file_directory, 'alos_to_nisar_l0b.py'), 
               '-i', input_alos1_path,
               '-o', output_l0b_path
              ]
    
    # Run the command and wait for it to complete
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred during alos1_to_l0b: {result.stderr}")
    
    return result.returncode

def generate_rslc_runconfig(input_l0b_path, output_rslc_path, dem_file, out_dir):

    ## STEP 1: Create the RSLC runconfig
    # Read in the template
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    rncfg_template = os.path.join(current_file_directory, "focus.yaml")
    
    # Load the default dswx_hls.yaml template
    yaml = YAML()
    with open(rncfg_template, 'r') as rc:
        runconfig = yaml.load(rc)

    # Update the yaml with the desired arguments
    runconfig['runconfig']['groups']['input_file_group']['input_file_path'] = \
                [input_l0b_path]

    runconfig['runconfig']['groups']['dynamic_ancillary_file_group']['dem_file'] = \
                dem_file

    runconfig['runconfig']['groups']['product_path_group']['sas_output_file'] = \
                output_rslc_path

    # save into the output directory
    runconfig_path = os.path.join(out_dir,'rslc.yaml')
    with open(runconfig_path, 'w') as rc:
        yaml.dump(runconfig, rc)

    return runconfig_path


def l0b_to_rslc(input_l0b_path, output_rslc_path, dem_file):
    """
    Focuses the NISAR-formatted L0B granule into a NISAR-formatted RSLC HDF5.

    Parameters
    ----------
    input_l0b_path : str
        The path to the input .zip file.
    output_rslc_path : str
        The path to the output RSLC file. Should use the `.h5` extension.
    
    Returns
    -------
    return_code : int
        The return code from the shell command. 0 indicates success.
    """
    
    # Construct the runconfig
    base_path = os.path.dirname(output_rslc_path)
    rslc_rncfg = generate_rslc_runconfig(input_l0b_path, output_rslc_path, dem_file, base_path)
    
    # Construct the command
    command = [
                'python',
                '-m',
                'nisar.workflows.focus',
                rslc_rncfg
              ]
    
    # Run the command and wait for it to complete
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred during l0b_to_rslc: {result.stderr}")
    
    return result.returncode

def generate_gcov_runconfig(
    input_rslc_path: str,
    output_gcov_path: str,
    dem_file_path: str,
    gcov_runconfig_path: str,
    bbox: LatLonBbox,
    posting: float
    ) -> str:
    """
    Generate a runconfig for GCOV processer.

    TODO: Have separate tunable parameters for Freq A and Freq B.
    For now, ALOS-1 only generates products with Freq A,
    so we can ignore freq B. Must update in the future.
    
    Parameters
    ----------
    input_rslc_path : str
        Path to the input RSLC HDF5 file.
    output_gcov_path : str
        Path to the output GCOV file. Should use the `.h5` extension.
    dem_file_path : str
        Path to DEM file that is usable by GCOV workflow. (.tif, .vrt, etc.).
    gcov_runconfig_path : str
        Filename (with path) of where to save the runconfig.
        Should use the .yaml extension.
    bbox : LatLonBbox
        Lat/lon bounding box to use for the GCOV Freq A and Freq B.
    posting : float
        Pixel posting (square pixels) for Freq A and Freq B,
        in same units as DEM's EPSG and bbox.
    """

    # Construct the command
    posting = str(posting)
    command = [
                'python',
                '-m',
                'nisar.workflows.dumpconfig',
                'gcov',
                '-d', dem_file_path,
                '-o', output_gcov_path,
                '--a-spacing', posting, posting,
                '--top-left', str(bbox.left), str(bbox.top),
                '--bottom-right', str(bbox.right), str(bbox.bottom),
                '--out-runconfig', gcov_runconfig_path,
                '--validate',
                input_rslc_path
              ]
    
    # Run the command and wait for it to complete
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred during generate_gcov_runconfig: {result.stderr}")
    

def rslc_to_gcov(
    input_rslc_path: str,
    output_gcov_path: str,
    dem_file_path: str,
    bbox: LatLonBbox,
    posting: float,
) -> int:
    """
    Focuses the NISAR-formatted RSLC granule into a NISAR-formatted GCOV HDF5.

    Parameters
    ----------
    input_rslc_path : str
        Path to the input RSLC HDF5 file.
    output_gcov_path : str
        Path to the output GCOV file. Should use the `.h5` extension.
    dem_file_path : str
        Path to DEM file that is usable by GCOV workflow. (.tif, .vrt, etc.)
    bbox : LatLonBbox
        Lat/lon bounding box.
    posting : float
        Pixel posting (square pixels) for Freq A and Freq B,
        in same units as DEM's EPSG and bbox.
    
    Returns
    -------
    return_code : int
        The return code from the shell command. 0 indicates success.
    """
    
    # save into the output directory
    out_dir = os.path.dirname(output_gcov_path)
    runconfig_path = os.path.join(out_dir, 'gcov.yaml')

    # Construct the runconfig
    generate_gcov_runconfig(
        input_rslc_path=input_rslc_path,
        output_gcov_path=output_gcov_path,
        dem_file_path=dem_file_path,
        gcov_runconfig_path=runconfig_path,
        bbox=bbox,
        posting=posting
    )
    
    # Construct the command
    command = [
                'python',
                '-m',
                'nisar.workflows.gcov',
                runconfig_path
              ]
    
    # Run the command and wait for it to complete
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise Exception(f"Error occurred during rslc_to_gcov: {result.stderr}\n\n{result.stdout}")
    
    return result.returncode


def parse_args():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser()

    # parser.add_argument("-v", "--version", action="version", version=__version__)

    parser.add_argument("-i",
                        "--in-file",
                        dest="in_file",
                        metavar="INPUT_FILE",
                        type=str,
                        help="Input filename with path. Must be of type `in_type`.",
                        required=True,
                       )

    parser.add_argument(
        "--dem",
        dest="dem",
        metavar="DEM_FILE",
        type=str,
        help="Path to DEM file. If not provided, will be fetched per a padded `bbox`.",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        metavar="OUTPUT_DIRECTORY",
        type=str,
        help="Path to output directory to store generated files",
        required=True,
    )

    parser.add_argument(
        "--bbox",
        type=float,
        help="lat/lon bounding box (example: --bbox -118.068 34.222 -118.058 34.228). Orientation: left, bottom, right, top",
        nargs=4,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        required=True,
    )

    parser.add_argument(
        "--in_type",
        dest="in_type",
        metavar="INPUT_PRODUCT_TYPE",
        choices=["alos1", "l0b", "rslc"],
        type=str,
        help='Product type of the input file. One of: "alos1", "l0b", "rslc". Will be processed through the ALOS1->L0B->RSLC->GCOV pipeline, ending with type specified by `out_type`.',
        required=True,
    )

    parser.add_argument(
        "--out_type",
        dest="out_type",
        metavar="OUTPUT_PRODUCT_TYPE",
        choices=["l0b", "rslc", "gcov"],
        type=str,
        help='Product type of the output file. One of: "l0b", "rslc", "gcov". In combination with `in_type`, determines which processing steps occur.',
        required=True,
    )

    parser.add_argument(
        "--posting",
        "--gcov_posting",
        type=float,
        default=0.000833334,
        help="Posting for pixel spacing for GCOV (square pixels), in same units as DEM's EPSG and bbox. (100 meters is ~0.000833334 degrees.) Defaults to 0.000833334.",
        metavar="GCOV_POSTING",
    )

    return parser.parse_args()


def main() -> None:
    """
    TODO
    """
    # Parse arguments
    args = parse_args()
    in_file = args.in_file
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    ## Temporary hack: Download input file from http URL inside the algorithm
    in_dir = args.out_dur
    in_dir = in_dir.replace("output", "input")
    os.makedirs(in_dir, exist_ok=True)
    from maap.maap import MAAP
    maap = MAAP()
    in_file = maap.downloadGranule(
        online_access_url=in_file,
        destination_path=in_dir
    )
    ## End temporary hack    
    
    in_type = args.in_type
    out_type = args.out_type

    # file basename, without directory and without extension.
    file_base_name = in_file_base_name = os.path.splitext(
            os.path.basename(in_file)
        )[0]
    file_base_name = file_base_name.split(".")[0]
    file_base_name = file_base_name.removeprefix("L0B_")
    file_base_name = file_base_name.removeprefix("RSLC_")

    # Unpack ALOS-1 zip file
    if in_type == "alos1":
        unzipped_alos1_path = unzip_file(args.in_file, out_dir)

        # Repackage ALOS-1 to L0B
        l0b_path = os.path.join(out_dir, f"L0B_{file_base_name}.h5")
        l0b_file = alos1_to_l0b(
            input_alos1_path=unzipped_alos1_path,
            output_l0b_path=l0b_path
        )
    elif in_type == "l0b":
        l0b_file = args.in_file
    else:
        assert in_type == "rslc"
        l0b_file = None

    if out_type == "l0b":
        # Exit early
        return

    gcov_bbox = LatLonBbox(
        left=args.bbox[0],
        bottom=args.bbox[1],
        right=args.bbox[2],
        top=args.bbox[3]
    )
    if args.dem is None:
        # Make dem.tif (required for both RSLC and GCOV processing)
        padded_bbox = pad_bbox(bbox=gcov_bbox, pad=1.0)
        dem_file = os.path.join(out_dir, f"dem.tif")
        make_dem(padded_bbox, dem_file)
    else:
        dem_file = str(args.dem)

    # Focus L0B to RSLC
    if in_type in ("alos1", "l0b"):
        rslc_file = os.path.join(out_dir, f"RSLC_{file_base_name}.h5")
        l0b_to_rslc(input_l0b_path=l0b_file,
                                  output_rslc_path=rslc_file,
                                  dem_file=dem_file)
    else:
        assert in_type == "rslc"
        rslc_file = args.in_file

    if out_type == "rslc":
        # Exit early
        return

    # Step 6: Process RSLC to GCOV
    gcov_file = os.path.join(out_dir, f"GCOV_{file_base_name}.h5")
    rslc_to_gcov(
        input_rslc_path = rslc_file,
        output_gcov_path = gcov_file,
        dem_file_path = dem_file,
        bbox = gcov_bbox,
        posting = args.posting,
    )
    

if __name__ == "__main__":
    main()
