# neuromast3d
Work in progress code for the neuromast cell/nuclear shape analysis project

This code is built around the [cvapipe_analysis](https://github.com/AllenCell/cvapipe_analysis)
package from the Allen Institute for Cell Science. It can be used to prepare a 
cell dataset for this pipeline. It also contains some functions to 
visualize the results and some scripts for cell and nucleus segmentation.

## How to use this code
Currently, the code can be run as a workflow (using the provided config.yaml file),
or as a series of individual steps.
The order of the pipeline goes like so:
 - segmentation (optional, broken into "nucleus" and "cell" steps)
 - create_fov_dataset
 - prep_single_cells
 - alignment (optional)
 - cvapipe_analysis (run the loaddata, computefeatures, preprocessing, and shapemodes steps in that order)
 - visualization

Each of these steps is summarized below.

## segmentation
Generate cell and nuclear instance segmentation using the watershed algorithm, 
and clean up the results in napari. Note: this step is optional - you can use 
your own method of choice to generate the cell and nuclear instance 
segmentations, although you must follow some restrictions. (to be described 
later)

## create_fov_dataset
This step takes directories containing raw and segmented images of the entire
fov (i.e. individual neuromasts) and creates a dataframe that stores info
about each fov image, including the paths to where they are stored on the 
local filesystem.

## prep_single_cells
This step is based on [this code](https://github.com/AllenCell/cvapipe/blob/master/cvapipe/utils/prep_analysis_single_cell_utils.py) 
also created by AICS. It takes the fov_dataset.csv generated by the
create_fov_dataset as input. From there, it reads each fov image, resizes it 
to isotropic pixel dimensions, and then crops and interpolates each cell
within the image. The single cell images are saved into new directories for each
fov image.

This step also generates a cell_manifest.csv that can be used directly as input
to cvapipe_analysis, if desired.

## alignment
Different scripts to align cells can be found here. This step is optional, as 
you could opt not to align your cells, use your own alignment method, or use 
the default method of alignment in cvapipe_analysis (align the cells to the 
long axis in xy). It takes a cell_manifest.csv as input and returns saved 
single cells that have been aligned/rotated using some strategy. A 
cell_manifest.csv that points to the aligned cells is also generated and can
be used as input to cvapipe_analysis.

Available alignement methods (TBD):
    - xy_xz
    - xy_xz_yz

# Current steps to reproduce 
This will likely be updated in the future as I look into things like using
setuptools for installation. For now, though...

(Note: these instructions assume a working conda installation and some 
knowledge of how to use git/GitHub.)

## Option 1: Using the conda env yaml file (for highly specific dependencies)
1. Clone the `neuromast3d` repository to your local machine using the `git 
clone` command.  To clone this branch, use `git clone --branch`.
2. Create an environment using the YAML file provided within the repository. 
Open a terminal, navigate to the `neuromast3d` directory, and run the command 
`conda env create --file neuromast3d_env1.yml`.
3. Activate the environment by running `conda activate neuromast3d_env1`. (Note 
that this environment should work for most workflow steps, but currently the 
visualization script requires a different environment and will not be able to 
work in this one.)
4. Edit the provided `config.yaml` file to point to the images you want to 
process and which steps you want to use.
5. Run the command `export PYTHONPATH="/path/to/neuromast3d"`. This environment 
variable is necessary for Python to be able to "see" the modules and functions 
associated with this project. It will not be necessary once I configure this 
project to be an installable package.
6. To run multiple steps as a workflow, set the `state` of those steps to "True" 
within the config file, and then run the command `python -m neuromast3d.run_neuromast3d.py neuromast3d/config.yaml`. 
(Note: you must be within the root `neuromast3d` directory to run this as written.)
You can optionally run single steps from the command line.

## Option 2: Using pip install (more general option, but less tested)
1. Clone the `neuromast3d` repository to your local machine using the `git 
clone` command.
2. Create a conda environment with python 3.8 installed by running the command 
`conda env create -n neuromast3d_env python=3.8`.
3. Activate the environment by running `conda activate neuromast3d_env`.
4. Navigate to the `neuromast3d` root directory and run `pip install .` to 
install dependencies. If you would like to have an editable install, run `pip 
install -e .`. (I need instructions for installing extras, TBD). To run 
napari, you will also need to run `pip install PyQt5`.
5. Proceed with editing the `config.yaml` file as described above.
6. Run the workflow by using the command `run_neuromast3d /path/to/config.yaml`

Please report any issues you have either by contacting me or opening an 
issue on Github. Of note, this code has currently only been tested on Ubuntu 
18.04 and 20.04, so I do not know how well it works on other operating 
systems.

## Contributing guidelines
I welcome suggestions and contributions to this project. If you would like to 
contribute, I ask that you please do not push to the main branch. Instead, 
create your own branch and open a pull request so that I may review the changes
first. 

## Testing (WIP)
Unit tests are found in the `tests` directory. If you would like to run unit 
tests on your installation, first install the test dependencies by running 
`pip install -e .[test]`. Some tests require testing data - as I do not 
currently have a way of distributing these, you may run the remainder of the 
tests using `pytest -m "not uses_data"`. I am also exploring the use of `tox` 
to run tests - stay tuned for more info on that.
