# neuromast3d
Work in progress code for the neuromast cell/nuclear shape analysis project

This code is built around the [cvapipe_analysis](https://github.com/AllenCell/cvapipe_analysis)
package from the Allen Institute for Cell Science. It is used to prepare a 
cell dataset for this pipeline, and also contains some functions to 
visualize the results.

## How to use this code
Currently, the code is run as a series of individual steps.
The order of the pipeline goes like so:
 - segmentation (optional)
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

## prep_single_cells
This step is based on [this code](https://github.com/AllenCell/cvapipe/blob/master/cvapipe/utils/prep_analysis_single_cell_utils.py) 
also created by AICS. The main script for this takes directories containing 
raw and segmented images of whole neuromasts, crops and interpolates them, and 
saves each of the single cropped cells.

## alignment
Different scripts to align cells can be found here. This step is optional, as 
you could opt not to align your cells, use your own alignment method, or use 
the default method of alignment in cvapipe_analysis (align the cells to the 
long axis in xy). In my case, I want to align the cells to a meaningful axis 
from the organism's perspective (e.g. AP or DV). 
