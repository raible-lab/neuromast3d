from pathlib import Path
from aicsimageio.writers import ome_tiff_writer
import napari
import numpy as np
from skimage import data

cells_1 = data.cells3d()
cells_2 = cells_1.copy()
list_of_images = [cells_1, cells_2]


for img in list_of_images:
    viewer = napari.Viewer()

    @viewer.bind_key('s')
    def save_layer(viewer):
        temp_labels = viewer.layers[1].data
        reader = ome_tiff_writer.OmeTiffWriter(path, overwrite_file=True)
        reader.save(temp_labels)

    image_layer = viewer.add_image(img)
    labels_layer = viewer.add_labels(np.zeros_like(data.cells3d()))
    path = Path('./test.tiff')

    napari.run()
