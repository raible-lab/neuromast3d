#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from magicgui import magicgui
import napari
from napari.layers import Labels
import pandas as pd

from neuromast3d.step_utils import read_raw_and_seg_img


def read_fov_dataset(project_dir: Path):
    path_to_dataset = project_dir / 'fov_dataset.csv'
    fov_df = pd.read_csv(path_to_dataset)
    return fov_df


class FovCurator:
    def __init__(self, fov_id):
        self.fov_id = fov_id
        self.excluded_cells = []
        self.hair_cells = []
        self.polarity = None

    def __pick_labels(
            self,
            picked_labels: list,
            labels: Labels,
            undo_last: bool = False
        ):
        # Expected to be decorated with magicgui when called
        if labels is not None:
            if undo_last:
                try:
                    picked_labels.pop()
                except IndexError:
                    pass

            else:
                label = labels.selected_label
                if label not in picked_labels:
                    picked_labels.append(label)

            return picked_labels

    def pick_bad_cells(self, labels: Labels, undo_last: bool = False):
        self.excluded_cells = self.__pick_labels(
                self.excluded_cells, 
                labels,
                undo_last
        )
        return self.excluded_cells

    def pick_hair_cells(self, labels: Labels, undo_last: bool = False):
        self.hair_cells = self.__pick_labels(
                self.hair_cells, 
                labels,
                undo_last
        )
        return self.hair_cells

    def indicate_polarity(self, axis: str):
        # Expected to be decorated with magicgui when called
        self.polarity = axis

    def info_to_dict(self):
        output_dict = {
                'fov_id': self.fov_id,
                'polarity': self.polarity,
                'cells_to_exclude': self.excluded_cells,
                'hair_cells': self.hair_cells
        }
        return output_dict


def main():
    parser = argparse.ArgumentParser(description='FOV curation script')
    parser.add_argument('project_dir')
    args = parser.parse_args()
    project_dir = Path(args.project_dir)
    fov_df = read_fov_dataset(project_dir)

    curated_fov_info = []
    for fov in fov_df.itertuples(index=False):
        print(f'Now viewing {fov.NM_ID}')
        raw_img, seg_img = read_raw_and_seg_img(
            fov.SourceReadPath,
            fov.SegmentationReadPath
        )

        raw_nuc = raw_img[fov.nucleus, :, :, :]
        raw_mem = raw_img[fov.membrane, :, :, :]
        seg_mem = seg_img[fov.cell_seg, :, :, :]

        viewer = napari.Viewer()
        viewer.add_image(raw_nuc)
        viewer.add_image(raw_mem)
        viewer.add_labels(seg_mem)

        curator = FovCurator(fov.NM_ID)
        polarity_indicator = magicgui(
                curator.indicate_polarity,
                axis={'choices': ['AP', 'DV']},
        )

        bc_picker = magicgui(curator.pick_bad_cells, result_widget=True)
        hc_picker = magicgui(curator.pick_hair_cells, result_widget=True)
        viewer.window.add_dock_widget(polarity_indicator, area='right')
        viewer.window.add_dock_widget(bc_picker, area='right', name='bad_cells')
        viewer.window.add_dock_widget(hc_picker, area='right', name='hair_cells')

        napari.run()

        curated_fov_info.append(curator.info_to_dict())

    # Will just save in project dir for now
    # May change how this works later
    curated_fov_df = pd.DataFrame(curated_fov_info)
    curated_fov_df.to_csv(project_dir / 'curated_fov_dataset.csv')


if __name__ == '__main__':
    main()
