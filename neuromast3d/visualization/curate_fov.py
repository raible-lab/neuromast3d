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
        self.picked_labels = []
        self.polarity = None

    def pick_labels(self, labels: Labels, undo_last: bool = False):
        # Expected to be decorated with magicgui when called
        if labels is not None:

            if undo_last:
                try:
                    self.picked_labels.pop()
                except IndexError:
                    pass

            else:
                label = labels.selected_label
                if label not in self.picked_labels:
                    self.picked_labels.append(label)
            
            return self.picked_labels

    def indicate_polarity(self, axis: str):
        # Expected to be decorated with magicgui when called
        self.polarity = axis 

    def info_to_dict(self):
        output_dict = {
                'fov_id': self.fov_id,
                'polairty': self.polarity,
                'cells_to_exclude': self.picked_labels
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

        viewer = napari.Viewer()
        viewer.add_image(raw_img)
        viewer.add_labels(seg_img)

        curator = FovCurator(fov.NM_ID)
        polarity_indicator = magicgui(
                curator.indicate_polarity, 
                axis={'choices': ['AP', 'DV']},
        )
        label_picker = magicgui(curator.pick_labels, result_widget=True)
        viewer.window.add_dock_widget(polarity_indicator, area='right')
        viewer.window.add_dock_widget(label_picker, area='right')

        napari.run()

        curated_fov_info.append(curator.info_to_dict())

    # Will just save in project dir for now
    # May change how this works later
    curated_fov_df = pd.DataFrame(curated_fov_info)
    curated_fov_df.to_csv(project_dir / 'curated_fov_datset.csv')


if __name__ == '__main__':
    main()
