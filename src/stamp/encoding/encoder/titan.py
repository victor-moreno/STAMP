import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel

from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType, Microns, PandasLabel, SlideMPP

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = ["Ding, et al. (https://github.com/mahmoodlab/TITAN)"]

_logger = logging.getLogger("stamp")


class Titan(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        super().__init__(
            model=model,
            identifier=EncoderName.TITAN,
            precision=torch.float32,
            required_extractors=[ExtractorName.CONCH1_5],
        )

    def _generate_slide_embedding(
        self,
        feats: Tensor,
        device: DeviceLikeType,
        coords: CoordsInfo | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Helper method to encode a single slide."""
        if coords is None:
            raise ValueError("Coords must be provided.")

        coords_tensor = torch.tensor(coords.coords_um, dtype=self.precision)

        # Convert coordinates from microns to pixels
        patch_size_lvl0 = math.floor(256 / coords.mpp)  # Inferred from TITAN docs
        coords_px = coords_tensor / coords.mpp  # Convert to pixels
        coords_px = coords_px.to(torch.int64).to(device)  # Convert to integer

        feats = feats.to(device=device)

        with torch.inference_mode():
            slide_embedding = self.model.encode_slide_from_patch_features(
                feats, coords_px, patch_size_lvl0
            )
            return slide_embedding.detach().squeeze().cpu().numpy()

    def _generate_patient_embedding(
        self,
        feats_list: list,
        device: DeviceLikeType,
        coords_list: list[CoordsInfo] | None = None,
        **kwargs,
    ) -> np.ndarray:
        if coords_list is None:
            raise ValueError("coords_list must be provided.")

        # Concatenate all feature to a single slide tensor
        all_feats_cat = torch.cat(feats_list, dim=0).unsqueeze(0)

        # Create a single CoordsInfo item for the virtual slide
        # Already validated that mpp values are all equal within patient slides
        tile_size_um: Microns = coords_list[0].tile_size_um
        tile_size_px = coords_list[0].tile_size_px
        # Combine all slide coords to a single virtual slide set of coordinates
        coords_um = np.concatenate([coord.coords_um for coord in coords_list], axis=0)
        # Create virtual slide's Coords Info object
        coords = CoordsInfo(coords_um, tile_size_um, tile_size_px)

        return self._generate_slide_embedding(all_feats_cat, device, coords)

    def encode_patients_(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        patient_label: PandasLabel,
        filename_label: PandasLabel,
        device,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        """Generate one virtual slide concatenating all the slides of a
        patient over the x axis."""
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        # generate the name for the folder containing the feats
        if generate_hash:
            encode_dir = (
                f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}"
            )
        else:
            encode_dir = f"{self.identifier}-pat"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        for patient_id, group in (progress := tqdm(patient_groups)):
            progress.set_description(str(patient_id))

            # skip patient in case feature file already exists
            output_path = (encode_dir / str(patient_id)).with_suffix(".h5")
            if output_path.exists():
                _logger.debug(
                    f"skipping {str(patient_id)} because {output_path} already exists"
                )
                continue

            all_feats_list = []
            all_coords_list = []
            current_x_offset = 0
            slides_mpp = SlideMPP(-1)

            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)

                feats, coords = self._validate_and_read_features(h5_path=h5_path)

                # Get the mpp of one slide and check that the rest have the same
                if slides_mpp < 0:
                    slides_mpp = coords.mpp
                elif not math.isclose(slides_mpp, coords.mpp, rel_tol=1e-5):
                    raise ValueError(
                        "All patient slides must have the same mpp value. "
                        "Try reprocessing the slides using the same tile_size_um and "
                        "tile_size_px values for all of them."
                    )

                # Add the offset to tile coordinates in x axis
                for coord in coords.coords_um:
                    coord[0] += current_x_offset

                # get the coordinates of the rightmost tile and add the
                # tile width as these coordinates are from the top-left
                # point. With that you get the total width of the slide.
                current_x_offset = max(coords.coords_um[:, 0]) + coords.tile_size_um

                # Add tile feats and coords to the patient virtual slide
                all_feats_list.append(feats)
                all_coords_list.append(coords)

            if not all_feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            patient_embedding = self._generate_patient_embedding(
                all_feats_list, device, all_coords_list
            )
            self._save_features_(output_path=output_path, feats=patient_embedding)
