"""
Crop images according to an index file that contains a HIERARCHIE column.

Images are processed in ascending order of hierarchy, meaning lower-priority images are cropped first,
ensuring they do not overlap with higher-priority images.
This is done so no image overlap and it makes next steps more straightfoward.
"""
import geopandas
from shapely import overlaps
from pathlib import Path
from tqdm import tqdm
import logging
import hydra
from omegaconf import DictConfig

from river_profile_lidar.crop_image.utils import crop_image_under_from_2_images
from river_profile_lidar.river_name_mapping import (
    get_river_save_name_from_index,
)


log = logging.getLogger(__name__)

DELETE_UNUSED_IMAGE_IN_FOLDER = False


@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def run(cfg: DictConfig):
    log.info(f"Doing Crop Image for: {cfg.river.name}")
    river_name = cfg.river.general_river_name
    index_file_path = cfg.river.index_file_path
    image_folder_path = cfg.river.rgb_image_folder_path
    image_to_process_df = geopandas.read_file(index_file_path)
    image_to_process_df = image_to_process_df.sort_values("HIERARCHIE").reset_index(
        drop=True
    )
    get_river_save_name_from_index_river = get_river_save_name_from_index(river_name)
    image_to_process_df["nom_image_save"] = image_to_process_df["NOM_IMAGE"].apply(
        get_river_save_name_from_index_river
    )
    if DELETE_UNUSED_IMAGE_IN_FOLDER:
        for image_path in filter(
            lambda path: ".tif" in str(path), Path(image_folder_path).iterdir()
        ):
            if (
                image_path.name.split(".")[0]
                not in image_to_process_df["nom_image_save"].values
            ):
                image_path.unlink()

    for image_under_index, image_under_row in tqdm(
        image_to_process_df[:-1].iterrows(), total=len(image_to_process_df[:-1])
    ):
        image_under_name = image_under_row["nom_image_save"]
        if Path(image_folder_path, image_under_name).exists() is False:
            continue
        for image_over_index, image_over_row in image_to_process_df[
            image_under_index + 1 :
        ].iterrows():
            image_over_name = image_over_row["nom_image_save"]
            if Path(image_folder_path, image_over_name).exists() is False:
                continue
            if overlaps(image_under_row.geometry, image_over_row.geometry):
                crop_image_under_from_2_images(
                    Path(image_folder_path, image_under_name),
                    Path(image_folder_path, image_over_name),
                )


if __name__ == "__main__":
    run()
