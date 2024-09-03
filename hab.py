import geopandas
from pathlib import Path

from river_profile_lidar.bathymetry.hab.create_hab_shade import (
    process_and_save_hab,
)


RGB_IMAGE_FOLDER_PATH = "D:/SERIF/ESC/PHOTOS/"
TRANSECT_PATH = (
    "../data/transects/Transects_Level_2_ESC/Transects_Level_2_ESC_q_split.shp"
)
TRAPEZOID_FOLDER_PATH = "../data/bathymetry/trapezoid/"
RASTER_RESOLUTION = 0.3
APPLY_SOBEL = True
SHADE_KWARGS = {
    "shade_mask_path": "../data/shade/MASK_HAB_ESC/MASK_HAB_ESC.shp",
    "cross_section_points_path": "../data/cross_section/cross_section_points.shp",
}
OUTPUT_PATH = "../data/bathymetry/hab3/"

if __name__ == "__main__":
    image_to_process_df = geopandas.read_file(
        "D:/SERIF/ESC/PHOTOS/Index/Index_Ortho_utilises_ESC.shp"
    )
    image_to_process_df = image_to_process_df.sort_values(by="Hierarchie").reset_index(
        drop=True
    )
    image_name_to_process_list = image_to_process_df["NOM_IMAGE"].to_list()
    number_image_to_process = len(image_name_to_process_list)
    for i, image_name in enumerate(image_name_to_process_list):
        print(f"image: {image_name} {i}/{number_image_to_process}")
        if Path(OUTPUT_PATH, f"{Path(image_name).stem}_hab.tif").exists():
            continue
        rgb_image_path = Path(RGB_IMAGE_FOLDER_PATH, image_name)
        SHADE_KWARGS["trapezoid_path"] = Path(
            TRAPEZOID_FOLDER_PATH, f"{Path(image_name).stem}_trapezoid.tif"
        )
        process_and_save_hab(
            rgb_image_path=rgb_image_path,
            transect_path=TRANSECT_PATH,
            apply_sobel=APPLY_SOBEL,
            shade_kwargs=SHADE_KWARGS,
            raster_resolution=RASTER_RESOLUTION,
            output_path=OUTPUT_PATH,
        )
