import geopandas
from pathlib import Path

from river_profile_lidar.bathymetry.trapezoid.create_trapezoid import (
    process_and_save_trapezoid_bathymetry,
)


RGB_IMAGE_FOLDER_PATH = "D:/SERIF/MAN/PHOTOS/"
TRANSECT_PATH = "../data/transects/Transects_Level_2_MAN/Transects_Level_2_MAN.shp"
INDEX_FILE_PATH = "D:/SERIF/MAN/PHOTOS/Index/Index_Ortho_utilisees.shp"
CROSS_SECTIONS_POINTS_PATH = "../data/cross_section/MAN/cross_section_points.shp"
RASTER_RESOLUTION = 0.3
OUTPUT_PATH = "../data/bathymetry/trapezoid/MAN"


if __name__ == "__main__":
    image_to_process_df = geopandas.read_file(INDEX_FILE_PATH)
    image_to_process_df = image_to_process_df.sort_values(by="Hierarchie").reset_index(
        drop=True
    )
    image_name_to_process_list = image_to_process_df["NOM_IMAGE"].to_list()
    number_image_to_process = len(image_name_to_process_list)
    for i, image_name in enumerate(image_name_to_process_list):
        image_name = f"{image_name}_4bandes_ortho.tif"
        print(f"image: {image_name} {i}/{number_image_to_process}")
        if Path(OUTPUT_PATH, f"{Path(image_name).stem}_trapezoid.tif").exists():
            continue
        rgb_image_path = Path(RGB_IMAGE_FOLDER_PATH, image_name)
        process_and_save_trapezoid_bathymetry(
            rgb_image_path=rgb_image_path,
            transect_path=TRANSECT_PATH,
            cross_sections_points_path=CROSS_SECTIONS_POINTS_PATH,
            raster_resolution=RASTER_RESOLUTION,
            output_path=OUTPUT_PATH,
        )
