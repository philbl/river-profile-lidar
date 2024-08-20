from river_profile_lidar.bathymetry.hab.create_hab import (
    process_and_save_hab,
)


RGB_IMAGE_PATH = "../data/img/Q12007_479_4bandes_ortho.tif"
TRANSECT_PATH = (
    "../data/transects/Transects_Level_2_ESC/Transects_Level_2_ESC_q_split.shp"
)
RASTER_RESOLUTION = 0.3
APPLY_SOBEL = True
OUTPUT_PATH = "../data/bathymetry/hab/"
DEBUG_FILE = True


if __name__ == "__main__":
    process_and_save_hab(
        rgb_image_path=RGB_IMAGE_PATH,
        transect_path=TRANSECT_PATH,
        apply_sobel=APPLY_SOBEL,
        debug_file=DEBUG_FILE,
        raster_resolution=RASTER_RESOLUTION,
        output_path=OUTPUT_PATH,
    )
