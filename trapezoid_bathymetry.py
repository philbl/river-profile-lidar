from river_profile_lidar.bathymetry.trapezoid.create_trapezoid import (
    process_and_save_trapezoid_bathymetry,
)


RGB_IMAGE_PATH = "../data/img/Q12007_479_4bandes_ortho.tif"
TRANSECT_PATH = (
    "../data/transects/Transects_Level_2_ESC/Transects_Level_2_ESC_q_split.shp"
)
CROSS_SECTIONS_POINTS_PATH = "../data/cross_section/cross_section_points.shp"
RASTER_RESOLUTION = 0.3
OUTPUT_PATH = "../data/bathymetry/trapezoid/trapezoid.tif"


if __name__ == "__main__":
    process_and_save_trapezoid_bathymetry(
        rgb_image_path=RGB_IMAGE_PATH,
        transect_path=TRANSECT_PATH,
        cross_sections_points_path=CROSS_SECTIONS_POINTS_PATH,
        raster_resolution=RASTER_RESOLUTION,
        output_path=OUTPUT_PATH,
    )
