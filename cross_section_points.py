import geopandas

from river_profile_lidar.bathymetry.trapezoid.cross_section_points.create_cross_section_points import (
    create_cross_section_points,
)


TRANSECT_DATA_PATH = "../transect_project/data/new_transect/Transects_Level_2_ESC/Transects_Level_2_ESC_q_split.shp"
SAVING_FOLDER_PATH = "../data/cross_section/"


if __name__ == "__main__":
    data = geopandas.read_file(TRANSECT_DATA_PATH)
    create_cross_section_points(data, SAVING_FOLDER_PATH)
