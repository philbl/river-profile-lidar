import geopandas

from river_profile_lidar.bathymetry.trapezoid.cross_section_points.create_cross_section_points import (
    create_cross_section_points,
)


TRANSECT_DATA_PATH = "../data/transects/Transects_Level_2_MAN/Transects_Level_2_MAN.shp"
SAVING_FOLDER_PATH = "../data/cross_section/MAN"


if __name__ == "__main__":
    data = geopandas.read_file(TRANSECT_DATA_PATH)
    create_cross_section_points(data, SAVING_FOLDER_PATH)
