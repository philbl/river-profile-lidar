import geopandas
import pickle

from river_profile_lidar.bathymetry.hab.utils import calculate_da
from habitat_model import HabitatModel  # noqa: F401


TRANSECT_PATH = "D:/SERIF/TRI/Transect_mathias_nov_2024/Transects_Level_2_TRI.shp"
OUTPUT_TRANSECT_PATH = (
    "D:/SERIF/TRI/HAB_2024/iqh_linear/surfacique_Transects_Level_2_TRI.shp"
)


if __name__ == "__main__":
    transect_polygon_df = geopandas.read_file(TRANSECT_PATH)
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Backwater"] == 0]
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Lac"] == 0]
    transect_polygon_df = transect_polygon_df[~transect_polygon_df["Slope"].isna()]
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Slope"] > 0]
    transect_polygon_df = transect_polygon_df.sort_values(by="PK").reset_index(
        drop=True
    )
    transect_polygon_df = transect_polygon_df.drop(
        columns=["FACC_COR", "FACC_CORR", "FACC_M2"]
    )

    transect_polygon_df["DA"] = transect_polygon_df.apply(
        lambda row: calculate_da(row["Q_IMG_spli"], row["Slope"], row["WAT_WIDTH"]),
        axis=1,
    )
    transect_polygon_df["Vi"] = (
        transect_polygon_df["Q_IMG_spli"] / transect_polygon_df["WAT_WIDTH"]
    )

    with open("habitat_model.pkl", "rb") as f:
        habital_model = pickle.load(f)

    transect_polygon_df["iqh_1d"] = transect_polygon_df["D84"].apply(
        habital_model.get_hist_1d_estimation_from_value
    )
    transect_polygon_df["iqh_2d"] = transect_polygon_df.apply(
        lambda row: habital_model.get_hist_2d_estimation_from_value(
            row["DA"], row["Vi"]
        ),
        axis=1,
    )
    transect_polygon_df["iqh"] = (
        transect_polygon_df["iqh_1d"] * transect_polygon_df["iqh_2d"]
    )

    transect_polygon_df.to_file(OUTPUT_TRANSECT_PATH)
