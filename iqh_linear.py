import geopandas
import pickle

from river_profile_lidar.bathymetry.hab.utils import calculate_da
from habitat_model import HabitatModel  # noqa: F401


TRANSECT_PATH = "E:/SERIF/MAP/Linear_mathias_2024/Linear/Transects_Tr_N2_MAP.shp"
OUTPUT_TRANSECT_PATH = "E:/SERIF/MAP/HAB_2024/MAP_Pr/iqh_linear/Transects_Tr_N2_MAP.shp"


if __name__ == "__main__":
    linear_df = geopandas.read_file(TRANSECT_PATH)
    linear_df = linear_df[linear_df["Accessible"] == "True"]
    linear_df["Slope"] = linear_df["Slope"].astype(float)

    min_non_zero_slope = linear_df[linear_df["Slope"] > 0]["Slope"].min()
    zero_or_nan_slope_idx = (linear_df["Slope"].isna()) | (linear_df["Slope"] == 0)
    linear_df.loc[zero_or_nan_slope_idx, "Slope"] = min_non_zero_slope

    linear_df = linear_df[linear_df["Slope"] > 0].reset_index(drop=True)

    linear_df["DA"] = linear_df.apply(
        lambda row: calculate_da(row["Q_IMG"], row["Slope"], row["WIDTH"]),
        axis=1,
    )
    linear_df["Vi"] = linear_df["Q_IMG"] / linear_df["WIDTH"]

    with open("habitat_model.pkl", "rb") as f:
        habital_model = pickle.load(f)

    linear_df["iqhp_1d"] = linear_df["D84"].apply(
        habital_model.get_hist_1d_estimation_from_value
    )
    linear_df["iqhp_2d"] = linear_df.apply(
        lambda row: habital_model.get_hist_2d_estimation_from_value(
            row["DA"], row["Vi"]
        ),
        axis=1,
    )
    linear_df["iqhp"] = linear_df["iqhp_1d"] * linear_df["iqhp_2d"]
    linear_df["surface"] = linear_df["WIDTH"] * 5
    linear_df["UPHP"] = linear_df["iqhp"] * linear_df["surface"]
    schema = {"properties": {"FACC_max": "float:20.5", "FACC_COR": "float:20.5"}}
    linear_df.to_file(OUTPUT_TRANSECT_PATH, driver="ESRI Shapefile", engine="fiona")
