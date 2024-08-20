import geopandas
import numpy
from pathlib import Path

from river_profile_lidar.utils import (
    get_cropped_image_to_remove_black_padding,
    get_water_rgb_array_from_transect_df,
)
from river_profile_lidar.bathymetry.hab.utils import (
    add_pixel_distribution_to_transect_polygon_df,
    calculate_da,
    add_beta_and_max_pixel_value,
    rasterize_geometry,
    smooth_beta_and_max_value_with_touching_transect,
    reproject_raster,
    caculate_depth_value,
    detect_and_interpolate_bad_water_pixels,
    get_basic_out_meta_data_tiff,
    smooth_transect_edges,
    save_array_as_raster,
)

RGB_IMAGE_PATH = "../data/img/Q12007_479_4bandes_ortho.tif"
TRANSECT_PATH = (
    "../data/transects/Transects_Level_2_ESC/Transects_Level_2_ESC_q_split.shp"
)
RASTER_RESOLUTION = 0.3
APPLY_SOBEL = False
OUTPUT_PATH = "../data/bathymetry/hab/"
DEBUG_FILE = False


def process_and_save_hab(
    rgb_image_path,
    transect_path,
    apply_sobel,
    debug_file,
    raster_resolution,
    output_path,
):
    """
    Process bathymetry data by removing black padding from the image, extracting transect polygons,
    interpolating bad pixels, calculating depth estimates, and saving the results as raster files.

    Parameters:
    -----------
    rgb_image_path : str
        Path to the RGB image file.
    transect_path : str
        Path to the transect shapefile.
    apply_sobel : bool
        Whether to apply Sobel filtering and interpolation to remove bad pixels.
    debug_file : bool
        Whether to save intermediate debug files.
    raster_resolution : float
        The resolution of the raster output.
    output_path : str
        Path to save the final smoothed depth estimate raster file.
    """
    # Step 1: Load and preprocess the image and transects
    cropped_image, profile = get_cropped_image_to_remove_black_padding(rgb_image_path)
    transect_polygon_df = geopandas.read_file(transect_path)
    transect_polygon_df = transect_polygon_df[transect_polygon_df["Backwater"] == 0]
    transect_polygon_df = transect_polygon_df.sort_values(by="PK").reset_index(
        drop=True
    )

    # Step 2: Extract water RGB array and relevant data
    (
        water_rgb_array,
        affine_transform,
        transect_polygon_in_rgb,
        transect_polygon,
    ) = get_water_rgb_array_from_transect_df(cropped_image, transect_polygon_df)
    water_mask = (water_rgb_array > 0).all(axis=2)

    # Step 3: Detect and interpolate bad pixels if required
    if apply_sobel:
        (
            water_rgb_array,
            is_water_bad_pixel_mask,
        ) = detect_and_interpolate_bad_water_pixels(water_rgb_array, water_mask)

    # Step 4: Optionally save debug files
    if debug_file:
        water_rgb_array_bad_pixel = water_rgb_array.copy().astype(float)
        water_rgb_array_bad_pixel[~is_water_bad_pixel_mask] = None
        water_rgb_array_bad_pixel_mask = water_rgb_array_bad_pixel[:, :, 0]
        water_rgb_array_bad_pixel_mask[~numpy.isnan(water_rgb_array_bad_pixel_mask)] = 1
        water_rgb_array_bad_pixel_mask = numpy.atleast_3d(
            water_rgb_array_bad_pixel_mask
        ).transpose(2, 0, 1)
        out_meta = get_basic_out_meta_data_tiff()
        save_array_as_raster(
            water_rgb_array_bad_pixel_mask,
            out_meta,
            profile["crs"],
            affine_transform,
            Path(output_path, "bad_pixel_mask.tif"),
        )

        water_rgb_array_out = water_rgb_array.transpose(2, 0, 1)
        save_array_as_raster(
            water_rgb_array_out,
            out_meta,
            profile["crs"],
            affine_transform,
            Path(output_path, "interpolated_pixels.tif"),
        )

    # Step 5: Calculate transect pixel distribution and depth
    height, width, _ = water_rgb_array.shape
    transect_polygon_in_rgb = add_pixel_distribution_to_transect_polygon_df(
        transect_polygon_in_rgb, water_rgb_array, (height, width), affine_transform
    )

    transect_polygon_in_rgb["DA"] = transect_polygon_in_rgb.apply(
        lambda row: calculate_da(row["Q_IMG_spli"], row["Slope"], row["WAT_WIDTH"]),
        axis=1,
    )
    transect_polygon_in_rgb = add_beta_and_max_pixel_value(transect_polygon_in_rgb)
    transect_polygon_in_rgb = smooth_beta_and_max_value_with_touching_transect(
        transect_polygon_in_rgb, 7
    )

    # Step 6: Rasterize and reproject beta and max pixel values
    raster_beta_smoothed, transform_beta_smoothed = rasterize_geometry(
        transect_polygon_in_rgb, "beta_smoothed", raster_resolution=raster_resolution
    )
    raster_max_value_smoothed, transform_max_value_smoothed = rasterize_geometry(
        transect_polygon_in_rgb,
        "max_pixel_value_smoothed",
        raster_resolution=raster_resolution,
    )

    raster_beta_reproject_smoothed = reproject_raster(
        numpy.atleast_3d(raster_beta_smoothed).transpose(2, 0, 1),
        transform_beta_smoothed,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    raster_max_value_reproject_smoothed = reproject_raster(
        numpy.atleast_3d(raster_max_value_smoothed).transpose(2, 0, 1),
        transform_max_value_smoothed,
        profile["crs"],
        (1, height, width),
        affine_transform,
    )[0]

    # Step 7: Calculate depth estimate and smooth edges
    depth_estimate = caculate_depth_value(
        water_rgb_array[:, :, 0].astype(float),
        raster_max_value_reproject_smoothed,
        raster_beta_reproject_smoothed,
    )
    depth_estimate[~water_mask] = None
    depth_estimate[depth_estimate < 0] = 0

    depth_estimate_smoothed = smooth_transect_edges(
        transect_polygon_in_rgb,
        depth_estimate,
        raster_resolution,
        height,
        width,
        affine_transform,
        profile["crs"],
    )

    # Step 8: Save the final smoothed depth estimate raster
    if output_path:
        depth_estimate_smoothed_raster = numpy.atleast_3d(
            depth_estimate_smoothed
        ).transpose(2, 0, 1)
        out_meta = get_basic_out_meta_data_tiff()
        save_array_as_raster(
            depth_estimate_smoothed_raster,
            out_meta,
            profile["crs"],
            affine_transform,
            Path(output_path, "hab.tif"),
        )
