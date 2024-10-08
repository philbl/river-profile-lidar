import geopandas
from rasterio.features import geometry_mask
import rasterio
import numpy


from river_profile_lidar.utils import (
    get_water_rgb_array_from_transect_df,
    get_cropped_image_to_remove_black_padding,
)
from river_profile_lidar.bathymetry.trapezoid.utils import (
    combine_list_of_array_and_transform_in_one_raster,
    process_transects_to_create_z_grids,
    fix_nan_pixels_with_gaussian_convolution,
)


def process_and_save_trapezoid_bathymetry(
    rgb_image_path: str,
    transect_path: str,
    cross_sections_points_path: str,
    raster_resolution: float,
    output_path: str,
):
    """
    Process RGB image and transect data to create and save a bathymetry raster with NaN values fixed.

    This function performs a series of steps to process an RGB image and transect polygons,
    including cropping the image, extracting relevant water areas, generating Z-value grids,
    merging them into a single raster, and fixing NaN values using Gaussian convolution. The
    final result is saved as a GeoTIFF file.

    Parameters:
    -----------
    rgb_image_path : str
        The file path to the input RGB image (e.g., a GeoTIFF file).
    transect_path : str
        The file path to the shapefile containing transect polygons.
    cross_sections_points_path : str
        The file path to the shapefile containing cross sections points.
    RASTER_RESOLUTION : float
        The resolution of the raster grid used for Z-value interpolation.
    output_path : str
        The file path where the output bathymetry raster will be saved.

    Returns:
    --------
    None
        The function saves the final processed raster to the specified output path.
    """
    # Step 1: Crop image to remove black padding
    cropped_image, _ = get_cropped_image_to_remove_black_padding(rgb_image_path)

    # Step 2: Load transect polygons and points data
    transect_polygon_df = geopandas.read_file(transect_path)
    points = geopandas.read_file(cross_sections_points_path)

    # Step 3: Extract transect polygon than are in the image
    _, _, transect_polygon_in_rgb, _ = get_water_rgb_array_from_transect_df(
        cropped_image, transect_polygon_df
    )

    # Filter out transects that are Backwater
    transect_polygon_in_rgb = transect_polygon_in_rgb[
        transect_polygon_in_rgb["Backwater"] == 0
    ].reset_index(drop=True)

    # Step 4: Create Z-value grids for transects
    z_array_list, transform_list = process_transects_to_create_z_grids(
        transect_polygon_in_rgb, points, raster_resolution
    )

    # Step 5: Merge Z-value grids into a single raster
    merged_array, merged_transform = combine_list_of_array_and_transform_in_one_raster(
        z_array_list, transform_list
    )

    # Step 6: Create a mask for the water regions
    water_mask = geometry_mask(
        transect_polygon_in_rgb.geometry.values,
        out_shape=(merged_array.shape[0], merged_array.shape[1]),
        transform=merged_transform,
        invert=True,
    )

    # Step 7: Fix NaN values using Gaussian convolution
    merged_array_nan_fixed = fix_nan_pixels_with_gaussian_convolution(
        merged_array, water_mask
    )

    # Step 8: Define metadata and save the final raster
    out_meta = {
        "driver": "GTiff",
        "height": merged_array.shape[0],
        "width": merged_array.shape[1],
        "count": 1,
        "dtype": merged_array.dtype,
        "crs": points.crs,
        "transform": merged_transform,
        "nodata": numpy.nan,
    }
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(merged_array_nan_fixed, 1)
