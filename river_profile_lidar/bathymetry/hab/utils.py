from affine import Affine
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.utils.exceptions import AstropyUserWarning
import numpy
import pandas
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize, geometry_mask
from rasterio.warp import reproject, Resampling
from skimage.filters import sobel
from scipy.ndimage import gaussian_filter
from typing import Tuple
import warnings


def add_pixel_distribution_to_transect_polygon_df(
    transect_polygon_df: pandas.DataFrame,
    water_rgb_array: numpy.ndarray,
    image_height_width: Tuple[int, int],
    transform: Affine,
) -> pandas.DataFrame:
    """
    Add pixel distribution data to each transect polygon in a DataFrame.

    This function iterates over each transect polygon in the given DataFrame, extracts the pixel values from
    the water RGB array that fall within each polygon, and adds these pixel values as a new column in the DataFrame.

    Parameters:
    -----------
    transect_polygon_df : pandas.DataFrame
        A DataFrame containing transect polygons with a "geometry" column representing the polygon shapes.
    water_rgb_array : numpy.ndarray
        A 3D numpy array representing the RGB values of the water region.
    image_height_width : tuple
        A tuple (height, width) representing the dimensions of the water RGB array.
    transform : affine.Affine
        An affine transformation matrix that maps pixel coordinates to geographic coordinates.

    Returns:
    --------
    transect_polygon_df : pandas.DataFrame
        The input DataFrame with an additional column, "masked_subset_transect_rgb", containing
        the pixel values from the water RGB array that fall within each transect polygon.
    """
    height, width = image_height_width
    for i, row in tqdm(
        transect_polygon_df.iterrows(),
        total=len(transect_polygon_df),
        desc="Get Pixels by Transect",
    ):
        transect_polygon = row["geometry"]
        mask_transect_rgb = geometry_mask(
            [transect_polygon],
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
        masked_subset_transect_rgb = water_rgb_array[mask_transect_rgb]
        transect_polygon_df.loc[[i], "masked_subset_transect_rgb"] = pandas.Series(
            [masked_subset_transect_rgb], index=[i]
        )
    return transect_polygon_df


def calculate_da(q: float, slope: float, wetted_width: float) -> float:
    """
    Calculate the average depth (da) of a river or stream based on water flow, slope, and wetted width.

    This function uses a specific empirical formula to estimate the average depth of a river or stream
    given the water flow (q), slope, and wetted width.

    Parameters:
    -----------
    q : float
        The water flow (discharge) in the river or stream.
    slope : float
        The slope of the river or stream.
    wetted_width : float
        The wetted width of the river or stream, which is the width of the water surface across the channel.

    Returns:
    --------
    da : float
        The estimated average depth of the river or stream.
    """
    return (q / (3.125 * wetted_width * (slope**0.12))) ** 0.55


def calculate_beta(
    mean_log_pixel_value: float, dn_0: float, mean_depth: float
) -> float:
    """
    Calculate the attenuation coefficient (beta) based on the mean log pixel value, DN value at depth 0, and mean depth.

    This function calculates the attenuation coefficient (beta), which represents the rate at which light diminishes
    with depth in water. The calculation is based on the logarithm of the mean pixel value, the digital number (DN) value
    at depth 0, and the mean depth of the water.

    Parameters:
    -----------
    mean_log_pixel_value : float
        The mean logarithm of the pixel values from the water body.
    dn_0 : float
        The digital number (DN) value at depth 0.
    mean_depth : float
        The mean depth of the water body.

    Returns:
    --------
    beta : float
        The calculated attenuation coefficient, representing the rate of light attenuation in the water.
    """
    beta = -(mean_log_pixel_value - numpy.log(dn_0)) / mean_depth
    return beta


def caculate_depth_value(pixel_value: float, dn_0: float, beta: float) -> float:
    """
    Calculate the depth value from a pixel value, the digital number (DN) value at depth 0,
    and attenuation coefficient (beta).

    Parameters:
    -----------
    pixel_value : float
        The pixel value(s) from the image, representing the water reflectance.
    dn_0 : float
        The digital number (DN) value at depth 0.
    beta : float
        The attenuation coefficient, representing the rate of light attenuation in the water.

    Returns:
    --------
    depth : float
        The calculated depth value(s), based on the pixel value and attenuation coefficient.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth = -numpy.log(pixel_value / dn_0) / beta
    return depth


def calculate_estimated_q(width: float, mean_depth: float, slope: float) -> float:
    """
    Estimate the discharge (Q) of a river or stream based on channel width, mean depth, and slope.

    This function estimates the discharge (Q) using a formula that combines the channel width,
    mean depth, and slope of the river or stream. The formula is derived from hydraulic geometry
    relationships and is useful for estimating flow in natural channels.

    Parameters:
    -----------
    width : float
        The width of the river or stream channel.
    mean_depth : float
        The mean depth of the water in the channel.
    slope : float
        The slope of the river or stream bed.

    Returns:
    --------
    estimated_q : float
        The estimated discharge (Q) of the river or stream.
    """
    estimated_q = width * (mean_depth**1.83) * (slope**0.12) / 0.32
    return estimated_q


def add_beta_and_max_pixel_value(
    transect_polygon_in_rgb: pandas.DataFrame,
) -> pandas.DataFrame:
    """
    Calculate and add the beta coefficient and maximum pixel value to each row in a DataFrame of transect polygons.

    This function iterates over each row in the provided DataFrame, which contains transect polygons. For each
    transect, it calculates the maximum pixel value from the RGB values and the beta coefficient,
    which is derived from the mean logarithm of pixel values, the maximum pixel value, and the average depth (DA).
    The calculated values are then added as new columns ("max_pixel_value" and "beta") in the DataFrame.

    Parameters:
    -----------
    transect_polygon_in_rgb : pandas.DataFrame
        A DataFrame containing transect polygons with the following columns:
        - "DA": The average depth of the transect.
        - "masked_subset_transect_rgb": A 2D numpy array of RGB values for the transect.

    Returns:
    --------
    transect_polygon_in_rgb : pandas.DataFrame
        The input DataFrame with additional columns:
        - "max_pixel_value": The maximum pixel value in the interpolated RGB array.
        - "beta": The calculated beta coefficient based on the pixel values and depth.
    """
    for i, row in transect_polygon_in_rgb.iterrows():
        da = row["DA"]
        masked_subset_transect_rgb = row["masked_subset_transect_rgb"]
        max_value = masked_subset_transect_rgb[:, 0].max()
        mean_log_pixel_value = numpy.log(masked_subset_transect_rgb[:, 0]).mean()
        beta = calculate_beta(mean_log_pixel_value, max_value, da)
        transect_polygon_in_rgb.loc[i, "max_pixel_value"] = max_value
        transect_polygon_in_rgb.loc[i, "beta"] = beta
    return transect_polygon_in_rgb


def smooth_beta_and_max_value_with_touching_transect(
    transect_polygon_in_rgb: pandas.DataFrame, window: int
) -> pandas.DataFrame:
    """
    Smooth the beta coefficient and maximum pixel value across touching transects.

    This function iterates over each transect in the provided DataFrame and calculates a smoothed
    value for the beta coefficient and maximum pixel value by averaging these values across a
    window of neighboring transects that share boundaries ("touching" transects). The window size
    must be an odd number to ensure symmetry around each transect.

    Parameters:
    -----------
    transect_polygon_in_rgb : pandas.DataFrame
        A DataFrame containing transect polygons with the following columns:
        - "geometry": A shapely Polygon object representing the transect polygon.
        - "beta": The beta coefficient for the transect.
        - "max_pixel_value": The maximum pixel value in the transect.

    window : int
        The size of the smoothing window. Must be an odd number.

    Returns:
    --------
    transect_polygon_in_rgb : pandas.DataFrame
        The input DataFrame with additional columns:
        - "beta_smoothed": The smoothed beta coefficient.
        - "beta_smoothed_len": The number of non-NaN values used in the smoothing.
        - "max_pixel_value_smoothed": The smoothed maximum pixel value.

    Raises:
    -------
    ValueError
        If the `window` parameter is not an odd number.

    Notes:
    ------
    - The function searches for neighboring transects before and after the current transect,
      looking for polygons that touch the current polygon. It attempts to include as many
      touching neighbors as defined by half the window size.
    - If the number of touching neighbors found is less than half the window size, only the
      available neighbors are used for smoothing.
    """
    if window % 2 == 0:
        raise ValueError("Window need to be odd")
    window_one_size = window // 2
    for i, row in transect_polygon_in_rgb.iterrows():
        beta_list = []
        max_pixel_value_list = []

        # Transects before the current one
        current_geometry = row["geometry"]
        nb_obs_before = 0
        i_before = i
        nb_of_try = 0
        while nb_obs_before < window_one_size and i_before > 0 and nb_of_try < 100:
            i_before -= 1
            nb_of_try += 1
            row_before = transect_polygon_in_rgb.loc[i_before]
            geometry_before = row_before["geometry"]
            if geometry_before.touches(current_geometry):
                beta_list.append(row_before["beta"])
                max_pixel_value_list.append(row_before["max_pixel_value"])
                nb_obs_before += 1
                current_geometry = geometry_before

        # Transects after the current one
        current_geometry = row["geometry"]
        nb_obs_after = 0
        i_after = i
        nb_of_try = 0
        while (
            nb_obs_after < window_one_size
            and i_after < len(transect_polygon_in_rgb) - 1
            and nb_of_try < 100
        ):
            i_after += 1
            nb_of_try += 1
            row_after = transect_polygon_in_rgb.loc[i_after]
            geometry_after = row_after["geometry"]
            if geometry_after.touches(current_geometry):
                beta_list.append(row_after["beta"])
                max_pixel_value_list.append(row_after["max_pixel_value"])
                nb_obs_after += 1
                current_geometry = geometry_after

        # Include the current transect's values
        beta_list.append(row["beta"])
        max_pixel_value_list.append(row["max_pixel_value"])

        # Calculate and add smoothed values
        transect_polygon_in_rgb.loc[i, "beta_smoothed"] = numpy.nanmean(beta_list)
        transect_polygon_in_rgb.loc[i, "beta_smoothed_len"] = (
            ~numpy.isnan(beta_list)
        ).sum()
        transect_polygon_in_rgb.loc[i, "max_pixel_value_smoothed"] = numpy.nanmean(
            max_pixel_value_list
        )

    return transect_polygon_in_rgb


def rasterize_geometry(transect_polygon, col_to_rasterize, raster_resolution):
    """
    Rasterize a geometry column from a GeoDataFrame based on a specified resolution.

    This function takes a GeoDataFrame containing polygon geometries and a column with associated values,
    and rasterizes the geometries into a grid with the specified resolution. The resulting raster
    represents the spatial distribution of the values in the specified column across the grid.

    Parameters:
    -----------
    transect_polygon : geopandas.GeoDataFrame
        A GeoDataFrame containing polygon geometries in a column named "geometry", along with other
        columns containing data values.
    col_to_rasterize : str
        The name of the column in `transect_polygon` containing the values to rasterize.
    raster_resolution : float
        The resolution of the output raster, defining the size of each pixel.

    Returns:
    --------
    raster : numpy.ndarray
        A 2D array representing the rasterized values from the specified column.
    transform : affine.Affine
        An affine transformation matrix that maps pixel coordinates to geographic coordinates.

    Notes:
    ------
    - The rasterization is performed using the `rasterize` function from the `rasterio` library.
    - The `all_touched=True` option ensures that all pixels touched by the geometry are included in
      the rasterization.
    - The output raster is initialized with NaN values, which are replaced by the rasterized values
      from the specified column.
    """
    xmin, ymin, xmax, ymax = transect_polygon.total_bounds
    width = int((xmax - xmin) / raster_resolution)
    height = int((ymax - ymin) / raster_resolution)
    transform = from_origin(xmin, ymax, raster_resolution, raster_resolution)
    geometry_value = [
        (geom, value)
        for geom, value in zip(
            transect_polygon.geometry, transect_polygon[col_to_rasterize]
        )
    ]
    raster = rasterize(
        geometry_value,
        out_shape=(height, width),
        transform=transform,
        fill=numpy.nan,
        all_touched=True,
        dtype="float32",
    )
    return raster, transform


def reproject_raster(raster, in_transform, crs, out_shape, out_transform):
    """
    Reproject a raster to a shape and transform.

    This function reprojects a raster from its current coordinate reference system (CRS) and transform
    to a specified output shape and transform, using the nearest-neighbor resampling method.

    Parameters:
    -----------
    raster : numpy.ndarray
        A 2D array representing the input raster data to be reprojected.
    in_transform : affine.Affine
        The affine transformation matrix for the input raster.
    crs : dict or str
        The coordinate reference system (CRS) of the input raster. This can be specified as a dictionary
        in PROJ format or as an EPSG code string (e.g., "EPSG:4326").
    out_shape : tuple of int
        The shape of the output raster as a tuple (height, width).
    out_transform : affine.Affine
        The affine transformation matrix for the output raster.

    Returns:
    --------
    reprojected_raster : numpy.ndarray
        A 2D array representing the reprojected raster data.

    Notes:
    ------
    - The reprojected raster is created using the `reproject` function from the `rasterio` library.
    - The resampling method used is `Resampling.nearest`, which assigns the value of the nearest input
      pixel to the output pixel during reprojection.
    """
    reprojected_raster = numpy.empty(out_shape, dtype=rasterio.float32)
    reproject(
        source=raster,
        destination=reprojected_raster,
        src_transform=in_transform,
        src_crs=crs,
        dst_transform=out_transform,
        dst_crs=crs,
        resampling=Resampling.nearest,
    )
    return reprojected_raster


def save_array_as_raster(array, profile, crs, transform, path):
    """
    Save a 3D numpy array as a raster file.

    This function writes a numpy array to a raster file using the specified profile and transform.
    The profile is updated to match the dimensions and data type of the array.

    Parameters:
    -----------
    array : numpy.ndarray
        The numpy array to be saved as a raster.
    profile : dict
        A dictionary containing the metadata for the raster, such as driver.
    crs : dict or str
        The coordinate reference system for the output raster.
    transform : affine.Affine
        The affine transformation matrix defining the georeferencing of the raster.
    path : str
        The file path where the raster will be saved.

    Returns:
    --------
    None

    Notes:
    ------
    - The profile is automatically updated to match the array's height, width, count (number of bands),
      and data type.
    - The raster is saved in "float64" format.
    """
    profile.update(
        {
            "height": array.shape[1],
            "width": array.shape[2],
            "count": array.shape[0],
            "transform": transform,
            "crs": crs,
            "dtype": "float64",
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)


def get_sobel_mask_and_masked_rgb_array(
    water_rgb_array, sobel_threshold, gaussian_sigma, gaussian_threshold
):
    """
    Generate a Sobel mask from an RGB array and apply optional Gaussian smoothing.

    This function calculates the Sobel gradient of a given water RGB array to detect edges, then applies a
    threshold to create a binary mask identifying pixels with a strong gradient. Optionally, Gaussian
    filtering can be applied to the Sobel mask to smooth it.

    Parameters:
    -----------
    water_rgb_array : numpy.ndarray
        A 3D numpy array representing the RGB values of the water region.
    sobel_threshold : float
        The threshold for the Sobel mask to detect edges, where pixels with a gradient above this value
        are marked as True (bad pixels).
    gaussian_sigma : float
        The standard deviation for the Gaussian kernel used to smooth the Sobel mask. If set to 0, no
        Gaussian filtering is applied.
    gaussian_threshold : float
        The threshold applied to the Gaussian-smoothed mask, where pixels with values above this threshold
        are marked as True.

    Returns:
    --------
    sobel_mask : numpy.ndarray
        A 2D binary numpy array representing the Sobel mask, where True values indicate pixels with strong
        gradients (i.e., bad pixels).
    """
    sobel_result = sobel(water_rgb_array).mean(axis=2)
    sobel_mask = sobel_result > sobel_threshold
    if gaussian_sigma > 0:
        sobel_mask = (
            gaussian_filter(sobel_mask.astype(float), gaussian_sigma)
            > gaussian_threshold
        )
    return sobel_mask


def interpolate_bad_pixels(
    water_rgb_array_good_pixel_filtered,
    is_water_bad_pixel_mask,
    water_mask,
    kernel_sigma,
    kernel_size,
):
    """
    Interpolate bad pixels in an RGB array using Gaussian convolution.

    This function iteratively interpolates the bad pixels (identified by a mask) within a filtered water RGB array.
    Gaussian convolution is applied to each color channel independently, using a specified kernel size and standard deviation.
    The interpolation continues until all bad pixels within the water mask are filled.

    Parameters:
    -----------
    water_rgb_array_good_pixel_filtered : numpy.ndarray
        A 3D numpy array representing the filtered RGB values of the water region, with good pixels retained and
        bad pixels set to NaN.
    is_water_bad_pixel_mask : numpy.ndarray
        A 2D binary mask array where True values indicate bad pixels in the water region that need interpolation.
    water_mask : numpy.ndarray
        A 2D binary mask array where True values indicate the water region.
    kernel_sigma : float
        The standard deviation for the Gaussian kernel used in convolution.
    kernel_size : int
        The size of the Gaussian kernel used in convolution.

    Returns:
    --------
    water_rgb_array_pixel_interpolated : numpy.ndarray
        The RGB array with bad pixels interpolated using Gaussian convolution, where bad pixels are replaced
        with interpolated values.
    """
    water_rgb_array_pixel_interpolated = water_rgb_array_good_pixel_filtered.copy()
    for chanel_name, chanel in zip(["Red", "Green", "Blue"], range(3)):
        gaussian_kernel = Gaussian2DKernel(
            x_stddev=kernel_sigma,
            y_stddev=kernel_sigma,
            x_size=kernel_size,
            y_size=kernel_size,
        )
        nan_pixel_to_interpolate_sum = is_water_bad_pixel_mask.sum()
        with tqdm(
            total=nan_pixel_to_interpolate_sum, desc=f"Interpolated {chanel_name} band"
        ) as pbar:
            while nan_pixel_to_interpolate_sum != 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyUserWarning)
                    chanel_interpolated = convolve(
                        water_rgb_array_pixel_interpolated[:, :, chanel],
                        gaussian_kernel,
                    )
                water_rgb_array_pixel_interpolated[
                    is_water_bad_pixel_mask, chanel
                ] = chanel_interpolated[is_water_bad_pixel_mask]
                nan_pixel_max_to_interpolate = (
                    numpy.isnan(water_rgb_array_pixel_interpolated[:, :, chanel])
                    & water_mask
                )
                new_nan_pixel_max_to_interpolate_sum = (
                    nan_pixel_max_to_interpolate.sum()
                )
                pbar.update(
                    nan_pixel_to_interpolate_sum - new_nan_pixel_max_to_interpolate_sum
                )
                nan_pixel_to_interpolate_sum = new_nan_pixel_max_to_interpolate_sum
    return water_rgb_array_pixel_interpolated


def detect_and_interpolate_bad_water_pixels(
    water_rgb_array,
    water_mask,
    sobel_threshold=0.04,
    sobel_gaussian_std=1,
    sobel_gaussian_threshold=0.3,
    interpolated_gaussian_std=1,
    interpolated_gaussian_size=11,
):
    """
    Process the water RGB array by detecting bad pixels using a Sobel mask, filtering good pixels, and
    interpolating bad pixels using Gaussian convolution.

    The Sobel filter identifies pixels with a large vertical or horizontal gradient.
    These pixels are typically associated with reflections or small ripples on the water surface.

    This function applies a Sobel edge detection to create a mask for bad pixels in the water RGB array.
    Good pixels are retained, and bad pixels are interpolated using Gaussian convolution. The resulting
    array is returned with bad pixels filled in.

    Parameters:
    -----------
    water_rgb_array : numpy.ndarray
        A 3D numpy array representing the RGB values of the water region.
    water_mask : numpy.ndarray
        A 2D numpy array masking the water region.
    sobel_threshold : float, optional
        The threshold for the Sobel mask to detect edges, which will be treated as bad pixels. Default is 0.04.
    sobel_gaussian_std : float, optional
        The standard deviation for the Gaussian kernel used in the Sobel mask. Default is 1.
    sobel_gaussian_threshold : float, optional
        The threshold for the Gaussian-filtered Sobel mask to classify bad pixels. Default is 0.3.
    interpolated_gaussian_std : float, optional
        The standard deviation for the Gaussian kernel used in interpolating bad pixels. Default is 1.
    interpolated_gaussian_size : int, optional
        The size of the Gaussian kernel used in interpolating bad pixels. Default is 11.

    Returns:
    --------
    water_rgb_array : numpy.ndarray
        The processed water RGB array with bad pixels interpolated.
    """
    sobel_mask = get_sobel_mask_and_masked_rgb_array(
        water_rgb_array, sobel_threshold, sobel_gaussian_std, sobel_gaussian_threshold
    )
    bad_pixel_mask = sobel_mask

    is_water_good_pixel_mask = water_mask & ~bad_pixel_mask
    is_water_bad_pixel_mask = water_mask & bad_pixel_mask

    water_rgb_array_good_pixel_filtered = water_rgb_array.copy().astype(float)
    water_rgb_array_good_pixel_filtered[~is_water_good_pixel_mask] = None

    water_rgb_array_interpolated = interpolate_bad_pixels(
        water_rgb_array_good_pixel_filtered,
        is_water_bad_pixel_mask,
        water_mask,
        interpolated_gaussian_std,
        interpolated_gaussian_size,
    )
    return water_rgb_array_interpolated, is_water_bad_pixel_mask


def smooth_transect_edges(
    transect_polygon_in_rgb,
    depth_estimate,
    raster_resolution,
    height,
    width,
    affine_transform,
    crs,
    sobel_threshold=0.1,
    gaussian_std=1,
    gaussian_size=11,
):
    """
    Smooth the edges of each transect to improve continuity in depth estimates.

    This function rasterizes the transect polygons, applies a Sobel filter to detect edges,
    reprojects the Sobel edge mask, and then applies Gaussian convolution to smooth the depth
    estimates along the detected edges.

    Parameters:
    -----------
    transect_polygon_in_rgb : GeoDataFrame
        A GeoDataFrame containing transect polygons and their associated data.
    depth_estimate : numpy.ndarray
        The array containing depth estimates to be smoothed.
    raster_resolution : float
        The resolution of the rasterization.
    height : int
        The height of the output raster.
    width : int
        The width of the output raster.
    affine_transform : Affine
        The affine transformation for the raster.
    crs : dict or str
        The coordinate reference system for the output raster.
    sobel_threshold : float, optional
        The threshold value for the Sobel filter to detect edges. Default is 0.1.
    gaussian_std : float, optional
        The standard deviation for the Gaussian kernel used in convolution. Default is 1.
    gaussian_size : int, optional
        The size of the Gaussian kernel used in convolution. Default is 11.
    Returns:
    --------
    depth_estimate_smoothed : numpy.ndarray
        The smoothed depth estimate array with edges enhanced.
    """
    # Rasterize the transect polygons using the provided resolution
    raster_pk, transform_pk = rasterize_geometry(
        transect_polygon_in_rgb, "PK", raster_resolution=raster_resolution
    )

    # Apply Sobel filter to detect edges in the rasterized transects
    sobel_transect = (sobel(raster_pk) > sobel_threshold).astype(float)

    # Reproject the Sobel edge mask to match the desired coordinate reference system and dimensions
    sobel_transect_reproject = reproject_raster(
        numpy.atleast_3d(sobel_transect).transpose(2, 0, 1),
        in_transform=transform_pk,
        crs=crs,
        out_shape=(1, height, width),
        out_transform=affine_transform,
    )[0]

    # Define the Gaussian kernel for smoothing
    gaussian_kernel = Gaussian2DKernel(
        x_stddev=gaussian_std,
        y_stddev=gaussian_std,
        x_size=gaussian_size,
        y_size=gaussian_size,
    )

    # Apply Gaussian convolution to smooth the depth estimates along the Sobel-detected edges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        depth_estimate_gaussian = convolve(depth_estimate, gaussian_kernel)

    # Copy the depth estimate array and replace the values along the edges with the smoothed values
    depth_estimate_smoothed = depth_estimate.copy()
    depth_estimate_smoothed[sobel_transect_reproject == 1] = depth_estimate_gaussian[
        sobel_transect_reproject == 1
    ]

    return depth_estimate_smoothed


def get_basic_out_meta_data_tiff():
    """
    Generate basic metadata for a GeoTIFF file.

    This function returns a dictionary containing common metadata settings for creating a GeoTIFF file.
    The metadata specifies the file format, block size, tiling, compression method, and interleave mode.

    Returns:
    --------
    out_meta : dict
        A dictionary containing metadata for creating a GeoTIFF file with the following keys:
        - "driver": The file format driver, set to "GTiff" for GeoTIFF.
        - "blockysize": The size of the blocks along the y-axis, set to 128.
        - "blockxsize": The size of the blocks along the x-axis, set to 128.
        - "tiled": A boolean indicating whether the output image should be tiled, set to True.
        - "compress": The compression method used for the output file, set to "lzw".
        - "interleave": The interleave mode for the pixel data, set to "pixel".
    """
    out_meta = {
        "driver": "GTiff",
        "blockysize": 128,
        "blockxsize": 128,
        "tiled": True,
        "compress": "lzw",
        "interleave": "pixel",
    }
    return out_meta
