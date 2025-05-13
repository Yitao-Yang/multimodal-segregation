import pandas as pd
import numpy as np
import logging
import math
from collections import defaultdict
import geopandas
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import ast
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants from the text ---
TIME_FRAME_HOURS_PSM = 1  # 1-hour time frames
NUM_INCOME_GROUPS = 4
LOG_NUM_INCOME_GROUPS = np.log(NUM_INCOME_GROUPS)


# === Helper Functions ===

def get_time_slot_id(timestamp, time_frame_hours, study_start_time=None):
    """
    Assigns a time slot ID to a given timestamp.

    Input:
        timestamp: A pandas Timestamp object.
        time_frame_hours: Duration of each time slot in hours.
        study_start_time: Optional pandas Timestamp for the absolute start of the study period.
                          If provided, slot IDs are sequential from this point.
                          If None, and time_frame_hours is 1, uses hour of the day.
    Output:
        An integer representing the time slot ID.
    """
    if pd.isna(timestamp):
        return None

    if time_frame_hours == 1 and study_start_time is None:
        return timestamp.hour  # Simple hour of day for 1-hour slots
    elif study_start_time is not None:
        if timestamp < study_start_time: return None  # Timestamp before study start
        time_delta_seconds = (timestamp - study_start_time).total_seconds()
        slot_id = math.floor(time_delta_seconds / (time_frame_hours * 3600.0))
        return slot_id
    else:  # Fallback for other time_frame_hours without a study_start_time (less ideal)
        # This would require a clear convention, e.g., slots from midnight
        day_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        time_delta_seconds = (timestamp - day_start).total_seconds()
        slot_id = math.floor(time_delta_seconds / (time_frame_hours * 3600.0))
        return slot_id


def create_spatial_grids_from_shp(urban_shp_filepath, cell_size_km, crs_projected="EPSG:4793"):
    """
    Creates spatial grids covering the urban built-up area defined by a shapefile.

    Methodology:
    1. Reads the urban built-up area shapefile.
    2. Reprojects to a suitable projected CRS for accurate distance-based gridding.
    3. Gets the total bounds of the urban area.
    4. Generates a regular grid of `cell_size_km` x `cell_size_km` covering these bounds.
    5. Assigns a unique ID to each grid cell (e.g., (row_idx, col_idx)).
    6. Converts grid cells back to WGS84 (EPSG:4326) for general use.

    Input:
        urban_shp_filepath: Path to the shapefile defining the urban built-up area.
        cell_size_km: Desired grid cell size in kilometers.
        crs_projected: A suitable projected CRS for the region
    Output:
        grid_gdf: GeoDataFrame of grid cells in WGS84 (EPSG:4326).
                  Columns: 'grid_id' (tuple), 'geometry' (Polygon).
        study_area_bounds_wgs84: Tuple (min_lon, min_lat, max_lon, max_lat) of the gridded area in WGS84.
                                 Or None if gridding fails.
    """
    logging.info(f"Creating spatial grids ({cell_size_km}km) from {urban_shp_filepath}...")
    try:
        urban_gdf = geopandas.read_file(urban_shp_filepath)
    except Exception as e:
        logging.error(f"Failed to read urban shapefile: {e}")
        return pd.DataFrame(columns=['grid_id', 'geometry']), None

    if urban_gdf.empty:
        logging.error("Urban shapefile is empty.")
        return pd.DataFrame(columns=['grid_id', 'geometry']), None

    # Ensure CRS is set, then project
    if urban_gdf.crs is None:
        logging.warning(f"Urban shapefile has no CRS defined. Assuming WGS84 (EPSG:4326).")
        urban_gdf.set_crs("EPSG:4326", inplace=True)

    try:
        urban_gdf_projected = urban_gdf.to_crs(crs_projected)
    except Exception as e:
        logging.error(f"Failed to reproject urban shapefile to {crs_projected}: {e}")
        return pd.DataFrame(columns=['grid_id', 'geometry']), None

    # Get the total bounds in the projected CRS
    minx, miny, maxx, maxy = urban_gdf_projected.total_bounds
    cell_size_m = cell_size_km * 1000.0

    # Create grid cells
    grid_cells_projected = []
    col_idx = 0
    cur_minx = minx
    while cur_minx < maxx:
        row_idx = 0
        cur_miny = miny
        while cur_miny < maxy:
            grid_poly_projected = box(cur_minx, cur_miny, cur_minx + cell_size_m, cur_miny + cell_size_m)
            grid_cells_projected.append({'grid_id': (row_idx, col_idx), 'geometry': grid_poly_projected})
            cur_miny += cell_size_m
            row_idx += 1
        cur_minx += cell_size_m
        col_idx += 1

    if not grid_cells_projected:
        logging.error("No grid cells generated. Check bounds and cell size.")
        return pd.DataFrame(columns=['grid_id', 'geometry']), None

    grid_gdf_projected = geopandas.GeoDataFrame(grid_cells_projected, crs=crs_projected)

    # Intersect with urban built-up area to keep only relevant grids
    dissolved_urban_area = urban_gdf_projected.unary_union # Dissolve urban polygons
    relevant_grids = grid_gdf_projected[grid_gdf_projected.intersects(dissolved_urban_area)]
    grid_gdf_projected = relevant_grids

    # Convert back to WGS84 for general use
    try:
        grid_gdf_wgs84 = grid_gdf_projected.to_crs("EPSG:4326")
    except Exception as e:
        logging.error(f"Failed to reproject grids to WGS84: {e}")
        return pd.DataFrame(columns=['grid_id', 'geometry']), None

    # Get overall bounds of the gridded area in WGS84
    final_min_lon, final_min_lat, final_max_lon, final_max_lat = grid_gdf_wgs84.total_bounds
    study_area_bounds_wgs84 = (final_min_lon, final_min_lat, final_max_lon, final_max_lat)

    logging.info(f"Successfully created {len(grid_gdf_wgs84)} grid cells ({cell_size_km}km).")
    return grid_gdf_wgs84, study_area_bounds_wgs84


# === § Probabilistic segregation measure (PSM) ===

def determine_route_occupancy_for_psm(
        user_id,
        mode,  # 'active', 'private', 'public_bus', 'public_railway'
        route_segments,  # List of dicts from generated_routes_df
        departure_time,  # pandas Timestamp for this trip
        spatial_unit_type,  # 'grid' or 'transit'
        grid_gdf,  # GeoDataFrame of grid cells (if spatial_unit_type is 'grid')
        time_frame_hours,
        study_start_time_abs  # Absolute start timestamp of the study period for consistent time slot IDs
):
    """
    Determines which spatial units and time slots a given mode-specific route occupies.
    This is a core function for calculating I_{i,s,t} for PSM.

    Input:
        ... (user_id, mode, route_segments, departure_time as before) ...
        spatial_unit_type: 'grid' or 'transit'.
        grid_gdf: GeoDataFrame of grid cells (output of `create_spatial_grids_from_shp`).
                  Required if `spatial_unit_type` is 'grid'. Contains 'grid_id' and 'geometry'.
        time_frame_hours: Duration of time slots.
        study_start_time_abs: Absolute start Timestamp for the entire study period, for consistent time slot IDs.

    Output:
        A set of tuples (spatial_unit_id, time_slot_id) that this route occupies for the given mode.
    """
    occupied_units_times = set()
    current_time = departure_time

    if not route_segments or pd.isna(departure_time):
        return occupied_units_times

    for seg_idx, segment in enumerate(route_segments):
        segment_start_time = current_time
        segment_duration_seconds = segment.get('duration_s', 0)
        if pd.isna(segment_duration_seconds) or segment_duration_seconds <= 0:
            segment_duration_seconds = 60  # 1 minute default if invalid/zero

        segment_end_time = segment_start_time + pd.Timedelta(seconds=segment_duration_seconds)

        # Determine points to check for occupancy within this segment's duration
        # Iterate at a certain temporal resolution
        # For each point in time, determine its location and map to spatial unit.
        times_to_check_in_segment = [segment_start_time]
        if segment_duration_seconds > 60:  # If segment is longer than 1 min, add a mid-time check
            times_to_check_in_segment.append(segment_start_time + pd.Timedelta(seconds=segment_duration_seconds / 2))
        times_to_check_in_segment.append(segment_end_time - pd.Timedelta(seconds=1))  # Just before end

        for eval_time in times_to_check_in_segment:
            if eval_time >= segment_end_time and eval_time != segment_start_time: continue  # Ensure within segment
            time_slot_id = get_time_slot_id(eval_time, time_frame_hours, study_start_time_abs)
            if time_slot_id is None: continue

            if spatial_unit_type == 'grid':
                if grid_gdf is None or grid_gdf.empty: continue
                # Determine location at eval_time.
                # This requires interpolating location along the segment's 'points' list.
                # 'points' in segment is a list of [lat,lon] for the segment.
                segment_points_wgs84 = segment.get('points', [])  # Assuming WGS84 [[lat,lon],...]
                if not segment_points_wgs84: continue

                # Use representative points of the segment to check against grids. If any of these fall into a grid, the route is considered in that grid.
                points_to_check_geom = []
                points_to_check_geom.append(
                    ShapelyPoint(segment_points_wgs84[0][1], segment_points_wgs84[0][0]))  # lon, lat
                if len(segment_points_wgs84) > 1:
                    points_to_check_geom.append(ShapelyPoint(segment_points_wgs84[-1][1], segment_points_wgs84[-1][0]))
                if len(segment_points_wgs84) > 2:
                    mid_idx = len(segment_points_wgs84) // 2
                    points_to_check_geom.append(
                        ShapelyPoint(segment_points_wgs84[mid_idx][1], segment_points_wgs84[mid_idx][0]))

                for point_geom in points_to_check_geom:
                    # Find which grid this point falls into
                    # This uses .contains which is accurate. grid_gdf should be in WGS84 here.
                    try:
                        containing_grids = grid_gdf[grid_gdf.geometry.contains(point_geom)]
                        for _, grid_row in containing_grids.iterrows():
                            occupied_units_times.add((grid_row['grid_id'], time_slot_id))
                    except Exception as e_gcontain:
                        logging.debug(f"Error checking grid containment for point {point_geom}: {e_gcontain}")

            elif spatial_unit_type == 'transit':
                transit_segment_id = segment.get(
                    'segment_id')  # e.g., (from_station, to_station, line, unique_idx_if_needed)
                if transit_segment_id:  # Ensure it's hashable, e.g. a tuple
                    if isinstance(transit_segment_id, list): transit_segment_id = tuple(transit_segment_id)
                    occupied_units_times.add((transit_segment_id, time_slot_id))

        current_time = segment_end_time

    return occupied_units_times

def calculate_expected_populations_for_psm(
        trips_df,  # user_id, income_group, prob_mode_m, departure_time
        generated_routes_df,  # trip_id, mode, route_segments (parsed with points/segment_id, duration_s)
        target_mode,
        spatial_unit_type,  # 'grid' or 'transit'
        grid_gdf,  # GeoDataFrame of grids if type is 'grid'
        time_frame_hours,
        study_start_time_abs,  # Absolute start Timestamp of the study period
        all_income_groups  # List like ['q1', 'q2', 'q3', 'q4']
):
    """
    Calculates E_{s,t,q} for a specific travel mode.

    Input:
        ... (trips_df, generated_routes_df, target_mode as before) ...
        spatial_unit_type: 'grid' or 'transit'.
        grid_gdf: GeoDataFrame of grids if spatial_unit_type is 'grid'.
        time_frame_hours: Duration of time slots.
        study_start_time_abs: Absolute start Timestamp of the study for consistent time slot IDs.
        all_income_groups: List of unique income group identifiers.

    Output:
        expected_populations: Dict {(spatial_unit_id, time_slot_id): {income_group: count, ...}}
    """
    logging.info(f"Calculating expected populations for mode: {target_mode} (Type: {spatial_unit_type})...")
    expected_populations = defaultdict(lambda: defaultdict(float))

    prob_col_name = f'prob_{target_mode}'
    required_trip_cols = ['trip_id', 'user_id', 'income_group', prob_col_name, 'departure_time']
    if not all(col in trips_df.columns for col in required_trip_cols):
        logging.error(f"Trips DataFrame missing one or more required columns for E_stq: {required_trip_cols}")
        return expected_populations

    mode_routes = generated_routes_df[generated_routes_df['mode'] == target_mode]
    if mode_routes.empty:
        logging.warning(f"No routes found for target mode: {target_mode} in generated_routes_df.")
        return expected_populations

    trips_with_routes = pd.merge(
        trips_df[required_trip_cols],
        mode_routes[['trip_id', 'route_segments']],
        on='trip_id', how='inner'
    )
    if trips_with_routes.empty:
        logging.warning(f"No trips merged with routes for mode {target_mode}.")
        return expected_populations

    trips_with_routes['departure_time'] = pd.to_datetime(trips_with_routes['departure_time'])

    processed_count = 0
    total_to_process = len(trips_with_routes)
    for _, row in trips_with_routes.iterrows():
        user_id = row['user_id']
        income_group = row['income_group']
        prob_mode_choice = row[prob_col_name]
        route_segments = row['route_segments']  # List of dicts
        departure_time = row['departure_time']

        if pd.isna(prob_mode_choice) or prob_mode_choice < 1e-6 or not route_segments or pd.isna(departure_time):
            continue

        occupied_s_t = determine_route_occupancy_for_psm(
            user_id, target_mode, route_segments, departure_time,
            spatial_unit_type, grid_gdf, time_frame_hours, study_start_time_abs
        )

        for s, t in occupied_s_t:
            expected_populations[(s, t)][income_group] += prob_mode_choice

        processed_count += 1
        if processed_count % (total_to_process // 10 if total_to_process > 100 else 100) == 0:  # Log roughly 10 times
            logging.info(f"Occupancy processed for {processed_count}/{total_to_process} trips ({target_mode})...")

    logging.info(f"Finished E_stq for {target_mode}. Found {len(expected_populations)} occupied (unit,slot) pairs.")
    return expected_populations

def calculate_psm_from_expected_populations(expected_populations_st, all_income_groups):
    """
    Calculates PSM (Eq.2 from paper) from pre-calculated E_{s,t,q}.
    Input:
        expected_populations_st: Dict {(s,t): {income_group: count, ...}}
        all_income_groups: List of all income group labels (e.g., ['q1', 'q2', 'q3', 'q4'])
    Output:
        psm_results_df: DataFrame with 'spatial_unit_id', 'time_slot_id', 'psm_value',
                        'total_expected_pop', and 'pop_qX' for each group.
    """
    logging.info("Calculating PSM values from E_stq...")
    psm_data = []

    for (s, t), group_counts_dict in expected_populations_st.items():
        total_expected_pop_st = sum(group_counts_dict.values())
        psm_value = np.nan

        if total_expected_pop_st > 1e-9:  # Avoid division by zero, ensure some presence
            entropy = 0
            for income_group_q in all_income_groups:  # Iterate all possible groups
                count_q = group_counts_dict.get(income_group_q, 0.0)  # Get count, default to 0 if group not present
                if count_q > 1e-9:  # log(0) is undefined, ensure count is meaningfully positive
                    proportion_q = count_q / total_expected_pop_st
                    entropy -= proportion_q * np.log(proportion_q)

            if LOG_NUM_INCOME_GROUPS > 1e-9:  # Avoid division by zero if NUM_INCOME_GROUPS is 1
                psm_value = entropy / LOG_NUM_INCOME_GROUPS
            elif NUM_INCOME_GROUPS == 1 and total_expected_pop_st > 1e-9:
                psm_value = 0.0  # Perfect segregation if only one group possible and present

        row_data = {'spatial_unit_id': s, 'time_slot_id': t, 'psm_value': psm_value,
                    'total_expected_pop': total_expected_pop_st}
        for q_label in all_income_groups:
            row_data[f'pop_{q_label}'] = group_counts_dict.get(q_label, 0.0)
        psm_data.append(row_data)

    psm_results_df = pd.DataFrame(psm_data)
    logging.info(f"PSM calculation complete. {len(psm_results_df)} (unit,slot) entries.")
    return psm_results_df


# === § Multimodal uniformity index (MUI) ===

def calculate_mui_for_regions(
        psm_results_by_mode,  # Dict: {'active': psm_df_active, 'private': psm_df_private, ...}
        region_unit_mapping_df,  # Maps 'spatial_unit_id' to 'region_id'
        modes_for_mui=['active', 'private', 'public_bus', 'public_railway']  # Modes to include in MUI
):
    """
    Calculates MUI for geographical regions and time frames.

    Input:
        psm_results_by_mode: Dict mapping mode name to its PSM DataFrame.
                             Each PSM DataFrame needs 'spatial_unit_id', 'time_slot_id', 'psm_value'.
        region_unit_mapping_df: DataFrame mapping 'spatial_unit_id' to 'region_id'.
                                Crucially, 'spatial_unit_id' here must be compatible with those
                                in the PSM DataFrames (i.e., grid IDs and transit segment IDs).
        modes_for_mui: List of mode names to be included in the MUI calculation.
    Output:
        mui_results_df: DataFrame with 'region_id', 'time_slot_id', 'mui_value',
                        and 'avg_psm_<mode>' for each mode in modes_for_mui.
    """
    logging.info("Calculating Multimodal Uniformity Index (MUI)...")
    if not psm_results_by_mode or region_unit_mapping_df.empty:
        logging.error("Missing PSM results or region-unit mapping for MUI calculation.")
        return pd.DataFrame(columns=['region_id', 'time_slot_id', 'mui_value'])

    # Aggregate PSM by region and time for each mode
    regional_avg_psms_list = []
    unique_region_time_keys = set()

    for mode, psm_df in psm_results_by_mode.items():
        if mode not in modes_for_mui: continue  # Skip if mode not for MUI
        if psm_df is None or psm_df.empty or 'psm_value' not in psm_df.columns:
            logging.warning(f"PSM data for mode '{mode}' is invalid or empty. Will use 0 for its contribution.")
            # Add structure so merge doesn't fail, but values will be NaN/0
            continue

        psm_df_cleaned = psm_df.dropna(subset=['psm_value', 'spatial_unit_id', 'time_slot_id'])
        if psm_df_cleaned.empty: continue

        # 'spatial_unit_id' must be mergeable.
        # If grid_id is tuple and transit_id is tuple, ensure mapping_df handles both or convert.
        try:
            # Attempt to make spatial_unit_id string for merging if it's a tuple (common for grid_id)
            # This is a potential point of failure if IDs are complex objects.
            if psm_df_cleaned['spatial_unit_id'].apply(lambda x: isinstance(x, tuple)).any():
                psm_df_cleaned['merge_id'] = psm_df_cleaned['spatial_unit_id'].astype(str)
                temp_mapping_df = region_unit_mapping_df.copy()
                if temp_mapping_df['spatial_unit_id'].apply(lambda x: isinstance(x, tuple)).any():
                    temp_mapping_df['merge_id'] = temp_mapping_df['spatial_unit_id'].astype(str)
                else:  # If mapping df already string or other
                    temp_mapping_df['merge_id'] = temp_mapping_df['spatial_unit_id']  # Assume it can merge
            else:  # If psm_df IDs are not tuples
                psm_df_cleaned['merge_id'] = psm_df_cleaned['spatial_unit_id']
                temp_mapping_df = region_unit_mapping_df.copy()
                temp_mapping_df['merge_id'] = temp_mapping_df['spatial_unit_id']

            merged_for_avg = pd.merge(psm_df_cleaned, temp_mapping_df, on='merge_id', how='inner')
        except Exception as e_merge:
            logging.error(f"Error merging PSM for {mode} with region map (spatial_unit_id matching issue?): {e_merge}")
            continue

        if not merged_for_avg.empty:
            avg_psm = merged_for_avg.groupby(['region_id', 'time_slot_id_x'])[
                'psm_value'].mean().reset_index()  # time_slot_id_x from psm_df
            avg_psm.rename(columns={'psm_value': f'avg_psm_{mode}', 'time_slot_id_x': 'time_slot_id'}, inplace=True)
            regional_avg_psms_list.append(avg_psm)
            for _, r in avg_psm.iterrows(): unique_region_time_keys.add((r['region_id'], r['time_slot_id']))

    if not regional_avg_psms_list:
        logging.warning("No regional average PSM data available to calculate MUI.")
        return pd.DataFrame(columns=['region_id', 'time_slot_id', 'mui_value'])

    # Create a base DataFrame with all unique (region_id, time_slot_id) pairs
    if not unique_region_time_keys:
        logging.warning("No unique (region, time) keys found after processing PSM averages.")
        return pd.DataFrame(columns=['region_id', 'time_slot_id', 'mui_value'])

    combined_df = pd.DataFrame(list(unique_region_time_keys), columns=['region_id', 'time_slot_id'])

    for avg_psm_df_mode in regional_avg_psms_list:
        combined_df = pd.merge(combined_df, avg_psm_df_mode, on=['region_id', 'time_slot_id'], how='left')

    # Fill NaNs with 0 for modes that might not have PSM in a region/time
    for mode in modes_for_mui:
        if f'avg_psm_{mode}' not in combined_df.columns:
            combined_df[f'avg_psm_{mode}'] = 0.0
        else:
            combined_df[f'avg_psm_{mode}'] = combined_df[f'avg_psm_{mode}'].fillna(0.0)

    mui_results = []
    log_num_modes_for_mui_norm = np.log(len(modes_for_mui)) if len(
        modes_for_mui) > 1 else 1.0  # Avoid log(1)=0 denominator

    for _, row in combined_df.iterrows():
        psm_A_t_m_values = [row.get(f'avg_psm_{mode}', 0.0) for mode in modes_for_mui]
        sum_psm_A_t_m = sum(psm_A_t_m_values)

        mui_value = np.nan  # Default
        if sum_psm_A_t_m > 1e-9:  # If there's any segregation score to compare
            entropy_mui = 0
            num_contributing_modes = 0  # Modes with PSM > 0
            for psm_val in psm_A_t_m_values:
                if psm_val > 1e-9:
                    proportion_r_m = psm_val / sum_psm_A_t_m
                    entropy_mui -= proportion_r_m * np.log(proportion_r_m)
                    num_contributing_modes += 1

            # Normalize by log(len(modes_for_mui)) which is log(4).
            if log_num_modes_for_mui_norm > 1e-9:
                mui_value = entropy_mui / log_num_modes_for_mui_norm
            elif num_contributing_modes == 1:  # Only one mode has non-zero PSM
                mui_value = 0.0  # Perfect non-uniformity (or max diversity of experience)
            # If num_contributing_modes is 0 (all PSMs were ~0), mui_value remains NaN.

        entry = {'region_id': row['region_id'], 'time_slot_id': row['time_slot_id'], 'mui_value': mui_value}
        for mode in modes_for_mui: entry[f'avg_psm_{mode}'] = row.get(f'avg_psm_{mode}', 0.0)
        mui_results.append(entry)

    mui_results_df = pd.DataFrame(mui_results)
    logging.info(f"MUI calculation complete. {len(mui_results_df)} (region,time) entries.")
    return mui_results_df


# === § Sensitivity analysis of spatiotemporal scales ===
def run_psm_mui_sensitivity_analysis(
        trips_df_main_input, generated_routes_df_main_input, urban_shp_filepath,
        region_unit_mapping_generator_func,  # Function: (grids_gdf, transit_segments_info) -> region_unit_mapping_df
        all_income_groups_main, study_start_time_abs_main,
        temporal_scales_minutes_list_sens,
        spatial_scales_km_list_sens,
        fixed_spatial_scale_km_for_temporal_sens,  # e.g., 1.0 km from paper
        fixed_temporal_minutes_for_spatial_sens,  # e.g., 60 minutes from paper
        projected_crs_for_gridding_sens  # e.g., "EPSG:4793"
):
    """
    Orchestrates sensitivity analysis for PSM and MUI by varying temporal and spatial scales.
    This function iteratively calls the core PSM and MUI calculation functions.

    Input:
        trips_df_main_input: Main DataFrame of trips with probabilities, income group, etc.
        generated_routes_df_main_input: Main DataFrame of generated routes for all trips/modes.
        urban_shp_filepath: Path to the urban built-up area shapefile (for gridding).
        region_unit_mapping_generator_func: A function that takes a `grids_gdf` (GeoDataFrame of current grids)
                                            and optionally `transit_segments_info` (list of unique transit segment IDs)
                                            and returns a `region_unit_mapping_df` compatible with current spatial units.
                                            The `spatial_unit_id` in the mapping DF MUST be stringified if grid IDs are tuples.
        all_income_groups_main: List of income group labels.
        study_start_time_abs_main: Absolute start Timestamp for the entire study.
        temporal_scales_minutes_list_sens: List of temporal scales in minutes to test.
        spatial_scales_km_list_sens: List of spatial grid scales in km to test for active/private modes.
        fixed_spatial_scale_km_for_temporal_sens: The grid size (km) to keep fixed for temporal sensitivity.
        fixed_temporal_minutes_for_spatial_sens: The time window (minutes) to keep fixed for spatial sensitivity.
        projected_crs_for_gridding_sens: Projected CRS string for grid generation.

    Output:
        results_collection: A dictionary storing PSM and MUI DataFrames for each tested scale.
                           Structure: {'temporal': { '30min': {'psm_all_modes': {...}, 'mui_df': df}, ...},
                                       'spatial':  { '0.5km': {'psm_all_modes': {...}, 'mui_df': df}, ...}}
    """
    logging.info("Starting full PSM/MUI sensitivity analysis...")
    results_collection = {"temporal": {}, "spatial": {}}

    modes_to_analyze = ['active', 'private', 'public_bus', 'public_railway']
    transit_spatial_unit_type = 'transit'  # Transit units are segments, not grids

    # --- 1. Temporal Sensitivity Analysis ---
    # Spatial scale for active/private modes is fixed. Public transport units (segments) are also fixed.
    logging.info(f"\n=== TEMPORAL SENSITIVITY (Fixed Spatial Grid: {fixed_spatial_scale_km_for_temporal_sens}km) ===")

    # Create the fixed spatial grids for active/private modes once
    grids_gdf_fixed_spatial, _ = create_spatial_grids_from_shp(
        urban_shp_filepath,
        fixed_spatial_scale_km_for_temporal_sens,
        projected_crs_for_gridding_sens,
        filter_by_intersection=True  # Recommended for realistic unit count
    )
    if grids_gdf_fixed_spatial.empty:
        logging.error("Failed to create fixed spatial grids for temporal sensitivity. Aborting this part.")
    else:
        # Generate region_unit_mapping for this fixed grid scale.
        # Extract unique transit segment IDs from the main generated_routes_df
        unique_transit_segment_ids = set()
        if not generated_routes_df_main_input.empty:
            public_routes_sens = generated_routes_df_main_input[
                generated_routes_df_main_input['mode'].isin(['public_bus', 'public_railway'])]
            for segments_list in public_routes_sens['route_segments']:
                if isinstance(segments_list, list):
                    for seg in segments_list:
                        if isinstance(seg, dict) and 'segment_id' in seg:
                            seg_id = seg['segment_id']
                            if isinstance(seg_id, list): seg_id = tuple(seg_id)  # Ensure hashable
                            unique_transit_segment_ids.add(seg_id)

        # The generator function must be able to create mapping for both grid and transit units.
        # It needs to know which region each grid cell and each transit station (and thus segment) belongs to.
        region_map_fixed_spatial_df = region_unit_mapping_generator_func(grids_gdf_fixed_spatial,
                                                                         list(unique_transit_segment_ids))
        if region_map_fixed_spatial_df.empty:
            logging.warning("Region-unit mapping for fixed spatial scale is empty. MUI will be affected.")

        for temp_min in temporal_scales_minutes_list_sens:
            time_frame_h = temp_min / 60.0
            logging.info(f"  -- Processing Temporal Scale: {temp_min} minutes --")

            current_scale_psm_results = {}
            for mode in modes_to_analyze:
                is_grid_mode = mode in ['active', 'private']
                spatial_type = 'grid' if is_grid_mode else transit_spatial_unit_type
                current_grid_gdf_input = grids_gdf_fixed_spatial if is_grid_mode else None

                logging.info(f"    Calculating E_stq for {mode} at {temp_min}min...")
                expected_pops_temp = calculate_expected_populations_for_psm(
                    trips_df_main_input, generated_routes_df_main_input, mode,
                    spatial_type, current_grid_gdf_input,
                    time_frame_h, study_start_time_abs_main,
                    all_income_groups_main
                )
                logging.info(f"    Calculating PSM for {mode} at {temp_min}min...")
                psm_df_temp = calculate_psm_from_expected_populations(expected_pops_temp, all_income_groups_main)
                current_scale_psm_results[mode] = psm_df_temp

            logging.info(f"    Calculating MUI at {temp_min}min...")
            mui_df_temp = calculate_mui_for_regions(current_scale_psm_results, region_map_fixed_spatial_df,
                                                    modes_for_mui=modes_to_analyze)

            results_collection["temporal"][f"{temp_min}min"] = {
                "psm_all_modes": current_scale_psm_results,
                "mui_df": mui_df_temp
            }
            logging.info(f"  Completed temporal scale: {temp_min} minutes.")

    # --- 2. Spatial Sensitivity Analysis ---
    # Temporal scale is fixed. Grid spatial scale for active/private modes varies.
    # Public transport units (segments) and their PSM (at the fixed temporal scale) remain unchanged.
    logging.info(f"\n=== SPATIAL SENSITIVITY (Fixed Temporal: {fixed_temporal_minutes_for_spatial_sens}min) ===")
    time_frame_h_fixed = fixed_temporal_minutes_for_spatial_sens / 60.0

    # Pre-calculate PSM for public transport modes at the fixed temporal scale, as these are constant for spatial sensitivity.
    psm_public_fixed_temporal = {}
    logging.info(
        f"  Pre-calculating PSM for public modes at fixed temporal scale ({fixed_temporal_minutes_for_spatial_sens}min)...")
    for mode in ['public_bus', 'public_railway']:
        # For public modes, no grid_gdf is passed to calculate_expected_populations_for_psm
        logging.info(f"    Calculating E_stq for {mode} (fixed temporal)...")
        expected_pops_public_fixed_t = calculate_expected_populations_for_psm(
            trips_df_main_input, generated_routes_df_main_input, mode,
            transit_spatial_unit_type, None,  # No grid_gdf for transit type
            time_frame_h_fixed, study_start_time_abs_main,
            all_income_groups_main
        )
        logging.info(f"    Calculating PSM for {mode} (fixed temporal)...")
        psm_public_fixed_temporal[mode] = calculate_psm_from_expected_populations(expected_pops_public_fixed_t,
                                                                                  all_income_groups_main)

    for spatial_km in spatial_scales_km_list_sens:
        logging.info(f"  -- Processing Spatial Grid Scale: {spatial_km} km --")

        # Create grids for this specific spatial scale
        grids_gdf_variable_spatial, _ = create_spatial_grids_from_shp(
            urban_shp_filepath, spatial_km, projected_crs_for_gridding_sens, filter_by_intersection=True
        )
        if grids_gdf_variable_spatial.empty:
            logging.warning(f"Failed to create grids for {spatial_km}km. Skipping this spatial scale.")
            results_collection["spatial"][f"{spatial_km}km"] = {"psm_all_modes": {}, "mui_df": pd.DataFrame()}
            continue

        # Generate region_unit_mapping for THIS variable grid scale
        # This mapping must also include the fixed transit segment IDs.
        region_map_variable_spatial_df = region_unit_mapping_generator_func(grids_gdf_variable_spatial, list(
            unique_transit_segment_ids))  # Re-use unique_transit_segment_ids from temporal part
        if region_map_variable_spatial_df.empty:
            logging.warning(f"Region-unit mapping for {spatial_km}km grid scale is empty. MUI will be affected.")

        current_scale_psm_results = {}
        # Calculate PSM for active and private modes using the new grids
        for mode in ['active', 'private']:
            logging.info(f"    Calculating E_stq for {mode} at {spatial_km}km grid (fixed temporal)...")
            expected_pops_spatial = calculate_expected_populations_for_psm(
                trips_df_main_input, generated_routes_df_main_input, mode,
                'grid', grids_gdf_variable_spatial,  # Use current variable grid
                time_frame_h_fixed, study_start_time_abs_main,
                all_income_groups_main
            )
            logging.info(f"    Calculating PSM for {mode} at {spatial_km}km grid (fixed temporal)...")
            psm_df_spatial = calculate_psm_from_expected_populations(expected_pops_spatial, all_income_groups_main)
            current_scale_psm_results[mode] = psm_df_spatial

        # Add the pre-calculated public transport PSM
        current_scale_psm_results['public_bus'] = psm_public_fixed_temporal.get('public_bus', pd.DataFrame())
        current_scale_psm_results['public_railway'] = psm_public_fixed_temporal.get('public_railway', pd.DataFrame())

        logging.info(f"    Calculating MUI at {spatial_km}km grid (fixed temporal)...")
        mui_df_spatial = calculate_mui_for_regions(current_scale_psm_results, region_map_variable_spatial_df,
                                                   modes_for_mui=modes_to_analyze)

        results_collection["spatial"][f"{spatial_km}km"] = {
            "psm_all_modes": current_scale_psm_results,
            "mui_df": mui_df_spatial
        }
        logging.info(f"  Completed spatial scale: {spatial_km} km.")

    logging.info("PSM/MUI sensitivity analysis orchestration finished.")
    return results_collection


# === Main Execution ===
if __name__ == "__main__":

    #  Define base path for data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir,"data_input_files")  # Data is in a subfolder named 'data_input_files'
    path_trips_details_input = os.path.join(base_data_path, "trips_with_full_details.csv")
    path_generated_routes_input = os.path.join(base_data_path, "all_generated_routes.csv")
    path_urban_shp_input = os.path.join(base_data_path, "urban_built_up_area.shp")
    path_region_unit_map_input = os.path.join(base_data_path, "region_to_spatial_unit_map.csv")

    #  Load Pre-processed Data ---
    income_groups_list = ['q1', 'q2', 'q3', 'q4']
    trips_df_input = pd.read_csv(path_trips_details_input, parse_dates=['departure_time'])
    study_start_time_abs_main = trips_df_input[
        'departure_time'].min() if not trips_df_input.empty else pd.Timestamp('2023-06-01')
    generated_routes_input_df = pd.read_csv(path_generated_routes_input)
    generated_routes_input_df['route_segments'] = generated_routes_input_df['route_segments'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    #  Define Primary Spatial Units & Region Mapping ---
    primary_spatial_scale_km = 1.0
    primary_projected_crs = "EPSG:4793"
    primary_grids_gdf, _ = create_spatial_grids_from_shp(
        path_urban_shp_input, primary_spatial_scale_km, primary_projected_crs, filter_by_intersection=True)

    # Dummy region_unit_mapping_generator_func for the demo
    def dummy_region_map_gen(current_grids_gdf, list_of_transit_segment_ids):
        map_entries = []
        # Map grid cells (stringified tuple IDs)
        if current_grids_gdf is not None and not current_grids_gdf.empty:
            for _, grid_r in current_grids_gdf.iterrows():
                map_entries.append({'spatial_unit_id': str(grid_r['grid_id']),
                                    'region_id': f"R{grid_r['grid_id'][0] % 3}"})  # Assign to 3 dummy regions
        # Map transit segments (stringified tuple IDs)
        if list_of_transit_segment_ids:
            for ts_id in list_of_transit_segment_ids:
                map_entries.append({'spatial_unit_id': str(ts_id), 'region_id': f"R{hash(ts_id) % 3}"})
        return pd.DataFrame(map_entries).drop_duplicates()

    # Generate primary region mapping
    unique_ts_ids = set()  # Extract from generated_routes_input_df if needed
    pub_r = generated_routes_input_df[
        generated_routes_input_df['mode'].isin(['public_bus', 'public_railway'])]
    for seg_list in pub_r['route_segments']:
        if isinstance(seg_list, list):
            for seg_item in seg_list:
                if isinstance(seg_item, dict) and 'segment_id' in seg_item:
                    sid = seg_item['segment_id'];
                    unique_ts_ids.add(tuple(sid) if isinstance(sid, list) else sid)
    primary_region_map_df = dummy_region_map_gen(primary_grids_gdf, list(unique_ts_ids))

    #  Calculate PSM and MUI for Primary Scale ---
    psm_results_primary = {}
    modes_for_analysis = ['active', 'private', 'public_bus', 'public_railway']
    for mode in modes_for_analysis:
        logging.info(f"  Calculating PSM for mode: {mode} (Primary Scale)")
        is_grid_based_mode = mode in ['active', 'private']
        spatial_type = 'grid' if is_grid_based_mode else 'transit'
        current_grid_input = primary_grids_gdf if is_grid_based_mode else None
        expected_populations_primary = calculate_expected_populations_for_psm(
            trips_df_input,
            generated_routes_input_df,
            mode,
            spatial_type,
            current_grid_input,
            TIME_FRAME_HOURS_PSM,
            study_start_time_abs_main,
            income_groups_list)
        # Calculate PSM
        psm_df_primary = calculate_psm_from_expected_populations(
            expected_populations_primary,
            income_groups_list)
        psm_results_primary[mode] = psm_df_primary
    # Calculate MUI
    mui_df_primary = calculate_mui_for_regions(
        psm_results_primary,  # Dict of PSM DataFrames
        primary_region_map_df,  # Mapping for the primary grid scale
        modes_for_mui=modes_for_analysis)

    #  Run Full Sensitivity Analysis ---
    sens_temporal_scales_min = [3, 5, 10, 20, 30, 60]
    sens_spatial_scales_km = [0.25, 0.5, 1.0, 2.0]
    all_sensitivity_results = run_psm_mui_sensitivity_analysis(
        trips_df_input,
        generated_routes_input_df,
        path_urban_shp_input,
        dummy_region_map_gen,  # Pass the actual generator function defined above
        income_groups_list,
        study_start_time_abs_main,
        temporal_scales_minutes_list_sens=sens_temporal_scales_min,
        spatial_scales_km_list_sens=sens_spatial_scales_km,
        fixed_spatial_scale_km_for_temporal_sens=1.0,
        fixed_temporal_minutes_for_spatial_sens=60,
        projected_crs_for_gridding_sens=primary_projected_crs
    )