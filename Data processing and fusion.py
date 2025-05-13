import pandas as pd
import numpy as np
import logging
import math
import time
import os
import requests
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import geopandas
from shapely.geometry import Point as ShapelyPoint


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Coordinate Transformation Utilities (Essential for Amap API) ===
# These functions are used by the route generation module that interacts with Amap API, which uses GCJ02 coordinates, while GPS data is in WGS84.
X_PI = 3.14159265358979324 * 3000.0 / 180.0
PI = 3.1415926535897932384626  # π
A_AXIS = 6378245.0  # Krasovsky 1940 ellipsoid models a = 6378245.0, 1/f = 298.3
EE = 0.00669342162296594323  # EE = (2f - f^2) = e^2 eccentricity squared

def _transform_lat(lng, lat):
    """Helper function for latitude transformation in GCJ02 conversion."""
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * PI) + 40.0 *
            math.sin(lat / 3.0 * PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * PI) + 320 *
            math.sin(lat * PI / 30.0)) * 2.0 / 3.0
    return ret

def _transform_lng(lng, lat):
    """Helper function for longitude transformation in GCJ02 conversion."""
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * PI) + 20.0 *
            math.sin(2.0 * lng * PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * PI) + 40.0 *
            math.sin(lng / 3.0 * PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * PI) + 300.0 *
            math.sin(lng / 30.0 * PI)) * 2.0 / 3.0
    return ret

def wgs84_to_gcj02(wgs_lat, wgs_lon):
    """
    Converts WGS84 coordinates to GCJ02 (Mars Coordinates).
    This is necessary for APIs like Amap that expect GCJ02 input.

    Input:
        wgs_lat: Latitude in WGS84.
        wgs_lon: Longitude in WGS84.
    Output:
        A list [gcj_lon, gcj_lat] representing coordinates in GCJ02, rounded to 6 decimal places.
        Returns original [wgs_lon, wgs_lat] if coordinates are outside China.
    """

    dlat = _transform_lat(wgs_lon - 105.0, wgs_lat - 35.0)
    dlng = _transform_lng(wgs_lon - 105.0, wgs_lat - 35.0)
    radlat = wgs_lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A_AXIS * (1 - EE)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (A_AXIS / sqrtmagic * math.cos(radlat) * PI)
    gcj_lat = wgs_lat + dlat
    gcj_lon = wgs_lon + dlng
    return [np.around(gcj_lon, 6), np.around(gcj_lat, 6)]

def gcj02_to_wgs84(gcj_lon, gcj_lat):
    """
    Converts GCJ02 (Mars Coordinates) to WGS84 coordinates.
    This is used to convert Amap API polyline outputs back to standard GPS coordinates.

    Input:
        gcj_lon: Longitude in GCJ02.
        gcj_lat: Latitude in GCJ02.
    Output:
        A list [wgs_lon, wgs_lat] representing coordinates in WGS84, rounded to 6 decimal places.
        Returns original [gcj_lon, gcj_lat] if coordinates are outside China.
    """

    dlat = _transform_lat(gcj_lon - 105.0, gcj_lat - 35.0)
    dlng = _transform_lng(gcj_lon - 105.0, gcj_lat - 35.0)
    radlat = gcj_lat / 180.0 * PI
    magic = math.sin(radlat)
    magic = 1 - EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A_AXIS * (1 - EE)) / (magic * sqrtmagic) * PI)
    dlng = (dlng * 180.0) / (A_AXIS / sqrtmagic * math.cos(radlat) * PI)
    mglat = gcj_lat + dlat
    mglng = gcj_lon + dlng
    wgs_lon = gcj_lon * 2 - mglng
    wgs_lat = gcj_lat * 2 - mglat
    return [np.around(wgs_lon, 6), np.around(wgs_lat, 6)]

def string_wgs84_to_gcj02_amap_format(lat_lon_str):
    """
    Converts a 'latitude,longitude' WGS84 string to 'longitude,latitude' GCJ02 string
    formatted for Amap API requests.

    Input:
        lat_lon_str: A string "latitude,longitude" in WGS84.
    Output:
        A string "longitude,latitude" in GCJ02, or an empty string if conversion fails.
    """
    try:
        if not lat_lon_str or not isinstance(lat_lon_str, str): return ""
        parts = lat_lon_str.split(',')
        if len(parts) != 2: return ""
        lat, lon = float(parts[0]), float(parts[1])
        gcj_lon, gcj_lat = wgs84_to_gcj02(lat, lon)
        return f"{gcj_lon},{gcj_lat}"
    except (IndexError, ValueError, TypeError) as e:
        logging.error(f"Error converting WGS84 string '{lat_lon_str}' to GCJ02 for Amap: {e}")
        return ""

def string_gcj02_to_wgs84_display_format(lon_lat_str):
    """
    Converts a 'longitude,latitude' GCJ02 string (common in Amap polylines)
    to a 'latitude,longitude' WGS84 string for display or storage.

    Input:
        lon_lat_str: A string "longitude,latitude" in GCJ02.
    Output:
        A string "latitude,longitude" in WGS84, or an empty string if conversion fails.
    """
    try:
        if not lon_lat_str or not isinstance(lon_lat_str, str): return ""
        parts = lon_lat_str.split(',')
        if len(parts) != 2: return ""
        lon, lat = float(parts[0]), float(parts[1])
        wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)  # Returns [lon, lat] WGS84
        return f"{wgs_lat},{wgs_lon}"  # Format as 'lat,lon'
    except (IndexError, ValueError, TypeError) as e:
        logging.error(f"Error converting GCJ02 string '{lon_lat_str}' to WGS84 display: {e}")
        return ""

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points (specified in decimal degrees) on Earth.

    Input:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.
    Output:
        Distance in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# --- § Home and workplace identification ---

# Constants for Home/Work Identification
DBSCAN_EPS_METERS_HW = 50
DBSCAN_MIN_SAMPLES_HW = 10
POI_ASSIGNMENT_RADIUS_METERS_HW = 100
MIN_STAY_DURATION_MINUTES_HW = 15
MAX_STAY_DURATION_HOURS_HW = 24
HOME_NIGHT_START_HOUR_HW = 21
HOME_NIGHT_END_HOUR_HW = 6
HOME_MIN_VISITS_HW = 25
WORK_DAY_START_HOUR_HW = 9
WORK_DAY_END_HOUR_HW = 17
WORK_MIN_VISITS_PER_5_WORKDAYS_HW = 4  # Implies 4 visits per typical work week


def identify_significant_stays(pings_df, poi_df):
    """
    Detects significant stays from individual GPS trajectory data using DBSCAN and POI matching.

    Methodology:
    1. Groups pings by user.
    2. For each user, applies DBSCAN to cluster trajectory points based on spatial proximity.
       Uses Haversine distance for DBSCAN.
    3. Assigns each valid cluster (stay candidate) to the nearest Point of Interest (POI)
       within a defined radius if the cluster center is close enough to a POI.
    4. Filters stays based on minimum and maximum duration thresholds.
    5. Clusters smaller than min_samples or not matching a POI are discarded.

    Input:
        pings_df: DataFrame of GPS pings.
                  Required columns: 'user_id', 'latitude', 'longitude', 'timestamp'.
        poi_df: DataFrame of Points of Interest.
                Required columns: 'poi_id', 'latitude', 'longitude', 'type' (e.g., 'residential').
    Output:
        stays_df: DataFrame of identified significant stays.
                  Columns: 'user_id', 'stay_id', 'latitude' (stay center), 'longitude' (stay center),
                           'start_time', 'end_time', 'duration_minutes', 'num_pings' (in stay cluster),
                           'poi_id' (matched POI), 'poi_type'.
                  Returns an empty DataFrame if no stays are identified or inputs are invalid.
    """

    all_stays_list = []
    # Convert DBSCAN_EPS_METERS_HW to radians for Haversine distance in DBSCAN
    earth_radius_km = 6371.0
    eps_rad = (DBSCAN_EPS_METERS_HW / 1000.0) / earth_radius_km

    # Prepare KDTree for fast POI lookup if poi_df is valid and anchoring is intended
    poi_kdtree = None
    if not poi_df.empty and POI_ASSIGNMENT_RADIUS_METERS_HW > 0:
        try:
            poi_kdtree = KDTree(poi_df[['latitude', 'longitude']].values)
        except Exception as e:
            logging.error(f"Failed to create KDTree from POI data: {e}. POI anchoring will be skipped.")

    for user_id, user_pings in pings_df.groupby('user_id'):
        if len(user_pings) < DBSCAN_MIN_SAMPLES_HW:
            continue  # Not enough pings for this user to form a cluster

        # DBSCAN requires coordinates in radians for Haversine metric
        coords_rad = np.radians(user_pings[['latitude', 'longitude']].values)

        try:
            db = DBSCAN(eps=eps_rad, min_samples=DBSCAN_MIN_SAMPLES_HW, algorithm='ball_tree', metric='haversine').fit(
                coords_rad)
            labels = db.labels_
        except Exception as e_dbscan:
            logging.warning(f"DBSCAN failed for user {user_id}: {e_dbscan}. Skipping user.")
            continue

        user_pings = user_pings.copy()
        user_pings.loc[:, 'cluster_label'] = labels

        for label in set(labels):
            if label == -1:  # Noise points
                continue

            cluster_points = user_pings[user_pings['cluster_label'] == label]
            # This check is technically redundant due to DBSCAN's min_samples, but good for safety
            if len(cluster_points) < DBSCAN_MIN_SAMPLES_HW:
                continue

            stay_center_lat = cluster_points['latitude'].mean()
            stay_center_lon = cluster_points['longitude'].mean()
            stay_start_time = cluster_points['timestamp'].min()
            stay_end_time = cluster_points['timestamp'].max()
            duration_minutes = (stay_end_time - stay_start_time).total_seconds() / 60.0

            # Filter by duration
            if not (MIN_STAY_DURATION_MINUTES_HW <= duration_minutes <= MAX_STAY_DURATION_HOURS_HW * 60):
                continue

            # POI Assignment
            matched_poi_id = None
            matched_poi_type = None
            if poi_kdtree is not None:
                # KDTree query gives Euclidean distance on lat/lon, need to verify with Haversine
                # For large POI sets, a spatial index designed for Haversine would be better, or project coordinates. KDTree on lat/lon is an approximation for finding candidates.
                try:
                    # Query for more than 1 NN to check distances if the first one is too far
                    distances_approx, indices = poi_kdtree.query([[stay_center_lat, stay_center_lon]],
                                                                 k=min(5, len(poi_df)),
                                                                 distance_upper_bound=eps_rad * 2)  # eps_rad*2 as rough upper bound for query
                    # Filter out infinite distances which mean no neighbor within bound
                    valid_indices = indices[np.isfinite(distances_approx)]

                    best_poi_idx = -1
                    min_actual_dist_m = float('inf')

                    for poi_idx in valid_indices:
                        if poi_idx >= len(poi_df):
                            continue  # Should not happen with valid kdtree

                        poi_candidate = poi_df.iloc[poi_idx]
                        actual_dist_m = haversine_distance(stay_center_lat, stay_center_lon,
                                                           poi_candidate['latitude'], poi_candidate['longitude'])
                        if actual_dist_m <= POI_ASSIGNMENT_RADIUS_METERS_HW:
                            if actual_dist_m < min_actual_dist_m:
                                min_actual_dist_m = actual_dist_m
                                best_poi_idx = poi_idx

                    if best_poi_idx != -1:
                        matched_poi_id = poi_df.iloc[best_poi_idx]['poi_id']
                        matched_poi_type = poi_df.iloc[best_poi_idx]['type']
                    else:
                        continue  # No POI matched within radius, discard stay as per paper logic
                except Exception as e_kdtree:
                    logging.warning(f"KDTree query or POI matching failed for user {user_id}, stay {label}: {e_kdtree}")
                    continue  # If POI matching fails, discard stay
            elif POI_ASSIGNMENT_RADIUS_METERS_HW > 0:  # If POI anchoring was intended but poi_kdtree is None
                logging.debug(
                    f"Skipping POI assignment for stay {label} of user {user_id} as POI data is unavailable/invalid.")
                continue  # Discard stay if POI anchoring is crucial

            all_stays_list.append({
                "user_id": user_id,
                "stay_id": f"{user_id}_stay_{label}",
                "latitude": stay_center_lat,
                "longitude": stay_center_lon,
                "start_time": stay_start_time,
                "end_time": stay_end_time,
                "duration_minutes": duration_minutes,
                "num_pings": len(cluster_points),
                "poi_id": matched_poi_id,
                "poi_type": matched_poi_type
            })

    stays_df = pd.DataFrame(all_stays_list)
    if not stays_df.empty:
        logging.info(f"Identified {len(stays_df)} significant stays for {stays_df['user_id'].nunique()} users.")
    else:
        logging.info("No significant stays were identified.")
    return stays_df


def infer_home_locations(stays_df):
    """
    Infers home locations for individuals based on nighttime stay patterns and POI types.

    Methodology:
    1. Filters for stays occurring during nighttime hours (e.g., 21:00-06:00).
    2. Groups these nighttime stays by user and location (approximated by POI or coordinates).
    3. Calculates cumulative duration and visit frequency (number of unique days) for each potential home candidate.
    4. Filters candidates: must meet a minimum visit threshold and be associated with a 'residential' POI.
    5. Selects the candidate with the longest total nighttime duration as the home location.

    Input:
        stays_df: DataFrame of significant stays (output from `identify_significant_stays`).
                  Required columns: 'user_id', 'latitude', 'longitude', 'start_time',
                                    'duration_minutes', 'poi_id', 'poi_type'.
    Output:
        home_locations_df: DataFrame with inferred home locations.
                           Columns: 'user_id', 'home_latitude', 'home_longitude', 'home_poi_id'.
                           Returns an empty DataFrame if no homes are inferred.
    """
    logging.info("Starting home location inference...")
    if stays_df.empty or not all(
            col in stays_df.columns for col in ['user_id', 'start_time', 'duration_minutes', 'poi_type']):
        logging.error("Stays DataFrame is empty or missing required columns for home inference.")
        return pd.DataFrame(columns=['user_id', 'home_latitude', 'home_longitude', 'home_poi_id'])

    # Ensure 'start_time' is datetime
    stays_df['start_time'] = pd.to_datetime(stays_df['start_time'])
    stays_df['hour'] = stays_df['start_time'].dt.hour
    stays_df['date'] = stays_df['start_time'].dt.date  # For counting unique visit days

    home_locations_list = []
    for user_id, user_stays in stays_df.groupby('user_id'):
        night_stays = user_stays[
            (user_stays['hour'] >= HOME_NIGHT_START_HOUR_HW) | (user_stays['hour'] < HOME_NIGHT_END_HOUR_HW)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning on 'date'

        if night_stays.empty:
            continue

        # Group by location (POI ID is a good proxy if available and reliable)
        grouping_cols = ['latitude', 'longitude']
        if 'poi_id' in night_stays.columns and night_stays['poi_id'].notna().any():
            grouping_cols = ['poi_id', 'latitude', 'longitude', 'poi_type']  # Use POI if available

        home_candidates = night_stays.groupby(grouping_cols, as_index=False).agg(
            total_night_duration=('duration_minutes', 'sum'),
            visit_days=('date', 'nunique')  # Count unique dates
        )

        # Filter by visit frequency and POI type
        home_candidates = home_candidates[home_candidates['visit_days'] >= HOME_MIN_VISITS_HW]
        if 'poi_type' in home_candidates.columns:  # Ensure 'poi_type' exists from grouping
            home_candidates = home_candidates[home_candidates['poi_type'] == 'residential']
        else:  # If poi_type was not used for grouping (e.g. all POIs were None)
            logging.debug(f"No 'poi_type' in home_candidates for user {user_id}, cannot filter by residential POI.")
            continue

        if home_candidates.empty:
            continue

        # Select best candidate (longest total nighttime duration)
        best_home_candidate = home_candidates.loc[home_candidates['total_night_duration'].idxmax()]

        home_data = {
            "user_id": user_id,
            "home_latitude": best_home_candidate['latitude'],
            "home_longitude": best_home_candidate['longitude'],
            "home_poi_id": best_home_candidate.get('poi_id', None)  # Get POI ID if it was part of grouping
        }
        home_locations_list.append(home_data)

    home_locations_df = pd.DataFrame(home_locations_list)
    if not home_locations_df.empty:
        logging.info(f"Inferred home locations for {len(home_locations_df)} users.")
    else:
        logging.info("No home locations were inferred.")
    return home_locations_df


def infer_workplace_locations(stays_df):
    """
    Infers workplace locations for individuals based on daytime stay patterns on workdays and POI types.

    Methodology:
    1. Filters for stays occurring during typical working hours (e.g., 09:00-17:00) on workdays (Mon-Fri).
    2. Groups these workday stays by user and location.
    3. Calculates cumulative duration and visit frequency for each potential workplace candidate.
    4. Filters candidates: must meet a minimum visit threshold (e.g., 4 visits per 5 workdays)
       and be associated with a 'commercial' or 'office' POI.
    5. Selects the candidate with the longest total working-hour duration as the workplace.

    Input:
        stays_df: DataFrame of significant stays.
                  Required columns: 'user_id', 'latitude', 'longitude', 'start_time', 'end_time',
                                    'duration_minutes', 'poi_id', 'poi_type'.
    Output:
        work_locations_df: DataFrame with inferred workplace locations.
                           Columns: 'user_id', 'work_latitude', 'work_longitude', 'work_poi_id'.
                           Returns an empty DataFrame if no workplaces are inferred.
    """
    logging.info("Starting workplace location inference...")
    if stays_df.empty or not all(
            col in stays_df.columns for col in ['user_id', 'start_time', 'duration_minutes', 'poi_type']):
        logging.error("Stays DataFrame is empty or missing required columns for workplace inference.")
        return pd.DataFrame(columns=['user_id', 'work_latitude', 'work_longitude', 'work_poi_id'])

    stays_df['start_time'] = pd.to_datetime(stays_df['start_time'])
    stays_df['hour'] = stays_df['start_time'].dt.hour
    stays_df['dayofweek'] = stays_df['start_time'].dt.dayofweek  # Monday=0, Sunday=6
    stays_df['date'] = stays_df['start_time'].dt.date

    work_locations_list = []

    # Determine number of workdays in the observation period for visit threshold calculation
    min_date_in_data = stays_df['date'].min()
    max_date_in_data = stays_df['date'].max()
    if pd.isna(min_date_in_data) or pd.isna(max_date_in_data):
        logging.warning("Cannot determine date range from stays_df for workday calculation. Using default 20 workdays.")
        num_workdays_in_period = 20  # Default if dates are problematic (approx 4 weeks)
    else:
        try:
            num_workdays_in_period = len(pd.bdate_range(min_date_in_data, max_date_in_data))
        except Exception:  # Handle cases where min_date == max_date etc.
            num_workdays_in_period = 1 if min_date_in_data == max_date_in_data and min_date_in_data.weekday() < 5 else 20

    min_total_work_visits = (num_workdays_in_period / 5.0) * WORK_MIN_VISITS_PER_5_WORKDAYS_HW \
        if num_workdays_in_period > 0 else WORK_MIN_VISITS_PER_5_WORKDAYS_HW
    min_total_work_visits = max(1, int(round(min_total_work_visits)))  # Ensure at least 1 visit

    for user_id, user_stays in stays_df.groupby('user_id'):
        workday_stays = user_stays[
            (user_stays['dayofweek'] < 5) &  # Monday to Friday
            (user_stays['hour'] >= WORK_DAY_START_HOUR_HW) &
            (user_stays['hour'] < WORK_DAY_END_HOUR_HW)
            ].copy()

        if workday_stays.empty:
            continue

        grouping_cols = ['latitude', 'longitude']
        if 'poi_id' in workday_stays.columns and workday_stays['poi_id'].notna().any():
            grouping_cols = ['poi_id', 'latitude', 'longitude', 'poi_type']

        work_candidates = workday_stays.groupby(grouping_cols, as_index=False).agg(
            total_work_duration=('duration_minutes', 'sum'),
            visit_days=('date', 'nunique')
        )

        work_candidates = work_candidates[work_candidates['visit_days'] >= min_total_work_visits]
        if 'poi_type' in work_candidates.columns:
            work_candidates = work_candidates[
                work_candidates['poi_type'].isin(['commercial', 'office'])]  # Paper mentions 'commercial POIs'
        else:
            logging.debug(
                f"No 'poi_type' in work_candidates for user {user_id}, cannot filter by commercial/office POI.")
            continue

        if work_candidates.empty:
            continue

        best_work_candidate = work_candidates.loc[work_candidates['total_work_duration'].idxmax()]

        work_data = {
            "user_id": user_id,
            "work_latitude": best_work_candidate['latitude'],
            "work_longitude": best_work_candidate['longitude'],
            "work_poi_id": best_work_candidate.get('poi_id', None)
        }
        work_locations_list.append(work_data)

    work_locations_df = pd.DataFrame(work_locations_list)
    if not work_locations_df.empty:
        logging.info(f"Inferred workplace locations for {len(work_locations_df)} users.")
    else:
        logging.info("No workplace locations were inferred.")
    return work_locations_df


def extract_trips_from_stays(stays_df, pings_df):
    """
    Extracts individual trips from a sequence of time-ordered significant stays for each user.
    A trip is defined as the movement between the end of one stay and the start of the next consecutive stay.

    Methodology:
    1. Sorts stays by user and start time.
    2. For each user, iterates through consecutive pairs of stays.
    3. Defines a trip with origin from the first stay and destination to the second.
    4. Calculates travel duration.

    Input:
        stays_df: DataFrame of significant stays (output from `identify_significant_stays`).
                  Required columns: 'user_id', 'stay_id', 'latitude', 'longitude',
                                    'start_time', 'end_time', 'poi_id' (optional but good).
        pings_df: DataFrame of raw GPS pings, used to extract trajectory points for each trip.
                  Required columns: 'user_id', 'latitude', 'longitude', 'timestamp'.
    Output:
        trips_df: DataFrame of extracted trips.
                  Columns: 'trip_id', 'user_id',
                           'origin_latitude', 'origin_longitude', 'origin_poi_id',
                           'destination_latitude', 'destination_longitude', 'destination_poi_id',
                           'departure_time' (end of origin stay), 'arrival_time' (start of dest stay),
                           'travel_duration_minutes', 'gps_trajectory_points' (list of [lat,lon] coords),
                           'o_yx_str' (origin 'lat,lon' WGS84),
                           'd_yx_str' (destination 'lat,lon' WGS84).
                  Returns an empty DataFrame if no trips are extracted.
    """
    logging.info("Starting trip extraction from stays...")
    if stays_df.empty or not all(col in stays_df.columns for col in ['user_id', 'start_time', 'end_time']):
        logging.error("Stays DataFrame is empty or missing required columns for trip extraction.")
        return pd.DataFrame()
    if pings_df.empty and 'gps_trajectory_points' in stays_df.columns:  # If we want to extract waypoints
        logging.warning("Pings DataFrame is empty. GPS trajectory points for trips will not be extracted.")
        # Allow to proceed, but gps_trajectory_points will be empty.

    # Ensure datetime types
    stays_df['start_time'] = pd.to_datetime(stays_df['start_time'])
    stays_df['end_time'] = pd.to_datetime(stays_df['end_time'])
    if not pings_df.empty:
        pings_df['timestamp'] = pd.to_datetime(pings_df['timestamp'])

    # Sort stays to process them chronologically for each user
    sorted_stays = stays_df.sort_values(by=['user_id', 'start_time'])

    trips_list = []
    trip_counter = 0

    for user_id, user_stays in sorted_stays.groupby('user_id'):
        if len(user_stays) < 2:  # Need at least two stays to form a trip
            continue

        for i in range(len(user_stays) - 1):
            origin_stay = user_stays.iloc[i]
            destination_stay = user_stays.iloc[i + 1]

            trip_departure_time = origin_stay['end_time']
            trip_arrival_time = destination_stay['start_time']

            # Ensure the destination stay starts after the origin stay ends
            if trip_arrival_time > trip_departure_time:
                travel_duration_minutes = (trip_arrival_time - trip_departure_time).total_seconds() / 60.0

                # Extract GPS trajectory points for the trip from raw pings
                trip_gps_points = []
                if not pings_df.empty:
                    user_trip_pings = pings_df[
                        (pings_df['user_id'] == user_id) &
                        (pings_df['timestamp'] > trip_departure_time) &  # Strictly after origin stay end
                        (pings_df['timestamp'] < trip_arrival_time)  # Strictly before dest stay start
                        ].sort_values('timestamp')

                    if not user_trip_pings.empty:
                        trip_gps_points = user_trip_pings[['latitude', 'longitude']].values.tolist()

                trips_list.append({
                    "trip_id": f"trip_{trip_counter}",  # Global trip counter for unique ID
                    "user_id": user_id,
                    "origin_latitude": origin_stay['latitude'],
                    "origin_longitude": origin_stay['longitude'],
                    "origin_poi_id": origin_stay.get('poi_id'),  # Use .get for optional column
                    "destination_latitude": destination_stay['latitude'],
                    "destination_longitude": destination_stay['longitude'],
                    "destination_poi_id": destination_stay.get('poi_id'),
                    "departure_time": trip_departure_time,
                    "arrival_time": trip_arrival_time,
                    "travel_duration_minutes": travel_duration_minutes,
                    "gps_trajectory_points": trip_gps_points,
                    "o_yx_str": f"{origin_stay['latitude']},{origin_stay['longitude']}",
                    "d_yx_str": f"{destination_stay['latitude']},{destination_stay['longitude']}"
                })
                trip_counter += 1

    trips_df = pd.DataFrame(trips_list)
    if not trips_df.empty:
        logging.info(f"Extracted {len(trips_df)} trips for {trips_df['user_id'].nunique()} users.")
    else:
        logging.info("No trips were extracted.")
    return trips_df


# --- § Population representativeness ---
def validate_population_representativeness(home_locations_df, census_df, growth_rate_estimate=0.0087):
    """
    Validates the population representativeness of mobile phone data against census data.

    Methodology:
    1. Adjusts census population data for temporal discrepancies using annual growth estimates.
    2. Aggregates home-based trip frequencies (proxied by number of identified home locations
       per township) from the mobile phone data. This requires `home_locations_df` to have
       a 'township_id' or be joinable to a townships geography.
    3. Calculates Pearson’s correlation coefficient (r) between the adjusted census population
       and the home-based trip frequencies at matched township units.

    Input:
        home_locations_df: DataFrame of inferred home locations.
                           Required columns: 'user_id', 'home_latitude', 'home_longitude'.

        census_df: DataFrame of census data.
                   Required columns: 'township_id', 'population_2020' (or similar base population).

        growth_rate_estimate: Annual population growth rate.
                              Used to project census data to the year of mobility data.

    Output:
        A dictionary containing 'pearson_r', 'p_value', and 'r_squared', or None if validation fails.
    """
    logging.info("Starting population representativeness validation...")
    if home_locations_df.empty or 'home_township_id' not in home_locations_df.columns:  # Assume 'home_township_id' exists
        logging.error("Home locations DataFrame is empty or missing 'home_township_id' for validation.")
        return None
    if census_df.empty or not all(col in census_df.columns for col in ['township_id', 'population_2020']):
        logging.error("Census DataFrame is empty or missing required columns.")
        return None

    # Adjust census population (assuming 3 years of growth from 2020 to 2023 as per paper)
    num_years_growth = 3
    census_df['population_adjusted'] = census_df['population_2020'] * ((1 + growth_rate_estimate) ** num_years_growth)

    # Aggregate home frequencies from mobility data
    home_freq_by_township = home_locations_df.groupby('home_township_id').size().reset_index(name='home_frequency')

    # Merge data for comparison
    comparison_df = pd.merge(census_df, home_freq_by_township,
                             left_on='township_id', right_on='home_township_id', how='inner')

    if len(comparison_df) < 2:  # Pearson r needs at least 2 data points
        logging.warning("Not enough matched townships for correlation after merging census and home data.")
        return None

    try:
        r, p_value = pearsonr(comparison_df['population_adjusted'], comparison_df['home_frequency'])
        r_squared = r ** 2
        logging.info(
            f"Population Representativeness: Pearson r = {r:.4f}, p-value = {p_value:.3g}, R-squared = {r_squared:.3f}")
        return {"pearson_r": r, "p_value": p_value, "r_squared": r_squared}
    except Exception as e:
        logging.error(f"Error calculating Pearson correlation: {e}")
        return None


# --- § Socioeconomic status inference ---
def infer_socioeconomic_status(home_locations_df, lianjia_property_df):
    """
    Infers socioeconomic status (approximated by income level) of individuals based on
    the transaction prices of residential communities near their inferred home locations.

    Methodology:
    1. Uses a property database (e.g., LianJia) with geotagged residential communities and average transaction prices.
    2. For each individual's home location, finds the nearest residential community from the database.
    3. Assigns the transaction price of the matched community to the individual.
    4. Validates the matching by checking the distribution of distances between homes and matched communities.
    5. Divides individuals into income quartiles based on these matched property prices.

    Input:
        home_locations_df: DataFrame of inferred home locations.
                           Required columns: 'user_id', 'home_latitude', 'home_longitude'.
        lianjia_property_df: DataFrame of LianJia property data.
                             Required columns: 'community_id','avg_transaction_price', 'latitude', 'longitude'.
    Output:
        users_with_ses_df: DataFrame (based on home_locations_df) with added columns:
                           'matched_lianjia_price', 'distance_to_community_m', 'income_quartile'.
                           Returns original home_locations_df if SES inference fails.
    """
    logging.info("Starting socioeconomic status inference...")
    if home_locations_df.empty or not all(
            col in home_locations_df.columns for col in ['user_id', 'home_latitude', 'home_longitude']):
        logging.error("Home locations DataFrame is empty or missing required columns for SES inference.")
        return home_locations_df.copy()  # Return a copy to avoid modifying original if passed by reference
    if lianjia_property_df.empty or not all(
            col in lianjia_property_df.columns for col in ['avg_transaction_price', 'latitude', 'longitude']):
        logging.error("LianJia property DataFrame is empty or missing required columns.")
        return home_locations_df.copy()

    users_with_ses_df = home_locations_df.copy()

    try:
        # Build KDTree from LianJia community locations for efficient nearest neighbor search
        lianjia_coords = lianjia_property_df[['latitude', 'longitude']].values
        lianjia_kdtree = KDTree(lianjia_coords)
    except Exception as e:
        logging.error(f"Failed to create KDTree from LianJia data: {e}. Cannot infer SES.")
        return users_with_ses_df

    home_coords = users_with_ses_df[['home_latitude', 'home_longitude']].values

    if len(home_coords) == 0:
        logging.warning("No valid home coordinates to query against LianJia data.")
        return users_with_ses_df

    try:
        distances_approx, indices = lianjia_kdtree.query(home_coords, k=1)  # Find 1 nearest neighbor
    except Exception as e:
        logging.error(f"KDTree query failed: {e}. Cannot infer SES.")
        return users_with_ses_df

    # Assign matched prices and calculate actual distances
    users_with_ses_df['matched_lianjia_price'] = lianjia_property_df.iloc[indices]['avg_transaction_price'].values

    distances_m_list = []
    for i, home_coord_row in enumerate(home_coords):
        home_lat, home_lon = home_coord_row
        matched_community_idx = indices[i]
        if matched_community_idx < len(lianjia_coords):  # Check index validity
            comm_lat, comm_lon = lianjia_coords[matched_community_idx]
            dist_m = haversine_distance(home_lat, home_lon, comm_lat, comm_lon)
            distances_m_list.append(dist_m)
        else:
            distances_m_list.append(np.nan)  # Should not happen if KDTree query is correct
    users_with_ses_df['distance_to_community_m'] = distances_m_list

    # Validate distance
    if 'distance_to_community_m' in users_with_ses_df and users_with_ses_df['distance_to_community_m'].notna().any():
        dist_80p = np.nanpercentile(users_with_ses_df['distance_to_community_m'], 80)
        dist_90p = np.nanpercentile(users_with_ses_df['distance_to_community_m'], 90)
        logging.info(
            f"Distance to matched community: 80th percentile = {dist_80p:.0f}m, 90th percentile = {dist_90p:.0f}m.")

    # Assign income quartiles
    if 'matched_lianjia_price' in users_with_ses_df and users_with_ses_df['matched_lianjia_price'].notna().any():
        try:
            users_with_ses_df['income_quartile'] = pd.qcut(
                users_with_ses_df['matched_lianjia_price'].dropna(), 4,
                labels=['Lower', 'Lower-Middle', 'Upper-Middle', 'Higher'],
                duplicates='drop'  # Important if many users match communities with same price
            )
            logging.info(f"Assigned income quartiles based on matched property prices.")
            logging.debug(
                f"Income quartile distribution:\n{users_with_ses_df['income_quartile'].value_counts(dropna=False)}")
        except ValueError as e_qcut:  # Can happen if not enough distinct price values for 4 quartiles
            logging.warning(
                f"Could not assign 4 income quartiles due to data distribution: {e_qcut}. Assigning fewer or NaN.")
            # Fallback: try fewer quartiles or just leave as NaN
            try:  # Try 2 groups (median split)
                users_with_ses_df['income_quartile'] = pd.qcut(
                    users_with_ses_df['matched_lianjia_price'].dropna(), 2,
                    labels=['Lower_Half', 'Upper_Half'], duplicates='drop'
                )
                logging.info("Assigned 2 income groups (median split) due to data distribution.")
            except:
                users_with_ses_df['income_quartile'] = np.nan
                logging.warning("Failed to assign any income groups due to data distribution.")

    else:
        logging.warning("No matched LianJia prices available to assign income quartiles.")
        users_with_ses_df['income_quartile'] = np.nan

    return users_with_ses_df


# --- § Travel mode choices ---
def train_travel_mode_model(geolife_trips_df):
    """
    Trains a Random Forest model to classify travel modes based on trip features.

    Methodology:
    1. Uses a labeled dataset (e.g., Geolife) with features like route length, OD distance,
       distance to public transport stations, and travel time.
    2. Splits data into training and validation sets.
    3. Trains a Random Forest classifier, potentially with class balancing.
    4. Evaluates the model using AUC and Brier score on the validation set.
    5. Calculates feature importances.

    Input:
        geolife_trips_df: DataFrame of labeled trips (e.g., from Geolife).
                          Required columns: 'Route length', 'OD distance', 'O_pubstation_dist',
                                            'D_pubstation_dist', 'Travel time', 'mode' (target variable).
    Output:
        A trained scikit-learn RandomForestClassifier model, or None if training fails.
        Also prints evaluation metrics to log.
    """
    logging.info("Starting travel mode model training...")
    features = ['Route length', 'OD distance', 'O_pubstation_dist', 'D_pubstation_dist', 'Travel time']
    target = 'mode'
    if geolife_trips_df.empty or not all(col in geolife_trips_df.columns for col in features + [target]):
        logging.error("Geolife trips DataFrame is empty or missing required columns for model training.")
        return None

    X = geolife_trips_df[features]
    y = geolife_trips_df[target]

    # Ensure target variable has enough classes for stratification
    if y.nunique() < 2:
        logging.error("Target variable 'mode' has fewer than 2 unique classes. Cannot train classifier.")
        return None

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e_split:
        logging.warning(f"Stratified train-test split failed: {e_split}. Using non-stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    try:
        model.fit(X_train, y_train)
    except Exception as e_fit:
        logging.error(f"Failed to fit RandomForest model: {e_fit}")
        return None

    logging.info("Travel mode model trained. Evaluating on validation set...")
    y_pred_proba_val = model.predict_proba(X_val)
    model_classes = model.classes_

    for i, mode_class in enumerate(model_classes):
        y_true_class = (y_val == mode_class).astype(int)
        auc = np.nan
        if len(np.unique(y_true_class)) > 1:  # AUC requires at least two classes in y_true
            try:
                auc = roc_auc_score(y_true_class, y_pred_proba_val[:, i])
            except ValueError as e_auc:
                logging.warning(f"Could not calculate AUC for class {mode_class}: {e_auc}")

        brier = brier_score_loss(y_true_class, y_pred_proba_val[:, i])
        logging.info(f"Mode '{mode_class}': AUC = {auc:.4f}, Brier Score = {brier:.4f}")

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance',
                                                                                                       ascending=False)
    logging.info(f"Feature importances:\n{feature_importance_df}")

    return model


def predict_travel_mode_probabilities(trips_df, model, poi_df):
    """
    Applies a pre-trained travel mode model to predict mode probabilities for new trips.

    Methodology:
    1. Calculates/extracts the required features for each trip in `trips_df`
       (Route length, OD distance, O/D_pubstation_dist, Travel time).
    2. Uses the trained `model` to predict probabilities for 'active', 'private', 'public' modes.
    3. Adds these probabilities as new columns to the `trips_df`.

    Input:
        trips_df: DataFrame of trips for which to predict modes.
                  Required columns: 'origin_latitude', 'origin_longitude',
                                    'destination_latitude', 'destination_longitude',
                                    'travel_duration_minutes'.
        model: A pre-trained scikit-learn classifier (e.g., RandomForest).
        poi_df: DataFrame of POIs, used to find nearest public transport stations.
                Required if 'O_pubstation_dist' or 'D_pubstation_dist' need calculation.
                Must contain 'latitude', 'longitude', 'type' (where type can be 'public_transport_station').

    Output:
        trips_df_with_probs: The input `trips_df` with added probability columns
                             (e.g., 'prob_active', 'prob_private', 'prob_public').
                             Returns the original DataFrame if prediction fails.
    """
    logging.info("Predicting travel mode probabilities...")
    if trips_df.empty:
        logging.error("Trips DataFrame is empty for mode prediction.")
        return trips_df.copy()
    if model is None:
        logging.error("No trained model provided for mode prediction.")
        return trips_df.copy()

    trips_df_with_probs = trips_df.copy()
    features_needed = ['Route length', 'OD distance', 'O_pubstation_dist', 'D_pubstation_dist', 'Travel time']

    # Calculate/prepare features
    # 1. OD distance
    trips_df_with_probs['OD distance'] = trips_df_with_probs.apply(
        lambda row: haversine_distance(row['origin_latitude'], row['origin_longitude'],
                                       row['destination_latitude'], row['destination_longitude']) / 1000.0  # km
        if pd.notnull(row['origin_latitude']) and pd.notnull(row['destination_latitude']) else np.nan,
        axis=1
    )
    # 2. Route length (approximated from OD distance if not available)
    if 'Route length' not in trips_df_with_probs.columns:
        trips_df_with_probs['Route length'] = trips_df_with_probs['OD distance'] * 1.2  # Common approximation

    # 3. Travel time (assume 'travel_duration_minutes' is it)
    if 'travel_duration_minutes' in trips_df_with_probs.columns:
        trips_df_with_probs['Travel time'] = trips_df_with_probs['travel_duration_minutes']
    else:
        logging.warning("'travel_duration_minutes' not in trips_df, 'Travel time' feature will be NaN.")
        trips_df_with_probs['Travel time'] = np.nan

    # 4. O_pubstation_dist and D_pubstation_dist
    # These require POI data with 'public_transport_station' type.
    pt_kdtree = None
    public_transport_pois_coords = None
    if poi_df is not None and not poi_df.empty and 'type' in poi_df.columns:
        pt_stations = poi_df[poi_df['type'] == 'public_transport_station']
        if not pt_stations.empty:
            try:
                public_transport_pois_coords = pt_stations[['latitude', 'longitude']].values
                pt_kdtree = KDTree(public_transport_pois_coords)
            except Exception as e_pt_kdtree:
                logging.warning(
                    f"Failed to create KDTree for public transport POIs: {e_pt_kdtree}. Station distances will be NaN.")
                pt_kdtree = None  # Ensure it's None if creation failed
        else:
            logging.warning("No POIs of type 'public_transport_station' found. Station distances will be NaN.")
    else:
        logging.warning(
            "POI data not available or invalid for calculating public transport station distances. These features will be NaN.")

    # Helper for station distance calculation
    def get_station_dist(lat, lon, kdtree, coords_array):
        if pd.isna(lat) or pd.isna(lon) or kdtree is None:
            return np.nan
        try:
            _, idx = kdtree.query([[lat, lon]], k=1)
            station_lat, station_lon = coords_array[idx[0]]
            return haversine_distance(lat, lon, station_lat, station_lon) / 1000.0  # km
        except Exception:
            return np.nan

    trips_df_with_probs['O_pubstation_dist'] = trips_df_with_probs.apply(
        lambda row: get_station_dist(row['origin_latitude'], row['origin_longitude'], pt_kdtree,
                                     public_transport_pois_coords), axis=1
    )
    trips_df_with_probs['D_pubstation_dist'] = trips_df_with_probs.apply(
        lambda row: get_station_dist(row['destination_latitude'], row['destination_longitude'], pt_kdtree,
                                     public_transport_pois_coords), axis=1
    )

    # Prepare feature matrix for prediction
    X_predict = trips_df_with_probs[features_needed].copy()
    X_predict = X_predict.fillna(0)

    if X_predict.empty:
        logging.warning("Feature matrix for prediction is empty after preparation.")
        return trips_df  # Return original if no data to predict

    try:
        mode_probabilities = model.predict_proba(X_predict)
        model_classes = model.classes_  # ['active', 'private', 'public'] in expected order

        for i, mode_class in enumerate(model_classes):
            trips_df_with_probs[f'prob_{mode_class}'] = mode_probabilities[:, i]
        logging.info("Added mode probabilities to trips DataFrame.")
    except Exception as e_predict:
        logging.error(f"Failed to predict mode probabilities: {e_predict}")
        # Add NaN columns if prediction fails but columns are expected
        for mc in (model.classes_ if hasattr(model, 'classes_') else ['active', 'private', 'public']):
            trips_df_with_probs[f'prob_{mc}'] = np.nan

    return trips_df_with_probs


# --- § Travel route generation ---

# Global variable for Amap API error count for route generation
AMAP_API_ERROR_COUNT = 0
AMAP_API_ERROR_LIMIT = 100  # Max errors before aborting route generation


def _parse_amap_v5_polyline(polyline_str):
    """
    Helper to parse Amap V5 polyline (lon,lat;lon,lat GCJ02) and convert to WGS84 list of [lat,lon].
    Input: polyline_str from Amap API.
    Output: List of [lat,lon] WGS84 coordinates.
    """
    if not polyline_str or not isinstance(polyline_str, str):
        return []
    coords_wgs84 = []
    for lon_lat_pair_gcj02 in polyline_str.split(';'):
        if not lon_lat_pair_gcj02: continue
        wgs_lat_lon_str = string_gcj02_to_wgs84_display_format(lon_lat_pair_gcj02)  # Returns 'lat,lon' WGS84
        if wgs_lat_lon_str:
            try:
                wgs_lat, wgs_lon = map(float, wgs_lat_lon_str.split(','))
                coords_wgs84.append([wgs_lat, wgs_lon])
            except ValueError:
                logging.warning(f"Could not parse WGS84 lat,lon from converted string: {wgs_lat_lon_str}")
    return coords_wgs84


def generate_amap_v5_routes_for_trips(input_trips_df, amap_api_key, city_code_amap='010', city_polygon_shape=None):
    """
    Generates travel routes using Amap API V5 for active, public, and private modes for each trip.

    Methodology:
    1. Iterates through each trip in `input_trips_df`.
    2. Converts origin/destination WGS84 coordinates to GCJ02 for Amap.
    3. Processes optional waypoints (from 'gps_trajectory_points') for driving routes.
    4. For each trip, calls Amap V5 Directions API for:
        - Walking ('active' mode)
        - Integrated Transit ('public' mode)
        - Driving ('private' mode)
    5. Parses the JSON response to extract route details: duration, distance, cost, and polyline segments (converted back to WGS84).
    6. Handles API errors, including rate limits.
    7. City polygon is used for an optional boundary check before calling API.

    Input:
        input_trips_df: DataFrame of trips.
                        Required columns: 'trip_id' (unique),
                                          'o_yx_str' (origin 'lat,lon' WGS84 string),
                                          'd_yx_str' (destination 'lat,lon' WGS84 string).
                        Optional column: 'gps_trajectory_points' (list of [lat,lon] WGS84 waypoints).
        amap_api_key: Your Amap Web Service API key.
        city_code_amap: Amap city code (e.g., '010' for Beijing).
        city_polygon_shape: trips with O/D outside polygon are skipped.
    Output:
        generated_routes_df: DataFrame with detailed route information for each trip-mode.
                             Columns: 'trip_id', 'mode', 'api_duration_s', 'api_distance_m',
                                      'api_cost_yuan', 'route_segments_wgs84' (list of dicts),
                                      'full_polyline_wgs84' (list of [lat,lon]), 'error' (if any).
        trips_with_route_summary_df: A copy of input_trips_df with added columns for route summary
                                     (e.g., 'active_route_details', 'public_route_details', 'private_route_details'),
                                     each containing a dictionary of key metrics or error info.
    """
    global AMAP_API_ERROR_COUNT  # Use the global error counter

    if input_trips_df is None or input_trips_df.empty:
        logging.warning("Input trips DataFrame is empty. No routes to generate.")
        return pd.DataFrame(), pd.DataFrame()
    if not amap_api_key:
        logging.error("Amap API key not provided. Cannot generate routes.")
        return pd.DataFrame(), pd.DataFrame()

    required_cols = ['trip_id', 'o_yx_str', 'd_yx_str']
    missing_cols = [col for col in required_cols if col not in input_trips_df.columns]
    if missing_cols:
        logging.error(f"Input trips DataFrame missing required columns: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame()

    logging.info(f"Starting Amap V5 route generation for {len(input_trips_df)} trips...")

    all_trip_routes_info = []
    trips_with_route_summary = input_trips_df.copy()
    for mode_col_suffix in ['active_route_details', 'public_route_details', 'private_route_details']:
        if mode_col_suffix not in trips_with_route_summary.columns:
            trips_with_route_summary[mode_col_suffix] = None  # Initialize as object type for dicts

    for index, trip_row in trips_with_route_summary.iterrows():
        if AMAP_API_ERROR_COUNT > AMAP_API_ERROR_LIMIT:
            logging.error("Amap API error limit reached globally. Stopping route generation process.")
            break

        trip_id = trip_row['trip_id']
        logging.debug(f"Processing trip {trip_id} ({index + 1}/{len(trips_with_route_summary)}) for V5 routes.")

        o_yx_wgs84_str = trip_row['o_yx_str']
        d_yx_wgs84_str = trip_row['d_yx_str']

        # Optional City Boundary Check
        if city_polygon_shape is not None:
            try:
                from shapely.geometry import Point as ShapelyPoint  # Import if using
                o_lat_str, o_lon_str = o_yx_wgs84_str.split(',')
                d_lat_str, d_lon_str = d_yx_wgs84_str.split(',')
                o_point = ShapelyPoint(float(o_lon_str), float(o_lat_str))
                d_point = ShapelyPoint(float(d_lon_str), float(d_lat_str))
                if not city_polygon_shape.contains(o_point) or not city_polygon_shape.contains(d_point):
                    logging.warning(f"Trip {trip_id} O/D outside city polygon. Skipping Amap call.")
                    error_detail = {"error": "OUTSIDE_CITY_BOUNDARY"}
                    for mode_skip in ['active', 'public', 'private']:
                        trips_with_route_summary.loc[index, f'{mode_skip}_route_details'] = error_detail
                        all_trip_routes_info.append(
                            {"trip_id": trip_id, "mode": mode_skip, **error_detail, "api_duration_s": 0,
                             "api_distance_m": 0, "api_cost_yuan": 0, "route_segments_wgs84": [],
                             "full_polyline_wgs84": []})
                    continue
            except ImportError:
                logging.debug("Shapely not available for city boundary check.")  # Allow to proceed without check
            except Exception as e_bound:
                logging.warning(f"Boundary check error for trip {trip_id}: {e_bound}")

        o_xy_gcj02_str = string_wgs84_to_gcj02_amap_format(o_yx_wgs84_str)
        d_xy_gcj02_str = string_wgs84_to_gcj02_amap_format(d_yx_wgs84_str)

        if not o_xy_gcj02_str or not d_xy_gcj02_str:
            logging.error(f"Coordinate conversion failed for trip {trip_id}. Skipping.")
            error_detail = {"error": "COORDINATE_CONVERSION_FAILED"}
            for mode_skip in ['active', 'public', 'private']:
                trips_with_route_summary.loc[index, f'{mode_skip}_route_details'] = error_detail
                all_trip_routes_info.append(
                    {"trip_id": trip_id, "mode": mode_skip, **error_detail, "api_duration_s": 0, "api_distance_m": 0,
                     "api_cost_yuan": 0, "route_segments_wgs84": [], "full_polyline_wgs84": []})
            continue

        # Waypoint processing
        waypoints_gcj02_str = ""
        if 'gps_trajectory_points' in trip_row and isinstance(trip_row['gps_trajectory_points'], list):
            raw_waypoints = trip_row.get('gps_trajectory_points', [])
            if len(raw_waypoints) > 0:
                valid_raw_waypoints = [wp for wp in raw_waypoints if isinstance(wp, (list, tuple)) and len(wp) == 2]
                if len(valid_raw_waypoints) > 0:
                    selected_waypoints_wgs84 = []
                    selected_waypoints_wgs84.append(valid_raw_waypoints[0])  # First waypoint
                    if len(valid_raw_waypoints) > 2:  # Max 2 distinct waypoints
                        selected_waypoints_wgs84.append(
                            valid_raw_waypoints[len(valid_raw_waypoints) // 2])  # Middle waypoint
                    elif len(valid_raw_waypoints) == 2:
                        selected_waypoints_wgs84.append(valid_raw_waypoints[1])  # Second if only two

                    waypoints_gcj02_list = []
                    for wp_lat_wgs, wp_lon_wgs in selected_waypoints_wgs84:
                        gcj_wp_str = string_wgs84_to_gcj02_amap_format(f"{wp_lat_wgs},{wp_lon_wgs}")
                        if gcj_wp_str: waypoints_gcj02_list.append(gcj_wp_str)
                    if waypoints_gcj02_list: waypoints_gcj02_str = ";".join(list(set(waypoints_gcj02_list)))

        for mode_name in ['active', 'public', 'private']:
            api_url, params = "", {}
            route_summary = {"error": None}
            api_duration_s, api_distance_m, api_cost_yuan = 0, 0, 0
            parsed_segments_wgs84, full_polyline_wgs84 = [], []
            route_json_response = {}  # Store API response for debugging

            # --- Setup API URL and Params for V5 ---
            params_base = {"origin": o_xy_gcj02_str, "destination": d_xy_gcj02_str, "key": amap_api_key,
                           "output": "JSON", "show_fields": "cost,polyline,navi"}
            if mode_name == 'active':
                api_url = "https://restapi.amap.com/v5/direction/walking"
                params = {**params_base}
            elif mode_name == 'public':
                api_url = "https://restapi.amap.com/v5/direction/transit/integrated"
                params = {**params_base, "city1": city_code_amap, "AlternativeRoute": "0"}
            elif mode_name == 'private':
                api_url = "https://restapi.amap.com/v5/direction/driving"
                params = {**params_base}
                if waypoints_gcj02_str: params["waypoints"] = waypoints_gcj02_str

            try:
                response = requests.get(api_url, params=params, timeout=10)  # 10s timeout
                response.raise_for_status()  # Raise HTTPError for bad responses
                route_json_response = response.json()

                if route_json_response.get('status') == '1' and 'route' in route_json_response and route_json_response['route']:
                    route_data = route_json_response['route']
                    # --- V5 Parsing Logic (Simplified from previous, ensure robustness) ---
                    if mode_name == 'active':
                        paths_data = route_data.get('paths', [])
                        if not paths_data and isinstance(route_data.get('steps'), list): paths_data = [
                            route_data]  # If route itself is the path
                        if paths_data:
                            path = paths_data[0]
                            api_duration_s = int(path.get('cost', {}).get('duration', 0))
                            api_distance_m = int(path.get('distance', 0))
                            for step in path.get('steps', []):
                                poly_str = step.get('polyline', {}).get('polyline') if isinstance(step.get('polyline'),dict) else step.get('polyline')
                                seg_poly = _parse_amap_v5_polyline(poly_str)
                                full_polyline_wgs84.extend(seg_poly)
                                parsed_segments_wgs84.append({"mode": "步行", "instruction": step.get('instruction', ''),
                                                              "distance_m": int(step.get('distance', 0)),
                                                              "duration_s": int(
                                                                  step.get('cost', {}).get('duration', 0)),
                                                              "polyline_wgs84": seg_poly})
                            route_summary = {"duration_s": api_duration_s, "distance_m": api_distance_m,
                                             "polyline_len": len(full_polyline_wgs84)}

                    elif mode_name == 'public':
                        transits = route_data.get('transits')
                        if transits:
                            first_transit = transits[0]
                            api_duration_s = int(first_transit.get('cost', {}).get('duration', 0))
                            api_distance_m = int(first_transit.get('distance', 0))
                            api_cost_yuan = float(first_transit.get('cost', {}).get('transit_fee', 0))
                            for segment_api in first_transit.get('segments', []):
                                for sub_mode_key, sub_mode_val in segment_api.items():
                                    if not sub_mode_val: continue
                                    seg_poly, s_mode, s_line = [], "未知", ""
                                    s_dist, s_dur = 0, 0
                                    if sub_mode_key == 'walking':
                                        s_mode = "步行";
                                        s_dist = int(sub_mode_val.get('distance', 0));
                                        s_dur = int(sub_mode_val.get('cost', {}).get('duration', 0))
                                        for step in sub_mode_val.get('steps', []):
                                            poly_str = step.get('polyline', {}).get('polyline') if isinstance(
                                                step.get('polyline'), dict) else step.get('polyline')
                                            seg_poly.extend(_parse_amap_v5_polyline(poly_str))
                                    elif sub_mode_key == 'bus' and sub_mode_val.get('buslines'):
                                        busline = sub_mode_val['buslines'][0]
                                        s_mode = busline.get('type', '公交');
                                        s_line = busline.get('name', '');
                                        s_dist = int(busline.get('distance', 0));
                                        s_dur = int(busline.get('cost', {}).get('duration', 0))
                                        poly_str = busline.get('polyline', {}).get('polyline') if isinstance(
                                            busline.get('polyline'), dict) else busline.get('polyline')
                                        seg_poly = _parse_amap_v5_polyline(poly_str)
                                    elif sub_mode_key == 'taxi':
                                        s_mode = "出租车";
                                        s_dist = int(sub_mode_val.get('distance', 0));
                                        s_dur = int(sub_mode_val.get('drivetime', 0))
                                        poly_str = sub_mode_val.get('polyline', {}).get('polyline') if isinstance(
                                            sub_mode_val.get('polyline'), dict) else sub_mode_val.get('polyline')
                                        seg_poly = _parse_amap_v5_polyline(poly_str)
                                    if s_dist > 0 or s_dur > 0 or seg_poly:
                                        full_polyline_wgs84.extend(seg_poly)
                                        parsed_segments_wgs84.append(
                                            {"mode": s_mode, "line": s_line, "distance_m": s_dist, "duration_s": s_dur,
                                             "polyline_wgs84": seg_poly})
                            route_summary = {"duration_s": api_duration_s, "distance_m": api_distance_m,
                                             "cost_yuan": api_cost_yuan, "polyline_len": len(full_polyline_wgs84)}

                    elif mode_name == 'private':
                        paths_data = route_data.get('paths', [])
                        if paths_data:
                            path = paths_data[0]
                            api_duration_s = int(path.get('cost', {}).get('duration', 0))
                            api_distance_m = int(path.get('distance', 0))
                            api_cost_yuan = float(route_data.get('taxi_cost', 0))  # Taxi cost from overall route
                            for step in path.get('steps', []):
                                seg_poly = _parse_amap_v5_polyline(
                                    step.get('polyline'))  # V5 driving step polyline is direct string
                                full_polyline_wgs84.extend(seg_poly)
                                parsed_segments_wgs84.append(
                                    {"instruction": step.get('instruction', ''), "road": step.get('road_name', ''),
                                     "distance_m": int(step.get('step_distance', 0)),
                                     "duration_s": int(step.get('cost', {}).get('duration', 0)),
                                     "polyline_wgs84": seg_poly})
                            route_summary = {"duration_s": api_duration_s, "distance_m": api_distance_m,
                                             "cost_yuan": api_cost_yuan, "polyline_len": len(full_polyline_wgs84)}
                # --- V5 Parsing End ---
                else:  # Status not '1' or no route key
                    error_msg = route_json_response.get('infocode', 'UNKNOWN') + ": " + route_json_response.get('info',
                                                                                                                'API_NO_ROUTE_OR_ERROR')
                    logging.warning(f"Amap V5 API for {mode_name}, trip {trip_id}: {error_msg}.")
                    route_summary["error"] = error_msg
                    if route_json_response.get('infocode') == '10003' or route_json_response.get(
                            'info') == 'USER_DAILY_QUERY_OVER_LIMIT':  # Amap daily query limit specific error code
                        AMAP_API_ERROR_COUNT = AMAP_API_ERROR_LIMIT + 1
                        logging.critical("Amap API: USER_DAILY_QUERY_OVER_LIMIT. Stopping further API calls.")
                        # Mark current and subsequent modes for this trip with error
                        for m_err_idx in range(trips_with_route_summary.columns.get_loc(f'{mode_name}_route_details'),
                                               len(trips_with_route_summary.columns)):
                            col_name = trips_with_route_summary.columns[m_err_idx]
                            if col_name.endswith("_route_details") and trips_with_route_summary.iloc[
                                index, m_err_idx] is None:
                                trips_with_route_summary.iloc[index, m_err_idx] = {
                                    "error": "USER_DAILY_QUERY_OVER_LIMIT"}
                        break  # Break inner mode loop for this trip
                    AMAP_API_ERROR_COUNT += 1

            except requests.exceptions.Timeout:
                logging.error(f"Amap V5 API request timed out for {mode_name}, trip {trip_id}.")
                route_summary["error"] = "REQUEST_TIMEOUT"
                AMAP_API_ERROR_COUNT += 1;
                time.sleep(5)  # Longer sleep after timeout
            except requests.exceptions.RequestException as e_req:
                logging.error(f"Amap V5 API request failed for {mode_name}, trip {trip_id}: {e_req}")
                route_summary["error"] = f"REQUEST_FAILED: {type(e_req).__name__}"
                AMAP_API_ERROR_COUNT += 1;
                time.sleep(3)
            except Exception as e_generic:
                logging.error(
                    f"Generic error processing Amap V5 for {mode_name}, trip {trip_id}: {e_generic}. Response: {route_json_response}")
                route_summary["error"] = f"PROCESSING_ERROR: {type(e_generic).__name__}"
                AMAP_API_ERROR_COUNT += 1

            all_trip_routes_info.append({
                "trip_id": trip_id, "mode": mode_name,
                "api_duration_s": api_duration_s, "api_distance_m": api_distance_m,
                "api_cost_yuan": api_cost_yuan, "route_segments_wgs84": parsed_segments_wgs84,
                "full_polyline_wgs84": full_polyline_wgs84, "error": route_summary.get("error")
            })
            trips_with_route_summary.loc[index, f'{mode_name}_route_details'] = route_summary

        # Check if API limit was hit inside the mode loop to break the outer trip loop
        if AMAP_API_ERROR_COUNT > AMAP_API_ERROR_LIMIT:
            # Check if the error was specifically 'USER_DAILY_QUERY_OVER_LIMIT' by inspecting one of the detail dicts
            # This check is a bit indirect, relying on the error being set in the dict.
            final_error_check_val = trips_with_route_summary.loc[
                index, 'private_route_details']  # Check last processed mode's error
            if isinstance(final_error_check_val, dict) and final_error_check_val.get(
                    "error") == "USER_DAILY_QUERY_OVER_LIMIT":
                logging.info("Breaking outer trip loop due to API daily query limit.")
                break

        time.sleep(0.02)  # Shorter delay due to Amap's QPS limits

    generated_routes_df = pd.DataFrame(all_trip_routes_info)
    if not generated_routes_df.empty:
        logging.info(
            f"Finished Amap V5 route generation. Processed {generated_routes_df['trip_id'].nunique()} unique trips.")
        successful_routes = generated_routes_df[generated_routes_df['error'].isna()]
        logging.info(f"Successfully obtained {len(successful_routes)} V5 route entries without API/parsing errors.")
    else:
        logging.warning("No Amap V5 routes were generated or stored.")

    return generated_routes_df, trips_with_route_summary


# --- § Cross-data validation ---
def perform_cross_data_validation(memda_speed_df, generated_routes_df, trips_with_mode_probs_df):
    """
    Performs cross-data validation by comparing road speeds from an independent dataset (MemDA)
    with expected traffic volumes inferred from mobile phone private mode trips.

    Methodology:
    1. Processes MemDA traffic speed data: aggregates average speed per road segment per hour.
    2. Processes mobile phone data:
        - Filters for trips predicted to be 'private' mode.
        - For each private trip, its Amap-generated route is assumed to be a sequence of road segments.
          (This requires map-matching routes to a road network, simplified here).
        - Calculates expected traffic volume for each road segment in hourly intervals by summing
          the private mode probabilities of trips using that segment.
    3. Normalizes both the MemDA speeds and mobile-derived volumes.
    4. Calculates Pearson correlation between normalized speeds and volumes for matching
       road segments and time intervals (e.g., morning peak, midday, evening peak).

    Input:
        memda_speed_df: DataFrame of road speeds (e.g., from MemDA).
                        Required columns: 'road_segment_id', 'timestamp', 'avg_speed'.
        generated_routes_df: DataFrame of Amap-generated routes (output from `generate_amap_v5_routes_for_trips`).
                             Required columns: 'trip_id', 'mode', 'route_segments_wgs84'.
                             Each item in 'route_segments_wgs84' should ideally have a 'road_name'
                             or be mappable to a 'road_segment_id'.
        trips_with_mode_probs_df: DataFrame of trips with mode probabilities.
                                  Required columns: 'trip_id', 'prob_private', 'departure_time'.
    Output:
        A dictionary of correlation results for different time periods, or None if validation fails.
    """
    logging.info("Starting cross-data validation with MemDA speeds...")
    if memda_speed_df.empty or not all(
            col in memda_speed_df.columns for col in ['road_segment_id', 'timestamp', 'avg_speed']):
        logging.error("MemDA speed DataFrame is empty or missing required columns.")
        return None
    if generated_routes_df.empty or not all(
            col in generated_routes_df.columns for col in ['trip_id', 'mode', 'route_segments_wgs84']):
        logging.error("Generated routes DataFrame is empty or missing columns.")
        return None
    if trips_with_mode_probs_df.empty or not all(
            col in trips_with_mode_probs_df.columns for col in ['trip_id', 'prob_private', 'departure_time']):
        logging.error("Trips DataFrame with mode probabilities is empty or missing columns.")
        return None

    # Process MemDA data
    memda_speed_df['timestamp'] = pd.to_datetime(memda_speed_df['timestamp'])
    memda_speed_df['hour'] = memda_speed_df['timestamp'].dt.hour
    memda_hourly_agg = memda_speed_df.groupby(['road_segment_id', 'hour'])['avg_speed'].mean().reset_index()

    if memda_hourly_agg.empty:
        logging.warning("No hourly speeds aggregated from MemDA data.")
        return None

    # Normalize speeds (z-score)
    memda_mean_speed = memda_hourly_agg['avg_speed'].mean()
    memda_std_speed = memda_hourly_agg['avg_speed'].std()
    if memda_std_speed == 0: memda_std_speed = 1  # Avoid division by zero if all speeds are same
    memda_hourly_agg['normalized_speed'] = (memda_hourly_agg['avg_speed'] - memda_mean_speed) / memda_std_speed

    # Process mobile phone data to get expected road volumes
    private_routes = generated_routes_df[generated_routes_df['mode'] == 'private'].copy()
    # Merge with trip probabilities and departure times
    # Ensure 'departure_time' is datetime
    trips_with_mode_probs_df['departure_time'] = pd.to_datetime(trips_with_mode_probs_df['departure_time'])

    routes_for_volume = pd.merge(private_routes,
                                 trips_with_mode_probs_df[['trip_id', 'prob_private', 'departure_time']],
                                 on='trip_id', how='inner')
    routes_for_volume.dropna(subset=['prob_private', 'departure_time'], inplace=True)

    all_segment_traffic_contributions = []
    if not routes_for_volume.empty:
        for _, route_row in routes_for_volume.iterrows():
            departure_hour = route_row['departure_time'].hour
            prob_private = route_row['prob_private']
            # 'route_segments_wgs84' contains dicts where each dict has a 'road' key that can be used as a mock 'road_segment_id'.
            for seg_idx, segment_data in enumerate(route_row.get('route_segments_wgs84', [])):
                if isinstance(segment_data, dict) and 'road' in segment_data:
                    # Using hash of road name + segment index for a mock ID
                    # This needs to align with how 'road_segment_id' in MemDA data is defined.
                    mock_road_id_str = segment_data.get('road', f'unknown_road_{seg_idx}')
                    mock_road_segment_id = f"road_{(hash(mock_road_id_str) % 1000)}"  # Hash to a smaller space for demo

                    all_segment_traffic_contributions.append({
                        "road_segment_id": mock_road_segment_id,
                        "hour": departure_hour,
                        "volume_contribution": prob_private  # Expected vehicles
                    })

    if not all_segment_traffic_contributions:
        logging.warning("No road segment traffic contributions derived from mobile data.")
        return None

    mobile_volumes_df = pd.DataFrame(all_segment_traffic_contributions)
    mobile_hourly_volumes = mobile_volumes_df.groupby(['road_segment_id', 'hour'])[
        'volume_contribution'].sum().reset_index()
    mobile_hourly_volumes.rename(columns={'volume_contribution': 'expected_volume'}, inplace=True)

    if mobile_hourly_volumes.empty:
        logging.warning("No hourly volumes aggregated from mobile data contributions.")
        return None

    # Normalize volumes (z-score)
    mobile_mean_volume = mobile_hourly_volumes['expected_volume'].mean()
    mobile_std_volume = mobile_hourly_volumes['expected_volume'].std()
    if mobile_std_volume == 0: mobile_std_volume = 1
    mobile_hourly_volumes['normalized_volume'] = (mobile_hourly_volumes[
                                                      'expected_volume'] - mobile_mean_volume) / mobile_std_volume

    # Merge MemDA speeds and mobile volumes for comparison
    comparison_df = pd.merge(memda_hourly_agg, mobile_hourly_volumes, on=['road_segment_id', 'hour'], how='inner')

    if len(comparison_df) < 2:
        logging.warning("Not enough matched road segment-hour pairs for correlation after merging.")
        return None

    correlation_results = {}
    time_periods = {
        "morning_peak": (9, 9),  # 9:00-10:00 (represented by hour 9)
        "midday": (13, 13),  # 13:00-14:00 (hour 13)
        "evening_peak": (17, 17)  # 17:00-18:00 (hour 17)
    }

    for period_name, (
    start_hour, end_hour) in time_periods.items():  # Simplified to single hour matching for this structure
        subset_df = comparison_df[
            (comparison_df['hour'] >= start_hour) & (comparison_df['hour'] <= end_hour)
            ]
        if len(subset_df) >= 2:
            subset_df_cleaned = subset_df[['normalized_speed', 'normalized_volume']].dropna()
            if len(subset_df_cleaned) >= 2:
                try:
                    r, p_value = pearsonr(subset_df_cleaned['normalized_speed'], subset_df_cleaned['normalized_volume'])
                    logging.info(
                        f"Cross-data validation for {period_name}: Pearson r = {r:.4f}, p-value = {p_value:.3g}")
                    correlation_results[period_name] = {'r': r, 'p_value': p_value}
                except Exception as e_corr:
                    logging.error(f"Error calculating correlation for {period_name}: {e_corr}")
                    correlation_results[period_name] = {'r': np.nan, 'p_value': np.nan}
            else:
                logging.info(f"Not enough non-NaN data for {period_name} correlation after cleaning.")
                correlation_results[period_name] = {'r': np.nan, 'p_value': np.nan}
        else:
            logging.info(f"Not enough data points for {period_name} correlation.")
            correlation_results[period_name] = {'r': np.nan, 'p_value': np.nan}

    return correlation_results


# === Main Execution Flow (Loads data, then calls processing functions) ===
if __name__ == "__main__":

    # --- User Configuration ---
    AMAP_API_KEY = "AMAP_API_KEY"  # !!! REPLACE !!!

    # Define base path for data files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir,"data_input_files")  # Data is in a subfolder named 'data_input_files'

    # File paths
    path_pings_data = os.path.join(base_data_path, "mobile_pings.csv")
    path_poi_data = os.path.join(base_data_path, "poi_data.csv")
    path_census_data = os.path.join(base_data_path, "census_data.csv")
    path_lianjia_data = os.path.join(base_data_path, "lianjia_property_data.csv")
    path_geolife_data = os.path.join(base_data_path, "geolife_trips.csv")
    path_memda_data = os.path.join(base_data_path, "memda_speed_data.csv")
    path_city_polygon = os.path.join(base_data_path, "city_boundary.shp")  # Example: beijing_boundary.shp

    # --- Load Initial Data ---
    pings_data = pd.read_csv(path_pings_data, parse_dates=['timestamp'])
    poi_data = pd.read_csv(path_poi_data)
    census_data = pd.read_csv(path_census_data)
    lianjia_data = pd.read_csv(path_lianjia_data)
    geolife_data = pd.read_csv(path_geolife_data)
    memda_data = pd.read_csv(path_memda_data, parse_dates=['timestamp'])
    city_polygon = geopandas.read_file(path_city_polygon)

    # --- Home and Workplace Identification ---
    stays_data = identify_significant_stays(pings_data, poi_data)
    home_locations_data = pd.DataFrame()
    work_locations_data = pd.DataFrame()
    all_trips_data = pd.DataFrame()  # This will be the main trips DF for subsequent steps
    home_locations_data = infer_home_locations(stays_data)
    work_locations_data = infer_workplace_locations(stays_data)
    users_data = pd.DataFrame(
        {'user_id': pings_data['user_id'].unique()})  # Start with all unique users from pings
    users_data = pd.merge(users_data, home_locations_data, on='user_id', how='left')
    users_data = pd.merge(users_data, work_locations_data, on='user_id', how='left')
    all_trips_data = extract_trips_from_stays(stays_data, pings_data)  # Extract trips from GPS pings


    # --- Population Representativeness ---
    pop_represent_results = validate_population_representativeness(
        home_locations_df=home_locations_data,
        census_df=census_data)


    # --- Socioeconomic Status Inference ---
    users_with_ses_data = infer_socioeconomic_status(
        home_locations_df=home_locations_data,
        lianjia_property_df=lianjia_data)


    # --- Travel Mode Choices ---
    trips_with_mode_probs_data = pd.DataFrame()  # Initialize
    trained_mode_model = train_travel_mode_model(geolife_data)
    trips_with_mode_probs_data = predict_travel_mode_probabilities(
        trips_df=all_trips_data.copy(),  # Pass a copy
        model=trained_mode_model,
        poi_df=poi_data)

    # Ensure trips_with_mode_probs_data exists for the next step, even if it's just all_trips_data without probs
    trips_with_mode_probs_data = all_trips_data.copy()
    for mc_col in ['prob_active', 'prob_private', 'prob_public']:
        trips_with_mode_probs_data[mc_col] = np.nan

    # --- Travel Route Generation ---
    city_polygon_gdf = geopandas.read_file(path_city_polygon)
    city_polygon_obj = city_polygon_gdf.to_crs(epsg=4326).unary_union

    generated_routes_data = pd.DataFrame()
    trips_with_route_summary_data = pd.DataFrame()
    generated_routes_data, trips_with_route_summary_data = generate_amap_v5_routes_for_trips(
        input_trips_df=all_trips_data,
        amap_api_key=AMAP_API_KEY,
        city_code_amap='010',  # Example for Beijing
        city_polygon_shape=city_polygon_obj)


    # --- Cross-Data Validation ---
    cross_val_results = perform_cross_data_validation(
        memda_speed_df=memda_data,
        generated_routes_df=generated_routes_data,
        trips_with_mode_probs_df=trips_with_mode_probs_data)
