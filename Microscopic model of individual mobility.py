import pandas as pd
import numpy as np
import logging
import os

# from scipy.optimize import grid_search # Not a direct function, manual grid search implemented
# For PSM calculation, functions from measuring_segregation.py would ideally be imported.
# For self-contained demo, simplified PSM versions are used internally.

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
NUM_INCOME_GROUPS_MODEL = 4
MODES_IN_MODEL = ['active', 'private', 'railway']

# Calibrated parameter values
# These will be used as defaults and as the "result" of the conceptual calibration.
CALIBRATED_DELTA_STAR = 3e-4  # delta* = 3x10^-4
CALIBRATED_BETA_ACTIVE_STAR = 0.22
CALIBRATED_BETA_PRIVATE_STAR = 2.1
CALIBRATED_BETA_RAILWAY_STAR = 0.07

# Pre-determined alpha_g values(assuming g=1 is lowest, g=4 is highest)
DEFAULT_ALPHA_G = {1: 0.203, 2: 0.349, 3: 0.596, 4: 1.0}

LOG_NUM_INCOME_GROUPS_PSM = np.log(NUM_INCOME_GROUPS_MODEL) if NUM_INCOME_GROUPS_MODEL > 1 else 0.0


# === § Model Specification ===

def calculate_perceived_travel_cost(travel_time_m, alpha_g, beta_m):
    """
    Calculates the perceived travel cost C_gm for an individual of a given income group
    using a specific travel mode.

    Methodology:
    The perceived cost is a linear combination of travel time, where the travel time (T_m)
    is weighted by a sum of the income group's value of time (alpha_g) and a
    mode-specific cost factor (beta_m).
    Cost C_gm = (alpha_g + beta_m) * T_m.

    Input:
        travel_time_m: Estimated travel time for mode 'm' (T_m). This is a numeric value.
        alpha_g: Monetary value of time for income group 'g'. This is a numeric value.
        beta_m: Mode-specific cost factor (beyond value of time) for mode 'm',
                expressed per unit of travel time. This is a numeric value.
    Output:
        Perceived travel cost (C_gm) as a numeric value. Returns np.nan if inputs are invalid.
    """
    if pd.isna(travel_time_m) or pd.isna(alpha_g) or pd.isna(beta_m):
        logging.debug("NaN input to calculate_perceived_travel_cost. Returning NaN.")
        return np.nan
    return (alpha_g + beta_m) * travel_time_m


def calculate_mode_choice_probabilities(
        perceived_costs_all_modes,
        delta_sensitivity,
        available_modes
):
    """
    Calculates the probability p_gm that an individual (from a specific income group implicitly
    defined by the input costs) chooses mode 'm', using the multinomial logit formula.

    Methodology:
    The probability of choosing a mode is proportional to the exponentiated negative
    perceived utility (or positive disutility/cost) of that mode, scaled by the sensitivity
    parameter delta. Probabilities are normalized across all available modes.
    p_gm = exp(-delta * C_gm) / sum_{m' in M} [exp(-delta * C_gm')].
    The numerical stability trick (subtracting max exponent) has been removed as per request.
    Care must be taken with the scale of delta and costs to avoid overflow/underflow.

    Input:
        perceived_costs_all_modes: A dictionary where keys are mode names (strings from
                                   `available_modes`) and values are the perceived travel
                                   costs (C_gm) for that mode for a specific individual/group.
                                   Example: {'active': 50, 'private': 100, 'railway': 30}
        delta_sensitivity: The sensitivity parameter delta (numeric), reflecting how
                           strongly cost differences influence choice.
        available_modes: A list of strings representing the names of the modes available
                         in the choice set (e.g., ['active', 'private', 'railway']).
    Output:
        mode_probabilities: A dictionary where keys are mode names and values are the
                            calculated choice probabilities (numeric, summing to 1 across modes).
                            Returns probabilities of 0 if calculation is problematic (e.g., all costs inf).
    """
    if not perceived_costs_all_modes or not available_modes:
        logging.warning("Perceived costs or available modes are empty for probability calculation.")
        return {mode: 0.0 for mode in available_modes}

    exp_neg_delta_costs = {}
    sum_exp_neg_delta_costs = 0.0

    for mode in available_modes:
        cost = perceived_costs_all_modes.get(mode)
        exp_val = 0.0  # Default if mode cost is not valid or leads to underflow

        if cost is not None and pd.notna(cost) and np.isfinite(cost):
            try:
                # Direct calculation: exp(-delta * C)
                # This can lead to overflow if -delta*C is large positive,
                # or underflow to 0.0 if -delta*C is large negative.
                exponent_value = -delta_sensitivity * cost
                if exponent_value < -700:  # Approx underflow limit for np.exp
                    exp_val = 0.0
                elif exponent_value > 700:  # Approx overflow limit for np.exp
                    exp_val = float('inf')
                    logging.warning(
                        f"Overflow encountered for mode {mode} (exponent: {exponent_value}). Cost/delta likely too extreme.")
                else:
                    exp_val = np.exp(exponent_value)
            except OverflowError:  # Should be caught by the > 700 check mostly
                exp_val = float('inf')
                logging.warning(f"OverflowError explicitly caught for mode {mode}. Assigning inf.")
        elif cost == float('inf'):  # Mode is infinitely costly (unavailable)
            exp_val = 0.0
        # If cost is None or NaN, exp_val remains 0.0

        exp_neg_delta_costs[mode] = exp_val
        if np.isfinite(exp_val):  # Only add finite values to the sum
            sum_exp_neg_delta_costs += exp_val
        elif exp_val == float('inf'):  # If any term is infinity, the sum will be infinity
            sum_exp_neg_delta_costs = float('inf')
            # No need to process further terms if one is already inf for the sum
            # (unless multiple are inf, which is handled below)

    mode_probabilities = {}
    if sum_exp_neg_delta_costs > 1e-9 and np.isfinite(sum_exp_neg_delta_costs):
        for mode in available_modes:
            mode_probabilities[mode] = exp_neg_delta_costs.get(mode, 0.0) / sum_exp_neg_delta_costs
    elif sum_exp_neg_delta_costs == float('inf'):  # Handle cases where one or more exp_val was 'inf'
        # If multiple modes have 'inf' utility, they share probability; if only one, it gets all.
        inf_modes = [m for m, ev in exp_neg_delta_costs.items() if ev == float('inf')]
        if len(inf_modes) > 0:
            prob_per_inf_mode = 1.0 / len(inf_modes)
            for mode in available_modes:
                mode_probabilities[mode] = prob_per_inf_mode if mode in inf_modes else 0.0
        else:  # Should not happen if sum_exp_neg_delta_costs is inf without any inf_modes
            for mode in available_modes: mode_probabilities[mode] = 0.0
    elif delta_sensitivity == 0:  # If delta is 0, all available choices are equally likely
        num_valid_cost_modes = len([m for m in available_modes if
                                    pd.notna(perceived_costs_all_modes.get(m)) and np.isfinite(
                                        perceived_costs_all_modes.get(m))])
        prob_val_delta0 = 1.0 / num_valid_cost_modes if num_valid_cost_modes > 0 else 0.0
        for mode in available_modes:
            is_valid_cost = pd.notna(perceived_costs_all_modes.get(mode)) and np.isfinite(
                perceived_costs_all_modes.get(mode))
            mode_probabilities[mode] = prob_val_delta0 if is_valid_cost and num_valid_cost_modes > 0 else 0.0
    else:
        # All probabilities are effectively zero (e.g., all costs were huge, all exp_val underflowed)
        logging.debug(
            f"Sum of exponentiated terms is effectively zero ({sum_exp_neg_delta_costs}). Assigning zero probabilities.")
        for mode in available_modes:
            mode_probabilities[mode] = 0.0

    return mode_probabilities


def simulate_individual_mode_choices(
        individuals_df,
        travel_times_df,
        alpha_g_values,
        beta_m_values,
        delta_sensitivity,
        available_modes
):
    """
    Simulates mode choices for a population of individuals based on perceived travel costs
    and a multinomial logit model.

    Methodology:
    For each individual in `individuals_df`:
    1. Retrieves their income group (`income_group`).
    2. Fetches their specific travel times (`T_m`) for each mode in `available_modes`
       from `travel_times_df`.
    3. Looks up the `alpha_g` for their income group and `beta_m` for each mode.
    4. Calculates the perceived travel cost `C_gm` for each mode using `calculate_perceived_travel_cost`.
    5. Calculates the mode choice probabilities `p_gm` for each mode using
       `calculate_mode_choice_probabilities`.
    The results are compiled into a DataFrame.

    Input:
        individuals_df: DataFrame of individuals.
                        Required columns: 'user_id' (unique identifier),
                                          'income_group' (identifier for their income group,
                                                          matching keys in `alpha_g_values`).
        travel_times_df: DataFrame of pre-calculated travel times.
                         Required columns: 'user_id' (to match `individuals_df`),
                                           'mode' (mode name, matching `available_modes` and
                                                   keys in `beta_m_values`),
                                           'travel_time' (numeric, T_m for that user-mode pair).
        alpha_g_values: Dictionary mapping income group identifiers to their `alpha_g`
                        (value of time) parameters. Example: {1: 0.203, 2: 0.349, ...}
        beta_m_values: Dictionary mapping mode names to their `beta_m` (mode-specific cost
                       factor) parameters. Example: {'active': 0.22, 'private': 2.1, ...}
        delta_sensitivity: The sensitivity parameter `delta` for the logit model (numeric).
        available_modes: A list of strings defining the modes in the choice set
                         (e.g., ['active', 'private', 'railway']).

    Output:
        mode_choice_probs_df: DataFrame containing mode choice probabilities for each individual.
                              Columns: 'user_id', 'income_group', 'prob_active', 'prob_private',
                                       'prob_railway' (or other modes as specified).
                              Returns an empty DataFrame if inputs are invalid or simulation fails.
    """
    logging.info("Simulating individual mode choices for population...")
    if individuals_df.empty or travel_times_df.empty:
        logging.error("Individuals or travel times data is empty. Cannot simulate mode choices.")
        return pd.DataFrame()
    if not alpha_g_values or not beta_m_values:
        logging.error("Alpha_g or Beta_m parameter dictionaries are empty.")
        return pd.DataFrame()

    all_individual_probs_list = []

    # Prepare travel times for efficient lookup
    travel_times_df['mode'] = travel_times_df['mode'].astype(str)  # Ensure mode is string for dict keys
    tt_lookup = travel_times_df.set_index(['user_id', 'mode'])['travel_time'].to_dict()

    for _, individual_row in individuals_df.iterrows():
        user_id = individual_row['user_id']
        income_g = individual_row['income_group']

        alpha_val = alpha_g_values.get(income_g)
        if alpha_val is None:
            logging.warning(f"alpha_g not found for income group {income_g} (user {user_id}). Skipping this user.")
            continue

        user_perceived_costs = {}
        for mode in available_modes:
            travel_time_tm = tt_lookup.get((user_id, mode))  # Returns None if key not found
            beta_val = beta_m_values.get(mode)

            if travel_time_tm is not None and pd.notna(travel_time_tm) and beta_val is not None:
                cost = calculate_perceived_travel_cost(travel_time_tm, alpha_val, beta_val)
                user_perceived_costs[mode] = cost if pd.notna(cost) else float('inf')
            else:
                user_perceived_costs[mode] = float('inf')
                if travel_time_tm is None: logging.debug(f"No travel time for user {user_id}, mode {mode}.")
                if beta_val is None: logging.warning(f"beta_m not found for mode {mode}.")

        user_mode_probs = calculate_mode_choice_probabilities(
            user_perceived_costs, delta_sensitivity, available_modes
        )

        prob_entry = {'user_id': user_id, 'income_group': income_g}
        prob_entry.update({f'prob_{mode}': prob for mode, prob in user_mode_probs.items()})
        all_individual_probs_list.append(prob_entry)

    mode_choice_probs_df = pd.DataFrame(all_individual_probs_list)
    if not mode_choice_probs_df.empty:
        logging.info(f"Mode choice probabilities simulated for {len(mode_choice_probs_df)} individuals.")
    else:
        logging.warning("No mode choice probabilities were simulated (check inputs and logs).")
    return mode_choice_probs_df


# === § Parameter Calibration ===

def predetermine_alpha_g(individuals_with_income_df):
    """
    Determines group-specific value of time parameters (alpha_g) by normalizing
    average inferred income levels of individuals within each income group.
    The highest income group is the reference (alpha=1).

    Methodology:
    1. Calculates the average inferred income level for each income group present in the data.
    2. Identifies the income group with the maximum average income to serve as a benchmark.
       If income group labels are not directly comparable (e.g. strings 'q1', 'q4'), this step
       assumes the highest numeric or lexicographically last group is the highest income.
       It's best if `income_group` is an ordered categorical or numeric type representing hierarchy.
    3. For each income group 'g', sets alpha_g = AvgIncome_g / AvgIncome_benchmark_group.

    Input:
        individuals_with_income_df: DataFrame of individuals.
                                    Required columns: 'user_id',
                                                      'income_group' (identifier that can be used to find the max,
                                                                      e.g., 1, 2, 3, 4 where 4 is highest income),
                                                      'inferred_income_level' (numeric value of income).
    Output:
        alpha_g_values: A dictionary mapping income group identifiers (e.g., 1, 2, 3, 4)
                        to their calculated alpha_g values (numeric).
                        Returns None if calculation fails.
    """
    logging.info("Predetermining alpha_g values based on average inferred income...")
    required_cols = ['income_group', 'inferred_income_level']
    if individuals_with_income_df.empty or not all(col in individuals_with_income_df.columns for col in required_cols):
        logging.error(f"Individuals DataFrame missing required columns for alpha_g: {required_cols}.")
        return None
    if not pd.api.types.is_numeric_dtype(individuals_with_income_df['inferred_income_level']):
        logging.error("'inferred_income_level' must be numeric to calculate averages.")
        return None

    # Drop rows where income_group or inferred_income_level is NaN before grouping
    valid_income_data = individuals_with_income_df.dropna(subset=required_cols)
    if valid_income_data.empty:
        logging.error("No valid data (after dropping NaNs in key columns) for alpha_g determination.")
        return None

    avg_income_by_group = valid_income_data.groupby('income_group')['inferred_income_level'].mean()
    if avg_income_by_group.empty:
        logging.error("Could not calculate average income by group (no valid groups found after filtering).")
        return None

    # Determine the highest income group label to use as reference (AvgIncome_4 in paper)
    # This assumes group labels are sortable or numeric where max() gives the highest income group.
    # If group labels are like 'q1', 'q2', 'q3', 'q4', then max() would be 'q4'.
    try:
        # If income_group is categorical and ordered, max() will work.
        # If it's just object/string, max() is lexicographical.
        # Best if income_group is explicitly defined as ordered categorical or numeric (e.g. 1,2,3,4)
        if pd.api.types.is_categorical_dtype(avg_income_by_group.index) and avg_income_by_group.index.ordered:
            highest_income_group_label = avg_income_by_group.index.max()
        elif pd.api.types.is_numeric_dtype(avg_income_by_group.index):
            highest_income_group_label = avg_income_by_group.index.max()
        else:  # Attempt to find group with max average income if labels are not directly ordered
            highest_income_group_label = avg_income_by_group.idxmax()
            logging.info(
                f"Income group labels not directly ordered. Using group '{highest_income_group_label}' with max average income as reference.")

    except Exception as e_max_group:
        logging.error(
            f"Could not determine highest income group label: {e_max_group}. Ensure 'income_group' is comparable.")
        return None

    avg_income_highest_group = avg_income_by_group.get(highest_income_group_label)

    if avg_income_highest_group is None or pd.isna(avg_income_highest_group) or abs(avg_income_highest_group) < 1e-6:
        logging.error(
            f"Avg income for reference group '{highest_income_group_label}' is problematic ({avg_income_highest_group}). Cannot normalize alpha_g.")
        return None

    alpha_g_values = (avg_income_by_group / avg_income_highest_group).to_dict()
    logging.info(f"Predetermined alpha_g values (relative to group '{highest_income_group_label}'): {alpha_g_values}")
    return alpha_g_values


# --- Simplified PSM calculation for Calibration ---
def _simplified_determine_route_occupancy(route_segments, spatial_unit_type):
    """
    Extremely simplified determination of spatial unit occupancy for calibration speed.
    This function is a placeholder and needs to be replaced with calls to the
    full `determine_route_occupancy_for_psm` from `measuring_segregation.py` for accuracy.

    Input:
        route_segments: List of segment dictionaries for a route.
        spatial_unit_type: String, 'grid' or 'transit'.
    Output:
        A set of dummy (spatial_unit_id, time_slot_id) tuples. For calibration,
        time_slot_id is assumed to be a single value (e.g., 0) representing the peak hour.
    """
    occupied = set()
    if not route_segments: return occupied
    time_slot_for_calib = 0  # Representing the single 9-10 AM peak hour slot

    if spatial_unit_type == 'grid':
        # Each route contributes to a few dummy grid cells for demo
        for i in range(min(2, len(route_segments))):
            # Create a simple, repeatable dummy grid ID based on segment index
            occupied.add(((0, i % 5), time_slot_for_calib))  # Example: (grid_id=(0, 0 to 4), time_slot=0)
    elif spatial_unit_type == 'transit':
        # Each segment is a unit
        for i, seg in enumerate(route_segments):
            # Use a hash of the segment_id if complex, or the id itself if simple & hashable
            seg_id_repr = str(seg.get('segment_id', f"dummy_transit_seg_{i}"))
            occupied.add((seg_id_repr, time_slot_for_calib))
    return occupied


def _simplified_calculate_psm_for_calibration(
        simulated_mode_choices_df,  # Contains user_id, income_group, prob_target_mode
        user_routes_for_mode_df,  # Contains user_id, route_segments for the target_mode
        target_mode,
        spatial_unit_type,
        all_income_groups_list  # List of all income group identifiers
):
    """
    Simplified PSM calculation for use within the calibration loop.
    This is a placeholder for speed and self-containment of this script.
    For accurate results, it should call the full PSM pipeline from `measuring_segregation.py`.

    Input:
        simulated_mode_choices_df: DataFrame from `simulate_individual_mode_choices`, filtered
                                   for users commuting at the target hour, and containing the
                                   mode choice probability for the `target_mode` (e.g., 'prob_active').
        user_routes_for_mode_df: DataFrame linking 'user_id' to their 'route_segments'
                                 for the `target_mode`.
        target_mode: The string name of the mode being analyzed (e.g., 'active').
        spatial_unit_type: String, 'grid' or 'transit', indicating the type of spatial units for this mode.
        all_income_groups_list: A list of all unique income group identifiers (e.g., [1, 2, 3, 4]).

    Output:
        A dictionary where keys are (spatial_unit_id, time_slot_id) tuples and values
        are the calculated PSM values (numeric).
    """
    expected_pops_for_calib = defaultdict(lambda: defaultdict(float))

    # Merge choices with routes to link probabilities to route segments
    prob_col = f'prob_{target_mode}'
    if prob_col not in simulated_mode_choices_df.columns:
        logging.error(f"Probability column {prob_col} not in choices df for simplified PSM.")
        return {}

    data_for_psm_calib = pd.merge(
        simulated_mode_choices_df[['user_id', 'income_group', prob_col]],
        user_routes_for_mode_df[['user_id', 'route_segments']],  # Assume this df is already filtered for target_mode
        on='user_id', how='inner'
    )

    for _, row in data_for_psm_calib.iterrows():
        prob_choice_val = row[prob_col]
        # Check for NaN/None before numeric comparison
        if pd.isna(prob_choice_val) or prob_choice_val < 1e-6 or not row['route_segments']:
            continue

        # Use simplified occupancy determination
        occupied_st_pairs_calib = _simplified_determine_route_occupancy(row['route_segments'], spatial_unit_type)

        for s, t in occupied_st_pairs_calib:  # t will typically be 0 (peak hour slot)
            expected_pops_for_calib[(s, t)][row['income_group']] += prob_choice_val

    psm_values_output_dict = {}
    # Use the global LOG_NUM_INCOME_GROUPS_PSM for normalization factor
    # Ensure all_income_groups_list is not empty for log_N_groups_calib
    log_N_groups_calib = np.log(len(all_income_groups_list)) if len(all_income_groups_list) > 1 else 0.0

    for (s, t), group_counts_dict_calib in expected_pops_for_calib.items():
        total_pop_calib = sum(group_counts_dict_calib.values())
        psm_val_calib = np.nan  # Default to NaN

        if total_pop_calib > 1e-9:  # If there is some expected population
            entropy_calib = 0.0
            for income_g_calib in all_income_groups_list:  # Iterate over all defined groups
                count_val_q = group_counts_dict_calib.get(income_g_calib, 0.0)  # Get count, default 0
                if count_val_q > 1e-9:  # Only contribute to entropy if count is positive
                    proportion_val_q = count_val_q / total_pop_calib
                    entropy_calib -= proportion_val_q * np.log(proportion_val_q)

            if log_N_groups_calib > 1e-9:  # Avoid division by zero if only 1 group
                psm_val_calib = entropy_calib / log_N_groups_calib
            elif len(all_income_groups_list) == 1 and total_pop_calib > 1e-9:  # Only one group possible and present
                psm_val_calib = 0.0  # Max segregation (or perfect uniformity of one group)

        psm_values_output_dict[(s, t)] = psm_val_calib

    return psm_values_output_dict


# --- End Simplified PSM for Calibration ---


def objective_function_for_calibration(
        params_to_calibrate_tuple,
        fixed_alpha_g_values,
        individuals_commuting_peak_df,
        travel_times_peak_df,
        observed_psm_st_mode_dict,
        user_routes_peak_df,
        available_modes_calib,
        spatial_unit_type_by_mode,
        all_income_groups_list_calib
):
    """
    Objective function L(Theta') for calibration, calculating Sum of Squared Errors (SSE)
    between model-predicted PSM and empirically observed PSM values during the target
    workday morning peak hour (9:00-10:00 AM).

    Methodology:
    1. Unpacks the `params_to_calibrate_tuple` into delta and beta_m values.
    2. Simulates individual mode choices for the `individuals_commuting_peak_df` using
       the current set of parameters and `fixed_alpha_g_values`.
    3. For each mode in `available_modes_calib`, calculates the predicted PSM values
       ($\widehat{PSM}_{s,t,m}(\Theta')$) using the simulated mode choices and routes.
       This step uses `_simplified_calculate_psm_for_calibration`.
    4. Compares these predicted PSM values with the `observed_psm_st_mode_dict` for
       all relevant spatial units 's' and the target time 't'.
    5. Computes the sum of squared differences.

    Input:
        params_to_calibrate_tuple: A tuple containing the current values of parameters
                                   being optimized: (delta, beta_active, beta_private, beta_railway).
        fixed_alpha_g_values: A dictionary of pre-determined alpha_g values.
        individuals_commuting_peak_df: DataFrame of individuals filtered for peak hour commute.
                                       Required: 'user_id', 'income_group'.
        travel_times_peak_df: DataFrame of travel times for these peak individuals and modes.
                              Required: 'user_id', 'mode', 'travel_time'.
        observed_psm_st_mode_dict: A dictionary of empirically observed PSM values for the peak hour.
                                   Structure: {mode: {(spatial_unit_id, time_slot_id): observed_psm_value, ...}}.
                                   The `time_slot_id` should correspond to the peak hour.
        user_routes_peak_df: DataFrame linking users to their route segments for each mode during peak.
                             Required: 'user_id', 'mode', 'route_segments'.
        available_modes_calib: List of mode names for calibration (e.g., ['active', 'private', 'railway']).
        spatial_unit_type_by_mode: Dictionary mapping each mode to its spatial unit type ('grid' or 'transit'),
                                   used for the simplified PSM calculation.
        all_income_groups_list_calib: List of all income group identifiers.

    Output:
        total_sse: The Sum of Squared Errors (SSE) value (numeric). Returns float('inf')
                   if the process fails or no comparable PSM values are found.
    """
    delta_cal, ba_cal, bp_cal, br_cal = params_to_calibrate_tuple

    current_beta_m_dict = {
        'active': ba_cal, 'private': bp_cal, 'railway': br_cal
    }

    # Step 1: Simulate mode choices with current parameters
    simulated_choices_df = simulate_individual_mode_choices(
        individuals_commuting_peak_df, travel_times_peak_df,
        fixed_alpha_g_values, current_beta_m_dict, delta_cal,
        available_modes_calib
    )
    if simulated_choices_df.empty:
        logging.debug(f"Objective fn: Mode choice simulation empty for params {params_to_calibrate_tuple}")
        return float('inf')  # High error if simulation fails

    total_sse = 0.0
    num_comparisons = 0

    # Step 2 & 3: Calculate predicted PSM for each mode and compare with observed
    for mode in available_modes_calib:
        # Filter routes for current mode for peak commuters
        user_routes_this_mode = user_routes_peak_df[user_routes_peak_df['mode'] == mode]
        if user_routes_this_mode.empty:
            logging.debug(f"Objective fn: No routes for mode {mode} for peak users.")
            continue  # No routes, so no predicted PSM for this mode

        current_mode_spatial_type = spatial_unit_type_by_mode.get(mode, 'grid')  # Default to grid

        # Calculate predicted PSM using the simplified internal function
        predicted_psm_st_dict_for_mode = _simplified_calculate_psm_for_calibration(
            simulated_choices_df,  # Has probs for all modes, use specific one inside
            user_routes_this_mode,  # Routes only for this mode
            mode,  # Target mode for PSM calc
            current_mode_spatial_type,
            all_income_groups_list_calib
        )

        # Get observed PSM for this mode
        observed_psm_this_mode_dict = observed_psm_peak_target_dict.get(mode, {})
        if not observed_psm_this_mode_dict:
            logging.debug(f"Objective fn: No observed PSM for mode {mode}.")
            continue  # No observed data to compare against

        # Step 4 & 5: Compare and sum squared errors
        for st_key, pred_psm_val in predicted_psm_st_dict_for_mode.items():
            obs_psm_val = observed_psm_this_mode_dict.get(st_key)
            # Ensure both predicted and observed values are valid numbers for comparison
            if obs_psm_val is not None and pd.notna(pred_psm_val) and pd.notna(obs_psm_val):
                total_sse += (pred_psm_val - obs_psm_val) ** 2
                num_comparisons += 1

    if num_comparisons == 0:
        logging.debug(
            f"Objective fn: No (spatial_unit, time_slot) units found for PSM comparison with params {params_to_calibrate_tuple}.")
        return float('inf')  # No comparable points, very high error

    # logging.debug(f"Params: {params_to_calibrate_tuple}, SSE: {total_sse}, N_comparisons: {num_comparisons}")
    return total_sse


def calibrate_model_parameters_grid_search(
        fixed_alpha_g_values,
        individuals_df_all,
        travel_times_df_all,
        observed_psm_peak_target_dict,  # Format: {mode: {(s,t_peak): obs_psm_val, ...}}
        generated_routes_df_all,
        param_ranges_and_steps,  # Dict for grid search: {'delta': (min,max,step), ...}
        all_income_groups_list_main,  # For PSM calculation
        peak_hour_commuter_selection_func=None  # Optional func: individuals_df -> peak_individuals_df
):
    """
    Calibrates model parameters (delta, beta_m for active, private, railway) using a
    Grid Search approach. It aims to minimize the Sum of Squared Errors (SSE) between
    model-predicted PSM and empirically observed PSM values, specifically for
    workday morning peak hours (e.g., 9:00-10:00 AM).

    Methodology:
    1. Filters `individuals_df_all` to select only those commuting during the target peak hour,
       using `peak_hour_commuter_selection_func` if provided, or a default sampling otherwise.
    2. Prepares the travel times and route data for this subset of peak commuters.
    3. Defines a grid of candidate parameter values for delta, beta_active, beta_private,
       and beta_railway based on the input `param_ranges_and_steps`.
    4. Iterates through every combination of parameters in this grid. For each combination:
        a. Calls `objective_function_for_calibration` to calculate the SSE.
    5. Identifies and returns the parameter combination that yielded the minimum SSE.

    Input:
        fixed_alpha_g_values: A dictionary of pre-determined alpha_g values.
        individuals_df_all: The full DataFrame of all individuals in the study.
                            Required columns: 'user_id', 'income_group'.
                            Needs to be filterable by `peak_hour_commuter_selection_func`.
        travel_times_df_all: The full DataFrame of travel times for all users and modes.
                             Required: 'user_id', 'mode', 'travel_time'.
        observed_psm_peak_target_dict: A dictionary of empirically observed PSM values for the peak hour.
                                       Structure: {mode: {(spatial_unit_id, time_slot_id_peak): obs_psm_val, ...}}.
                                       The `time_slot_id_peak` must consistently represent the target peak hour.
        generated_routes_df_all: The full DataFrame of generated routes for all users and modes.
                                 Required: 'user_id', 'mode', 'route_segments'.
        param_ranges_and_steps: A dictionary defining the search grid for each parameter.
                                Example: {'delta': (min_val, max_val, step_size),
                                          'beta_active': (min_val, max_val, step_size), ...}
        all_income_groups_list_main: List of all unique income group identifiers, passed to PSM.
        peak_hour_commuter_selection_func: An optional function that takes `individuals_df_all`
                                           and returns a DataFrame of individuals considered to be
                                           commuting during the peak hour for calibration. If None,
                                           a sample or all users might be used (less accurate).

    Output:
        best_params_calibrated: A dictionary containing the optimal calibrated parameters
                                (delta, beta_active, beta_private, beta_railway).
        min_objective_value: The minimum SSE achieved with these best parameters.
                             Returns (None, float('inf')) if calibration fails or no parameters tested.
    """
    logging.info("Starting model parameter calibration using Grid Search...")

    if peak_hour_commuter_selection_func:
        individuals_peak_df = peak_hour_commuter_selection_func(individuals_df_all)
    else:
        sample_frac_calib = 0.1 if len(individuals_df_all) > 1000 else 1.0  # Default sampling
        individuals_peak_df = individuals_df_all.sample(frac=sample_frac_calib,
                                                        random_state=42) if sample_frac_calib < 1.0 else individuals_df_all.copy()
        logging.warning(
            f"No peak commuter selection function. Using {len(individuals_peak_df)} individuals for calibration (sample/all).")

    if individuals_peak_df.empty:
        logging.error("No individuals selected for peak hour calibration. Aborting calibration.")
        return None, float('inf')

    peak_user_ids_set = set(individuals_peak_df['user_id'].unique())
    travel_times_peak_df_calib = travel_times_df_all[travel_times_df_all['user_id'].isin(peak_user_ids_set)]
    user_routes_peak_df_calib = generated_routes_df_all[generated_routes_df_all['user_id'].isin(peak_user_ids_set)]

    if travel_times_peak_df_calib.empty or user_routes_peak_df_calib.empty:
        logging.error("Missing travel times or routes for the selected peak individuals. Aborting calibration.")
        return None, float('inf')

    # Create parameter grid iterables from ranges and steps
    param_grids_iter = {}
    for param_name, (p_min, p_max, p_step) in param_ranges_and_steps.items():
        if p_step <= 1e-9:  # Avoid issues with zero or too small step
            logging.warning(f"Step for parameter '{param_name}' is {p_step}. Using single value {p_min} or adjusting.")
            if abs(p_min - p_max) < 1e-9:
                param_grids_iter[param_name] = np.array([p_min])  # Only one point
            else:  # Create a small number of points if step is problematic but range exists
                num_points = 5
                param_grids_iter[param_name] = np.linspace(p_min, p_max, num=num_points)
        else:
            # Add a small epsilon to p_max for np.arange to include it if it's a multiple of step
            param_grids_iter[param_name] = np.arange(p_min, p_max + p_step * 0.5, p_step)

    best_params_found = None
    min_sse_found = float('inf')

    total_iterations_est = np.prod([len(v_range) for v_range in param_grids_iter.values()])
    logging.info(f"Grid search will perform approximately {total_iterations_est} iterations.")
    current_iteration_count = 0

    # Modes used in this specific model, and their assumed spatial unit types for PSM calculation
    # This needs to align with how observed_psm_peak_target_dict keys are structured
    calibration_choice_modes = MODES_IN_MODEL  # ['active', 'private', 'railway']
    spatial_unit_types_for_calib = {'active': 'grid', 'private': 'grid', 'railway': 'transit'}

    # Manual iteration through the parameter grid
    for d_val_iter in param_grids_iter['delta']:
        for ba_val_iter in param_grids_iter['beta_active']:
            for bp_val_iter in param_grids_iter['beta_private']:
                for br_val_iter in param_grids_iter['beta_railway']:
                    current_iteration_count += 1
                    if total_iterations_est > 100 and current_iteration_count % max(1,
                                                                                    (total_iterations_est // 20)) == 0:
                        logging.info(f"Calibration iteration {current_iteration_count}/{total_iterations_est}...")

                    current_params_as_tuple = (d_val_iter, ba_val_iter, bp_val_iter, br_val_iter)

                    current_sse = objective_function_for_calibration(
                        current_params_as_tuple, fixed_alpha_g_values,
                        individuals_peak_df, travel_times_peak_df_calib,
                        observed_psm_peak_target_dict, user_routes_peak_df_calib,
                        calibration_choice_modes, spatial_unit_types_for_calib,
                        all_income_groups_list_main
                    )

                    if pd.notna(current_sse) and current_sse < min_sse_found:
                        min_sse_found = current_sse
                        best_params_found = {
                            'delta': d_val_iter, 'beta_active': ba_val_iter,
                            'beta_private': bp_val_iter, 'beta_railway': br_val_iter
                        }
                        logging.info(
                            f"  New best params: Delta={d_val_iter:.4g}, B_act={ba_val_iter:.3f}, B_priv={bp_val_iter:.3f}, B_rail={br_val_iter:.3f}. Min SSE: {min_sse_found:.4f}")

    if best_params_found:
        logging.info(
            f"Grid Search Calibration finished. Optimal parameters found: {best_params_found}. Minimum SSE: {min_sse_found:.4f}")
    else:
        logging.warning(
            "Grid Search Calibration did not find optimal parameters. Check ranges, steps, or objective function behavior.")

    return best_params_found, min_sse_found


# === § Policy Simulations ===
def run_policy_simulation(
        policy_scenario_name,
        delta_beta_param_name_key,  # e.g. "delta_beta_private" (key for results dict)
        delta_beta_increment_values,
        base_model_parameters,
        individuals_df_for_policy,
        travel_times_df_for_policy,
        user_routes_df_for_policy,
        spatial_unit_type_by_mode_for_psm,
        available_modes_for_policy,
        all_income_groups_list_for_psm,
        is_downtown_targeted_flag=False,
        target_beta_mode_name='private'  # Mode whose beta is modified
):
    """
    Simulates the impact of a transport policy by systematically modifying a specified
    beta_m parameter and recalculating mode choices, average costs, and segregation (PSM).

    Methodology:
    For each increment value in `delta_beta_increment_values`:
    1. Adjusts the beta parameter for `target_beta_mode_name`.
       If `is_downtown_targeted_flag` is True and `target_beta_mode_name` is 'private',
       this implies a conceptual modification for downtown-related trips. The current
       simulation applies the beta change uniformly to all individuals for that mode,
       and the "downtown-targeted" aspect would typically be handled in the
       analysis of results by comparing effects on downtown commuters vs. others.
    2. Re-simulates individual mode choices for the entire `individuals_df_for_policy`
       population using the adjusted beta values and other base model parameters.
    3. Calculates outcomes for this policy increment:
        - Mode shares for each income group.
        - Average perceived travel costs for each income group (this part is a conceptual
          placeholder in the current implementation and would need full calculation).
        - Mode-specific PSM values using the simulated choices and routes.
          (This uses `_simplified_calculate_psm_for_calibration` for speed).
    4. Collects all outcomes (mode shares, costs, PSM results) for this policy increment.

    Input:
        policy_scenario_name: A string name for identifying this simulation scenario (e.g., "Uniform Car Cost Policy").
        delta_beta_param_name_key: A string used as the key in the output dictionary to store the
                                   current increment value (e.g., "delta_beta_private").
        delta_beta_increment_values: A list or array of numeric increment values to be added to
                                     the baseline beta of the `target_beta_mode_name`.
        base_model_parameters: A dictionary containing the baseline calibrated model parameters:
                               'delta_star' (numeric sensitivity parameter),
                               'alpha_g_values' (dict: income_group_id -> alpha_g value),
                               'beta_m_star_values' (dict: mode_name -> baseline beta_m value).
        individuals_df_for_policy: DataFrame of individuals for whom to simulate policy impacts.
                                   Required columns: 'user_id', 'income_group'.
                                   If `is_downtown_targeted_flag` is True for a 'private' mode policy,
                                   this DataFrame should conceptually include an 'is_downtown_commuter'
                                   (boolean) column for subsequent differential analysis, though the
                                   beta modification here is applied uniformly for the simulation step.
        travel_times_df_for_policy: DataFrame of travel times for the individuals.
                                    Required: 'user_id', 'mode', 'travel_time'.
        user_routes_df_for_policy: DataFrame linking users to their route segments, needed for PSM calculation.
                                   Required: 'user_id', 'mode', 'route_segments'.
        spatial_unit_type_by_mode_for_psm: Dictionary mapping each mode to its spatial unit type
                                           ('grid' or 'transit') for the PSM calculation.
        available_modes_for_policy: List of mode names considered in this policy simulation
                                    (e.g., ['active', 'private', 'railway']).
        all_income_groups_list_for_psm: List of all unique income group identifiers for PSM calculation.
        is_downtown_targeted_flag: Boolean, True if the policy's cost change is conceptually targeted
                                     at downtown commuters (primarily for 'private' mode policies).
        target_beta_mode_name: String, the mode whose beta parameter is being modified by the policy
                                (e.g., 'private', 'railway', 'active').

    Output:
        policy_simulation_outputs_list: A list of dictionaries. Each dictionary corresponds to one
                                        increment value from `delta_beta_increment_values` and contains:
                                        {'policy_scenario': policy_scenario_name,
                                         delta_beta_param_name_key: increment_value,
                                         'mode_shares_by_group': {income_group: {mode: share, ...}},
                                         'avg_costs_by_group': {income_group: avg_cost_value},
                                         'psm_results_by_mode': {mode: {(s,t): psm_val, ...}}}.
                                        Returns an empty list if the simulation cannot run.
    """
    logging.info(f"--- Running Policy Simulation: {policy_scenario_name} ---")
    logging.info(
        f"Modifying beta for mode: '{target_beta_mode_name}' using increments for '{delta_beta_param_name_key}'.")

    collected_policy_results = []

    # Unpack base parameters
    delta_for_sim = base_model_parameters['delta_star']
    alpha_g_for_sim = base_model_parameters['alpha_g_values']
    beta_m_baseline_for_sim = base_model_parameters['beta_m_star_values'].copy()  # Use a copy

    if target_beta_mode_name not in beta_m_baseline_for_sim:
        logging.error(
            f"Target beta mode '{target_beta_mode_name}' not found in baseline beta parameters. Aborting policy simulation.")
        return []

    for current_delta_beta_val in delta_beta_increment_values:
        logging.info(f"  Simulating with {delta_beta_param_name_key} = {current_delta_beta_val:.3f}")

        # Create current beta_m values for this policy increment
        current_beta_m_policy_values = beta_m_baseline_for_sim.copy()

        # Apply policy modification to the target beta parameter
        # The "downtown-targeted" aspect: the paper implies beta_private is modified for trips to/from downtown.
        # This function, using a single set of beta_m for simulate_individual_mode_choices,
        # applies the change uniformly. The "targeted" nature would then be analyzed by comparing
        # outcomes for downtown_commuters vs. non_downtown_commuters from the individuals_df.
        modified_beta_value = beta_m_baseline_for_sim[target_beta_mode_name] + current_delta_beta_val
        current_beta_m_policy_values[target_beta_mode_name] = modified_beta_value

        if is_downtown_targeted_flag and target_beta_mode_name == 'private':
            logging.info(f"    Downtown-targeted policy context: beta_private for relevant trips conceptually "
                         f"changed to {modified_beta_value:.3f}. Simulation applies this beta uniformly; "
                         f"subsequent analysis should differentiate effects based on 'is_downtown_commuter' flag.")
        else:
            logging.info(f"    Uniform policy change: beta_{target_beta_mode_name} set to {modified_beta_value:.3f}.")

        # 1. Simulate mode choices with the new (policy-affected) beta values
        simulated_choices_policy_df = simulate_individual_mode_choices(
            individuals_df_for_policy, travel_times_df_for_policy,
            alpha_g_for_sim, current_beta_m_policy_values, delta_for_sim,
            available_modes_for_policy
        )
        if simulated_choices_policy_df.empty:
            logging.warning(
                f"  Mode choice simulation failed for {delta_beta_param_name_key}={current_delta_beta_val}. Storing empty results for this step.")
            collected_policy_results.append({
                'policy_scenario': policy_scenario_name, delta_beta_param_name_key: current_delta_beta_val,
                'mode_shares_by_group': {}, 'avg_costs_by_group': {}, 'psm_results_by_mode': {}
            })
            continue

        # 2. Calculate mode shares per income group
        mode_shares_by_income_group = {}
        if 'income_group' in simulated_choices_policy_df.columns:
            for inc_g, g_data in simulated_choices_policy_df.groupby('income_group'):
                g_shares = {}
                total_in_g = len(g_data)
                if total_in_g > 0:
                    for mode_p in available_modes_for_policy:
                        prob_col_p = f'prob_{mode_p}'
                        if prob_col_p in g_data.columns:
                            g_shares[mode_p] = g_data[prob_col_p].sum() / total_in_g
                mode_shares_by_income_group[inc_g] = g_shares

        # 3. Calculate average travel costs per income group (Conceptual - requires C_gm for each individual/mode)
        # Placeholder: Actual calculation involves E[Cost_g] = sum_m(p_gm_policy * C_gm_policy_for_individual_avg_over_group)
        avg_costs_by_income_group = {g: np.random.uniform(50, 150) for g in alpha_g_for_sim.keys()}  # Dummy values

        # 4. Re-calculate PSM values using the new mode choice probabilities
        psm_results_policy_by_mode = {}
        if not user_routes_df_for_policy.empty:
            for mode_p in available_modes_for_policy:
                routes_this_mode_p = user_routes_df_for_policy[user_routes_df_for_policy['mode'] == mode_p]
                mode_s_type_p = spatial_unit_type_by_mode_for_psm.get(mode_p, 'grid')

                # Using simplified PSM for demonstration speed. Replace with full PSM for accuracy.
                psm_dict_for_policy = _simplified_calculate_psm_for_calibration(
                    simulated_choices_policy_df,  # Contains updated probabilities
                    routes_this_mode_p,
                    mode_p,
                    mode_s_type_p,
                    all_income_groups_list_for_psm
                )
                psm_results_policy_by_mode[mode_p] = psm_dict_for_policy

        collected_policy_results.append({
            'policy_scenario': policy_scenario_name,
            delta_beta_param_name_key: current_delta_beta_val,  # Store the increment value with its original key name
            'mode_shares_by_group': mode_shares_by_income_group,
            'avg_costs_by_group': avg_costs_by_income_group,
            'psm_results_by_mode': psm_results_policy_by_mode  # Dict: {mode: {(s,t): psm_val, ...}}
        })

    logging.info(f"--- Finished Policy Simulation: {policy_scenario_name} ---")
    return collected_policy_results


# === Main Execution ===
if __name__ == "__main__":


    # --- Define File Paths and Load Prerequisite Data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir,"data_input_files")  # Data is in a subfolder named 'data_input_files'
    path_individuals = os.path.join(base_data_path, "individuals_for_model.csv")
    path_travel_times = os.path.join(base_data_path, "travel_times_for_model.csv")
    path_generated_routes = os.path.join(base_data_path, "routes_for_model.csv")
    path_observed_psm_peak = os.path.join(base_data_path,"observed_psm_peak_hour.json")  # PSM dict stored as JSON

    # Load data with error handling
    individuals_data = pd.read_csv(path_individuals)
    req_ind_cols = ['user_id', 'income_group', 'inferred_income_level', 'is_downtown_commuter']
    travel_times = pd.read_csv(path_travel_times)
    generated_routes = pd.read_csv(path_generated_routes)
    import ast
    generated_routes['route_segments'] = generated_routes['route_segments'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip().startswith('[') else [])


    observed_psm_peak_dict = {}
    with open(path_observed_psm_peak, 'r') as f:
        raw_observed_psm = json.load(f)
        observed_psm_peak_dict = raw_observed_psm  # Assuming keys are fine as strings for simplified PSM

    #  Parameter Calibration ---
    income_groups_list_main = list(
        individuals_data['income_group'].unique()) if not individuals_data.empty else [1, 2, 3, 4]
    alpha_g_calibrated = predetermine_alpha_g(individuals_data)
    if alpha_g_calibrated is None:
        logging.warning("Using default alpha_g values as predetermination failed.")
        alpha_g_calibrated = DEFAULT_ALPHA_G.copy()
    # Parameter ranges for Grid Search (as per paper text)
    param_grid_paper_spec = {
        'delta': (0.0001, 0.01, 0.0001),
        'beta_active': (0.0, 10.0, 0.01),
        'beta_private': (0.0, 10.0, 0.01),
        'beta_railway': (0.0, 10.0, 0.01)}
    # For a runnable demo, use a drastically reduced grid or skip calibration
    param_grid_for_demo_run = {  # A very small grid
        'delta': (CALIBRATED_DELTA_STAR, CALIBRATED_DELTA_STAR, CALIBRATED_DELTA_STAR),  # Single value
        'beta_active': (CALIBRATED_BETA_ACTIVE_STAR, CALIBRATED_BETA_ACTIVE_STAR, CALIBRATED_BETA_ACTIVE_STAR),
        'beta_private': (CALIBRATED_BETA_PRIVATE_STAR, CALIBRATED_BETA_PRIVATE_STAR, CALIBRATED_BETA_PRIVATE_STAR),
        'beta_railway': (CALIBRATED_BETA_RAILWAY_STAR, CALIBRATED_BETA_RAILWAY_STAR, CALIBRATED_BETA_RAILWAY_STAR)}

    best_calibrated_params_final = None
    min_sse_final_val = float('inf')

    # This selector function needs to be defined based on how peak commuters are identified in your data
    def demo_peak_hour_commuter_selector(df_individuals, sample_n_calib=50):  # Sample 50 for demo
        return df_individuals.sample(n=min(sample_n_calib, len(df_individuals)), random_state=22) if len(
            df_individuals) > sample_n_calib else df_individuals

    best_calibrated_params_final, min_sse_final_val = calibrate_model_parameters_grid_search(
        alpha_g_calibrated,
        individuals_data,
        travel_times,
        observed_psm_peak_dict,
        generated_routes,
        param_grid_for_demo_run,  # Use the very small demo grid
        all_income_groups_list_main=income_groups_list_main,
        peak_commuter_selector_func=demo_peak_hour_commuter_selector)

    # Use calibrated values if calibration was skipped or failed
    if best_calibrated_params_final is None:

        best_calibrated_params_final = {
            'delta': CALIBRATED_DELTA_STAR,
            'beta_active': CALIBRATED_BETA_ACTIVE_STAR,
            'beta_private': CALIBRATED_BETA_PRIVATE_STAR,
            'beta_railway': CALIBRATED_BETA_RAILWAY_STAR
        }
        min_sse_final_val = "Not Calculated (Used Paper Values)"


    # --- Policy Simulations ---

    policy_base_params = {
        'delta_star': best_calibrated_params_final['delta'],
        'alpha_g_values': alpha_g_calibrated,
        'beta_m_star_values': {
            'active': best_calibrated_params_final['beta_active'],
            'private': best_calibrated_params_final['beta_private'],
            'railway': best_calibrated_params_final['beta_railway']
        }
    }
    policy_sim_spatial_unit_types = {'active': 'grid', 'private': 'grid', 'railway': 'transit'}

    # Scenario 1: Uniform citywide car cost increase
    delta_beta_private_policy_range = np.arange(0, 15.1, 0.2)
    uniform_car_policy_results = run_policy_simulation(
        "Uniform Car Cost Increase", "delta_beta_private", delta_beta_private_policy_range,
        policy_base_params, individuals_data, travel_times,
        generated_routes, policy_sim_spatial_unit_types, MODES_IN_MODEL, income_groups_list_main,
        is_downtown_targeted_policy=False, target_beta_to_modify_policy='private'
    )

    # Scenario 2: Downtown-targeted car cost increase
    delta_beta_private_policy_range = np.arange(0, 5.05, 0.1)
    downtown_car_policy_results = run_policy_simulation(
        "Downtown-Targeted Car Cost Increase", "delta_beta_private_downtown", delta_beta_private_policy_range,
        policy_base_params, individuals_data, travel_times,
        # individuals_data needs 'is_downtown_commuter'
        generated_routes, policy_sim_spatial_unit_types, MODES_IN_MODEL, income_groups_list_main,
        is_downtown_targeted_policy=True, target_beta_to_modify_policy='private'
    )

    # Scenario 3: Public transport (railway) subsidy
    delta_beta_railway_policy_range = np.arange(0, -0.0701, -0.002)
    railway_subsidy_results = run_policy_simulation(
        "Public Transport Subsidy (Railway)", "delta_beta_public", delta_beta_railway_policy_range,
        policy_base_params, individuals_data, travel_times,
        generated_routes, policy_sim_spatial_unit_types, MODES_IN_MODEL, income_groups_list_main,
        target_beta_to_modify_policy='railway'
    )

    # Scenario 4: Promoting active travel
    delta_beta_active_policy_range = np.arange(0, -0.6, -0.02)
    active_promo_results = run_policy_simulation(
        "Promoting Active Travel", "delta_beta_active", delta_beta_active_policy_range,
        policy_base_params, individuals_data, travel_times,
        generated_routes, policy_sim_spatial_unit_types, MODES_IN_MODEL, income_groups_list_main,
        target_beta_to_modify_policy='active'
    )