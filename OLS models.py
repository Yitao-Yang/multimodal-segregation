import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function for VIF Calculation and Feature Selection ---
def select_features_by_vif(X_df, vif_threshold=10.0):
    """
    Selects features by iteratively removing those with VIF above a threshold.
    Input:
        X_df: DataFrame of explanatory variables (numeric).
        vif_threshold: The maximum VIF value allowed.
    Output:
        X_selected_df: DataFrame with selected features.
        selected_columns: List of column names for selected features.
    """
    logging.debug(f"Starting VIF selection. Initial features: {list(X_df.columns)}")
    X_to_process = X_df.copy()
    # Ensure all data is numeric for VIF calculation
    for col in X_to_process.columns:
        if not pd.api.types.is_numeric_dtype(X_to_process[col]):
            logging.warning(f"VIF: Column '{col}' is not numeric and will be dropped from VIF consideration.")
            X_to_process.drop(columns=[col], inplace=True)

    cols_to_keep = list(X_to_process.columns)
    dropped_features_count = 0

    while True:
        if len(cols_to_keep) < 2:  # VIF requires at least two features
            logging.debug("VIF selection: Fewer than 2 features remaining. Stopping.")
            break

        current_X_for_vif = X_to_process[cols_to_keep].dropna()  # VIF cannot handle NaNs
        if current_X_for_vif.shape[0] < 2 or current_X_for_vif.shape[1] < 2:  # Not enough data or features
            logging.debug("VIF selection: Not enough data/features after dropna for VIF. Stopping.")
            break

        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = current_X_for_vif.columns
            # Calculate VIF for each feature
            vif_values = []
            for i in range(len(current_X_for_vif.columns)):
                try:
                    vif_val = variance_inflation_factor(current_X_for_vif.values, i)
                    vif_values.append(vif_val)
                except Exception as e_vif_single:  # Catch cases like perfect collinearity for a single var
                    logging.warning(
                        f"Could not calculate VIF for {current_X_for_vif.columns[i]}: {e_vif_single}. Assigning high VIF.")
                    vif_values.append(float('inf'))  # Assign high VIF to potentially drop it
            vif_data["VIF"] = vif_values

            max_vif = vif_data["VIF"].max()

            if max_vif > vif_threshold:
                feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
                cols_to_keep.remove(feature_to_drop)
                dropped_features_count += 1
                logging.debug(f"  VIF selection: Dropped '{feature_to_drop}' (VIF: {max_vif:.2f})")
            else:
                logging.debug("VIF selection: All remaining features below threshold.")
                break
        except Exception as e:
            logging.error(f"Error during VIF calculation loop: {e}. Returning current set of columns.")
            break

    logging.info(
        f"VIF feature selection complete. Kept {len(cols_to_keep)} features. Dropped {dropped_features_count} features.")
    if not cols_to_keep:
        logging.warning(
            "VIF selection resulted in no features. This might indicate severe multicollinearity or data issues.")
        return pd.DataFrame(), []  # Return empty if no features kept

    return X_df[cols_to_keep].copy(), cols_to_keep


# --- OLS Model Estimation Function ---
def estimate_ols_model(dependent_var_series, explanatory_vars_df, vif_selection=True, vif_thresh=10.0):
    """
    Estimates an OLS regression model.
    Input:
        dependent_var_series: Pandas Series for the dependent variable (M_t).
        explanatory_vars_df: Pandas DataFrame for explanatory variables ({T_i}).
        vif_selection: Boolean, whether to perform VIF-based feature selection.
        vif_thresh: Float, VIF threshold for feature selection.
    Output:
        statsmodels.regression.linear_model.RegressionResultsWrapper object (fitted OLS model),
        or None if model estimation fails.
    """
    if dependent_var_series is None or dependent_var_series.empty:
        logging.error("Dependent variable series is empty. Cannot estimate OLS model.")
        return None
    if explanatory_vars_df is None or explanatory_vars_df.empty:
        logging.error("Explanatory variables DataFrame is empty. Cannot estimate OLS model.")
        return None

    # Align dependent and explanatory variables on their index (e.g., grid_id)
    # and drop rows with any NaNs in the combined dataset for OLS.
    combined_for_ols = pd.concat([dependent_var_series, explanatory_vars_df], axis=1, join='inner')
    # Drop rows where dependent variable is NaN, or any explanatory variable (after selection) is NaN
    combined_for_ols.dropna(subset=[dependent_var_series.name], inplace=True)  # Must have Y

    if combined_for_ols.empty:
        logging.error("No data left after aligning dependent and explanatory variables (join='inner' or dropna).")
        return None

    y = combined_for_ols[dependent_var_series.name]
    X_explanatory_aligned = combined_for_ols[explanatory_vars_df.columns]

    X_selected_for_model = X_explanatory_aligned
    if vif_selection:
        X_vif_selected, _ = select_features_by_vif(X_explanatory_aligned, vif_threshold=vif_thresh)
        # Check if X_vif_selected is not None and not empty
        if X_vif_selected is not None and not X_vif_selected.empty:
            X_selected_for_model = X_vif_selected
            # Re-align y with the rows remaining in X_vif_selected (if VIF caused row drops due to NaNs within its loop)
            common_index = y.index.intersection(X_selected_for_model.index)
            if len(common_index) < len(y) or len(common_index) < len(X_selected_for_model):
                logging.debug("Re-aligning y and X after VIF selection due to potential NaN handling in VIF.")
                y = y.loc[common_index]
                X_selected_for_model = X_selected_for_model.loc[common_index]

        else:  # VIF removed all features or returned empty
            logging.warning("VIF selection resulted in no features. OLS cannot be estimated.")
            return None

    if X_selected_for_model.empty or y.empty or X_selected_for_model.shape[0] != y.shape[0]:
        logging.error(
            "Explanatory variables or dependent variable is empty after VIF/alignment, or shapes mismatch. Cannot fit OLS.")
        return None

    # Ensure X_selected_for_model has only numeric types for statsmodels
    X_numeric_for_model = X_selected_for_model.select_dtypes(include=np.number)
    if X_numeric_for_model.shape[1] == 0:
        logging.error("No numeric explanatory variables remaining for the model.")
        return None

    # Add constant (intercept) to the model
    X_with_const = sm.add_constant(X_numeric_for_model, has_constant='skip')  # Add if not already there

    # Final check for NaNs that might have been introduced or not handled
    final_data_for_ols = pd.concat([y, X_with_const], axis=1).dropna()
    if final_data_for_ols.empty or final_data_for_ols.shape[0] < X_with_const.shape[1] + 1:  # Need enough observations
        logging.error("Not enough valid observations after final NaN drop to fit OLS model.")
        return None

    y_final = final_data_for_ols[y.name]
    X_final = final_data_for_ols[X_with_const.columns]

    try:
        model = sm.OLS(y_final, X_final)
        results = model.fit()
        logging.info(f"OLS model estimated for dependent var: {y.name}. Features: {list(X_numeric_for_model.columns)}")
        return results
    except Exception as e:
        logging.error(f"Error estimating OLS model for {y.name}: {e}")
        return None


# --- Functions to prepare data for OLS models based on granularity ---
def prepare_daily_granularity_data(
        psm_mui_data_hourly,
        explanatory_vars_at_grid_level,
        metric_column_name,
        target_day_type
):
    logging.debug(f"Preparing DAILY data: Metric '{metric_column_name}', Day '{target_day_type}'")

    required_cols_metrics = ['grid_id', 'time_slot_id', 'day_type', metric_column_name]
    if psm_mui_data_hourly is None or not all(col in psm_mui_data_hourly.columns for col in required_cols_metrics):
        logging.error(f"Hourly metrics data (psm_mui_data_hourly) is None or missing columns: {required_cols_metrics}")
        return None, None
    if explanatory_vars_at_grid_level is None or explanatory_vars_at_grid_level.empty or 'grid_id' not in explanatory_vars_at_grid_level.columns:
        logging.error(
            "Explanatory variables data (explanatory_vars_at_grid_level) is None, empty or missing 'grid_id'.")
        return None, None

    daily_data_filtered = psm_mui_data_hourly[psm_mui_data_hourly['day_type'] == target_day_type]
    if daily_data_filtered.empty:
        logging.warning(f"No data for metric '{metric_column_name}' on '{target_day_type}'.")
        return None, None

    if not pd.api.types.is_numeric_dtype(daily_data_filtered[metric_column_name]):
        logging.error(f"Metric column '{metric_column_name}' is not numeric. Cannot average.")
        return None, None

    y_daily_avg_series = daily_data_filtered.groupby('grid_id')[metric_column_name].mean().dropna()
    y_daily_avg_series.name = f"{metric_column_name}_avg_{target_day_type}"

    if y_daily_avg_series.empty:
        logging.warning(f"No daily average values after dropna for '{metric_column_name}' on '{target_day_type}'.")
        return None, None

    X_grid_features_indexed = explanatory_vars_at_grid_level.set_index('grid_id')

    # Align y_daily_avg_series and X_grid_features_indexed by their common grid_id index
    # This ensures that only grids present in both are used.
    common_grids = y_daily_avg_series.index.intersection(X_grid_features_indexed.index)
    if common_grids.empty:
        logging.warning(
            f"No common grids between daily metric and explanatory variables for {metric_column_name} ({target_day_type}).")
        return None, None

    y_aligned = y_daily_avg_series.loc[common_grids]
    X_aligned = X_grid_features_indexed.loc[common_grids]

    logging.debug(f"Data prepared for {len(y_aligned)} grids for daily OLS.")
    return y_aligned, X_aligned


def prepare_hourly_granularity_data(
        psm_mui_data_hourly,
        explanatory_vars_at_grid_level,
        metric_column_name,
        target_hour,
        target_day_type
):
    logging.debug(
        f"Preparing HOURLY data: Metric '{metric_column_name}', Hour '{target_hour}', Day '{target_day_type}'")

    required_cols_metrics = ['grid_id', 'time_slot_id', 'day_type', metric_column_name]
    if psm_mui_data_hourly is None or not all(col in psm_mui_data_hourly.columns for col in required_cols_metrics):
        logging.error(f"Hourly metrics data (psm_mui_data_hourly) is None or missing columns: {required_cols_metrics}")
        return None, None
    if explanatory_vars_at_grid_level is None or explanatory_vars_at_grid_level.empty or 'grid_id' not in explanatory_vars_at_grid_level.columns:
        logging.error(
            "Explanatory variables data (explanatory_vars_at_grid_level) is None, empty or missing 'grid_id'.")
        return None, None

    hourly_data_filtered = psm_mui_data_hourly[
        (psm_mui_data_hourly['time_slot_id'] == target_hour) &
        (psm_mui_data_hourly['day_type'] == target_day_type)
        ]
    if hourly_data_filtered.empty:
        logging.warning(f"No data for metric '{metric_column_name}' at hour '{target_hour}' on '{target_day_type}'.")
        return None, None

    y_specific_hour_series = hourly_data_filtered.set_index('grid_id')[metric_column_name].dropna()
    y_specific_hour_series.name = f"{metric_column_name}_h{target_hour}_{target_day_type}"

    if y_specific_hour_series.empty:
        logging.warning(
            f"No y values after dropna for '{metric_column_name}' at hour '{target_hour}' on '{target_day_type}'.")
        return None, None

    X_grid_features_indexed = explanatory_vars_at_grid_level.set_index('grid_id')

    common_grids = y_specific_hour_series.index.intersection(X_grid_features_indexed.index)
    if common_grids.empty:
        logging.warning(
            f"No common grids for hourly metric ({target_hour}, {target_day_type}) and explanatory variables.")
        return None, None

    y_aligned = y_specific_hour_series.loc[common_grids]
    X_aligned = X_grid_features_indexed.loc[common_grids]

    logging.debug(f"Data prepared for {len(y_aligned)} grids for hourly OLS.")
    return y_aligned, X_aligned


# === Main Execution Flow ===
if __name__ == "__main__":


    # Define base path for data files

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.join(script_dir,"data_input_files")  # Data is in a subfolder named 'data_input_files'
    path_explanatory_vars_grid = os.path.join(base_data_path, "grid_level_explanatory_variables.csv")

    # Path to the aggregated hourly PSM/MUI data per day type (workday and weekend).
    path_all_hourly_metrics = os.path.join(base_data_path, "all_hourly_psm_mui_metrics.csv")

    # --- Load Pre-calculated Segregation Metrics and Explanatory Variables ---
    explanatory_vars_df = pd.read_csv(path_explanatory_vars_grid)
    explanatory_vars_df['grid_id'] = explanatory_vars_df['grid_id'].astype(str)

    all_hourly_metrics_df = pd.read_csv(path_all_hourly_metrics)
    all_hourly_metrics_df['grid_id'] = all_hourly_metrics_df['grid_id'].astype(str)
    required_metric_cols = ['time_slot_id', 'day_type']  # And specific metric columns

    # Proceed only if essential data is loaded
    # Define metrics and day types for modeling
    # Ensure these metric column names exist in all_hourly_metrics_df
    metrics_to_model = ['psm_active', 'psm_private', 'psm_public_bus', 'psm_public_railway', 'mui_value']
    day_types_to_model = ['workday', 'weekend']
    # Filter metrics_to_model to only those present in all_hourly_metrics_df
    metrics_to_model = [m for m in metrics_to_model if m in all_hourly_metrics_df.columns]

    # --- 1. Daily Granularity Models ---
    daily_ols_results_storage = {}  # To store model results: {metric: {day_type: statsmodels_result}}
    for metric_name in metrics_to_model:
        daily_ols_results_storage[metric_name] = {}
        for day_type_name in day_types_to_model:
            # Prepare data (y_daily_avg series, X_daily DataFrame)
            y_daily_avg, X_daily_features = prepare_daily_granularity_data(
                all_hourly_metrics_df,
                explanatory_vars_df.copy(),  # Pass a copy to avoid modification by set_index
                metric_name,
                day_type_name
            )
            # Estimate OLS model
            ols_model_fit_daily = estimate_ols_model(
                y_daily_avg,
                X_daily_features.drop(columns=[col for col in ['grid_id', 'day_type', 'time_slot_id'] if
                                               col in X_daily_features.columns], errors='ignore'),
                # Ensure only T_i vars
                vif_selection=True,
                vif_thresh=10.0
            )
            daily_ols_results_storage[metric_name][day_type_name] = ols_model_fit_daily

            if ols_model_fit_daily:
                logging.info(f"    Daily Model Estimated for {metric_name} ({day_type_name}): "
                             f"R-squared = {ols_model_fit_daily.rsquared:.4f}, "
                             f"Adj. R-squared = {ols_model_fit_daily.rsquared_adj:.4f}, "
                             f"Observations = {ols_model_fit_daily.nobs}")

    # --- 2. Hourly Granularity Models ---
    hourly_ols_results_storage = {}  # {metric: {day_type: {hour: statsmodels_result}}}
    hours_to_model_demo = np.arange(24)  # hours of day
    for metric_name in metrics_to_model:
        hourly_ols_results_storage[metric_name] = {}
        for day_type_name in day_types_to_model:
            hourly_ols_results_storage[metric_name][day_type_name] = {}
            for hour_val in hours_to_model_demo:
                y_hourly_specific, X_hourly_features = prepare_hourly_granularity_data(
                    all_hourly_metrics_df,
                    explanatory_vars_df.copy(),
                    metric_name,
                    hour_val,
                    day_type_name
                )
                ols_model_fit_hourly = estimate_ols_model(
                    y_hourly_specific,
                    X_hourly_features.drop(
                        columns=[col for col in ['grid_id', 'day_type', 'time_slot_id'] if
                                 col in X_hourly_features.columns], errors='ignore'),
                    vif_selection=True,
                    vif_thresh=10.0
                )
                hourly_ols_results_storage[metric_name][day_type_name][hour_val] = ols_model_fit_hourly

                if ols_model_fit_hourly:
                    logging.info(
                        f"    Hourly Model Estimated for {metric_name} ({day_type_name}, Hour {hour_val}): "
                        f"R-squared = {ols_model_fit_hourly.rsquared:.4f}, "
                        f"Adj. R-squared = {ols_model_fit_hourly.rsquared_adj:.4f}, "
                        f"Observations = {ols_model_fit_hourly.nobs}")