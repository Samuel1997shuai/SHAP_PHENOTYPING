# -*- coding: utf-8 -*-
"""
Random Forest (RF) Regression Complete Toolkit (GPU/CPU Adaptive)
Fully consistent with the original PLS version, all file names are unified as rf_xxxxx
1. RF parameter grid search (n_estimators + mtry)
2. 3D performance surface plot (n_estimators vs mtry → RMSE) + corresponding data table
3. R²/RMSE performance statistics table including full growth period (Excel + CSV dual format)
4. Origin-style visualization (fitting curves for each growth period + full growth period)
5. Independent fitting visualization for 6 major feature groups (newly added function, maintaining compatibility)
6. True/predicted value data saving (CSV format, stored in the rf_data_tables directory)
"""

# ==============================================
# 1. Dependency Import (Integrate GPU/CPU Adaptive + Original Functional Libraries)
# ==============================================
import os
import sys
import math
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# GPU / CuPy / cuML Adaptive Configuration (RF Exclusive Configuration Provided by User)
GPU_AVAILABLE = False
USE_CUPY = False
USE_CUML = False
try:
    import cupy as cp

    USE_CUPY = True
    try:
        _ = cp.zeros((1,))
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    USE_CUPY = False
    GPU_AVAILABLE = False

try:
    # Only retain cuML Random Forest import
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF

    USE_CUML = True
except Exception:
    USE_CUML = False

# Feature Definition Import
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)
from config.feature_definitions import (
    Vegetation_Index,
    Color_Index,
    Texture_Feature,
    Meteorological_Factor,
    all_features,
    coverage
)

# Environment Configuration
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese font support
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
print(f"GPU_AVAILABLE={GPU_AVAILABLE}, USE_CUPY={USE_CUPY}, USE_CUML={USE_CUML}")


# ==============================================
# 2. Experiment Parameter Configuration (All file names start with rf_, consistent with the original PLS functionality)
# ==============================================
class ExperimentConfig:
    """RF Experiment Parameter Configuration Class (Consistent with original PLS functionality, rf_ prefix for naming)"""
    # Data Related
    DATA_PATH = '../resource/data_all.xlsx'
    TARGET_COL = 'LAI'
    MISSING_VALUE_HANDLER = 'drop'

    # Dataset Splitting
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_TEST_SAMPLES = 5

    # Output Configuration (All files/directories start with rf_, maintaining the original structure)
    OUTPUT_ROOT = './rf_3d_surfaces_with_tables'  # rf_ prefix directory
    FIG_DPI = 300
    FIG_SIZE = (14, 10)
    SURFACE_CMAP = 'jet'  # Color map matching RF example plot
    PLOT_FIG_SIZE = (14, 10)
    SINGLE_GROUP_PLOT_SIZE = (12, 9)

    # RF Core Parameter Grid (Replacing original PLS parameters)
    N_ESTIMATORS_GRID = list(range(0, 801, 50))  # Number of decision trees, 0 for baseline model
    MTRY_GRID = None  # Number of features considered per tree, adaptive based on feature quantity
    MAX_ITER = np.arange(50, 351, 50)  # Retained for compatibility, not actually used by RF
    INTERP_GRID_SIZE = 50
    Z_AXIS_LIMIT = None

    # Plot Configuration (Consistent with original PLS)
    GROUP_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    GROUP_NAMES = ['Growth Period 1', 'Growth Period 2', 'Growth Period 3', 'Full Growth Period']
    GROUP_LABELS = [1, 2, 3, 4]
    FEATURE_GROUP_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Growth Period and Feature Group Configuration (Fully consistent with original PLS)
    PERIODS = {
        'Growth Period 1': slice(0, 60),
        'Growth Period 2': slice(60, 120),
        'Growth Period 3': slice(120, 180),
        'Full Growth Period': slice(None)
    }
    FEATURE_GROUPS = {
        '(A) Vegetation Index': Vegetation_Index,
        '(B) Color Index': Color_Index,
        '(C) Texture features': Texture_Feature,
        '(D) Meteorological Factor': Meteorological_Factor,
        '(E) Integration of four features': Vegetation_Index + Color_Index + Texture_Feature + Meteorological_Factor,
        '(F) Fusion of Four Features and Coverage': Vegetation_Index + Color_Index + Texture_Feature + Meteorological_Factor + coverage,
        'All Features': all_features
    }
    TABLE_COLUMNS = [
        'Growth_Period_1_R²', 'Growth_Period_1_RMSE',
        'Growth_Period_2_R²', 'Growth_Period_2_RMSE',
        'Growth_Period_3_R²', 'Growth_Period_3_RMSE',
        'Full_Growth_Period_R²', 'Full_Growth_Period_RMSE'
    ]


# ==============================================
# 3. Initialize Output Directories (rf_ prefix, consistent with original PLS directory structure)
# ==============================================
os.makedirs(ExperimentConfig.OUTPUT_ROOT, exist_ok=True)
TABLES_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "rf_data_tables")  # rf_ prefix table directory
PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "rf_plots")  # rf_ prefix plot directory
SINGLE_GROUP_PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "rf_feature_group_plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SINGLE_GROUP_PLOTS_DIR, exist_ok=True)
for group_name in ExperimentConfig.FEATURE_GROUPS.keys():
    if group_name != "All Features":
        group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
        os.makedirs(group_dir, exist_ok=True)

print(f"📂 RF Result Root Directory: {ExperimentConfig.OUTPUT_ROOT}")
print(f"📊 RF Data Table Directory: {TABLES_DIR}")
print(f"🖼️  RF Visualization Directory: {PLOTS_DIR}")
print(f"🖼️  RF Feature Group Visualization Directory: {SINGLE_GROUP_PLOTS_DIR}")


# ==============================================
# 4. Utility Functions (Integrate GPU/CPU Adaptive + Original PLS Utility Functions + New Function to Save True/Predicted CSV)
# ==============================================
# ---------------------- GPU/CPU Data Conversion Utility Functions ----------------------
def to_device(arr):
    """Convert numpy array to cupy array (if GPU is available)"""
    if USE_CUPY and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_numpy(arr):
    """Convert cupy array to numpy array (if GPU is available)"""
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------- Evaluation Metric Utility Functions (Consistent with original PLS) ----------------------
def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE"""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R²"""
    return r2_score(y_true, y_pred)


# ---------------------- Surface Grid Generation Utility Functions (Consistent with original PLS) ----------------------
def create_surface_grid(x_param: np.ndarray, y_param: np.ndarray, z_values: np.ndarray, grid_size: int) -> tuple:
    """Generate smooth surface grid"""
    x_grid = np.linspace(x_param.min(), x_param.max(), grid_size)
    y_grid = np.linspace(y_param.min(), y_param.max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    points = np.vstack((x_param, y_param)).T
    z_mesh = griddata(points, z_values, (x_mesh, y_mesh), method='cubic', fill_value=np.nan)
    return x_mesh, y_mesh, z_mesh


# ---------------------- Baseline Model Utility Functions (RF Exclusive) ----------------------
def get_baseline_rmse(X_train, y_train, X_test, y_test):
    """Get RMSE of baseline model (mean prediction)"""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    return calc_rmse(y_test, y_pred_dummy)


# ---------------------- RF Model Training Utility Functions (GPU/CPU Adaptive) ----------------------
def train_rf_model(n_estimators, mtry, X_train, y_train, X_test, y_test, baseline_rmse):
    """Train RF model and return R² and RMSE"""
    if n_estimators == 0:
        # Use baseline model results
        return 0.0, baseline_rmse

    try:
        if USE_CUML:
            # GPU version of RF
            X_train_dev = to_device(X_train)
            y_train_dev = to_device(y_train)
            X_test_dev = to_device(X_test)

            model = cuRF(
                n_estimators=int(n_estimators),
                max_features=int(mtry),
                random_state=ExperimentConfig.RANDOM_STATE,
                n_streams=4
            )
            model.fit(X_train_dev, y_train_dev)
            y_pred = to_numpy(model.predict(X_test_dev))

        else:
            # CPU version of RF (sklearn)
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=int(n_estimators),
                max_features=int(mtry),
                random_state=ExperimentConfig.RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)
        return round(r2, 4), round(rmse, 4)

    except Exception as e:
        print(f"⚠️ RF Training Error (n_est={n_estimators}, mtry={mtry}): {str(e)[:50]}...")
        return np.nan, np.nan


# ---------------------- Save RF Grid Search Results (rf_ prefix) ----------------------
def save_rf_grid_table(df_grid, save_name):
    """Save RF grid search result table"""
    save_path = os.path.join(TABLES_DIR, f"rf_{save_name}.csv")  # rf_ prefix file name
    df_grid.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"📄 RF Grid Data Table Saved: {save_path} ({len(df_grid)} valid rows in total)")
    return df_grid


# ---------------------- Performance Statistics Table Utility Functions (Consistent with original PLS, rf_ prefix) ----------------------
def create_rf_performance_table(df, pls_params, baseline_rmse):
    """Generate RF performance statistics table (replacing original PLS model)"""
    table_data = []
    index_tuples = []
    target_col_count = len(ExperimentConfig.TABLE_COLUMNS)

    for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
        valid_features = [f for f in group_features if f in df.columns]
        if not valid_features:
            print(f"⚠️ No valid features in feature group {group_name}, skipping")
            continue

        group_metrics = [np.nan] * target_col_count
        for period_idx, (period_name, period_slice) in enumerate(ExperimentConfig.PERIODS.items()):
            period_df = df.iloc[period_slice]
            X = period_df[valid_features].values
            y = period_df[ExperimentConfig.TARGET_COL].values

            # Split training/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ExperimentConfig.TEST_SIZE,
                random_state=ExperimentConfig.RANDOM_STATE, shuffle=True
            )

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train RF model (using optimal parameters)
            r2, rmse = train_rf_model(
                n_estimators=pls_params['best_n_estimators'],
                mtry=pls_params['best_mtry'],
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                baseline_rmse=baseline_rmse
            )

            r2_col_idx = period_idx * 2
            rmse_col_idx = period_idx * 2 + 1
            if r2_col_idx < target_col_count and not np.isnan(r2):
                group_metrics[r2_col_idx] = r2
            if rmse_col_idx < target_col_count and not np.isnan(rmse):
                group_metrics[rmse_col_idx] = rmse

        table_data.append(group_metrics)
        index_tuples.append((group_name, 'RF'))

    rf_table = pd.DataFrame(
        table_data,
        index=pd.MultiIndex.from_tuples(index_tuples, names=['Feature Group', 'Model']),
        columns=ExperimentConfig.TABLE_COLUMNS
    )
    return rf_table


def save_rf_performance_table(rf_table):
    """Save RF performance statistics table (rf_ prefix file name)"""
    excel_path = os.path.join(TABLES_DIR, "rf_Performance_Table_With_R2_RMSE_Header.xlsx")
    csv_path = os.path.join(TABLES_DIR, "rf_Performance_Table_With_R2_RMSE_Header.csv")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        rf_table.to_excel(writer, sheet_name='RF Performance Statistics (Including Full Growth Period)', index=True)
    rf_table.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n📄 RF Performance Statistics Table Saved:")
    print(f"   - Excel Format: {excel_path}")
    print(f"   - CSV Format: {csv_path}")
    print(f"\n📋 RF Performance Statistics Table Preview:")
    print(rf_table)
    return rf_table


# ---------------------- New: Utility Function to Save True/Predicted Data to CSV ----------------------
def save_rf_true_pred_csv(y_true, y_pred, group_labels, save_name,
                          group_r2=None, group_rmse=None,
                          total_r2=None, total_rmse=None):
    """
    Save true and predicted data to CSV file (stored in rf_data_tables directory)
    Parameters:
        y_true: Array of true values
        y_pred: Array of predicted values
        group_labels: Group labels (growth period numbers)
        save_name: CSV file name prefix (no need to add rf_ prefix, automatically supplemented inside the function)
        group_r2: R² of each growth period (optional, stored in CSV as notes)
        group_rmse: RMSE of each growth period (optional, stored in CSV as notes)
        total_r2: Overall R² (optional, stored in CSV as notes)
        total_rmse: Overall RMSE (optional, stored in CSV as notes)
    """
    # Organize true and predicted data into DataFrame
    pred_data = pd.DataFrame({
        'True_Value_' + ExperimentConfig.TARGET_COL: y_true,
        'Predicted_Value_' + ExperimentConfig.TARGET_COL: y_pred,
        'Growth_Period_Label': group_labels,  # 1=Growth Period 1, 2=Growth Period 2, 3=Growth Period 3, 4=Full Growth Period
        'Growth_Period_Name': [
            ExperimentConfig.GROUP_NAMES[int(label) - 1] if label in ExperimentConfig.GROUP_LABELS
            else 'Unknown' for label in group_labels
        ]
    })

    # If evaluation metrics are provided, add them to the top of the DataFrame (as a note row)
    if group_r2 is not None and group_rmse is not None and total_r2 is not None and total_rmse is not None:
        # Construct note row
        note_row = {
            'True_Value_' + ExperimentConfig.TARGET_COL: 'Note:',
            'Predicted_Value_' + ExperimentConfig.TARGET_COL: f'Growth Period 1_R²={group_r2[0]:.4f}, RMSE={group_rmse[0]:.4f}',
            'Growth_Period_Label': f'Growth Period 2_R²={group_r2[1]:.4f}, RMSE={group_rmse[1]:.4f}',
            'Growth_Period_Name': f'Growth Period 3_R²={group_r2[2]:.4f}, RMSE={group_rmse[2]:.4f}; Full Growth Period_R²={total_r2:.4f}, RMSE={total_rmse:.4f}'
        }
        # Insert note row at the first row of DataFrame
        pred_data = pd.concat([pd.DataFrame([note_row]), pred_data], ignore_index=True)

    # Define save path (stored in rf_data_tables directory, named with rf_ prefix)
    save_path = os.path.join(TABLES_DIR, f"rf_{save_name}.csv")
    # Save CSV (support Chinese, ignore index)
    pred_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    valid_sample_count = len(pred_data) - 1 if 'Note:' in pred_data.iloc[0, 0] else len(pred_data)
    print(f"📄 True/Predicted Data CSV Saved: {save_path} ({valid_sample_count} valid samples in total)")
    return save_path


# ---------------------- Data Collection and Visualization Utility Functions (Consistent with original PLS, rf_ prefix) ----------------------
def collect_rf_true_pred_data(df, rf_params, baseline_rmse):
    """Collect RF true and predicted values for All Features"""
    config = ExperimentConfig()
    valid_features = [f for f in all_features if f in df.columns]
    if not valid_features:
        raise ValueError(f"❌ No valid features in all_features, cannot collect true and predicted values")

    all_y_true = []
    all_y_pred = []
    all_group_labels = []
    group_r2_list = []
    group_rmse_list = []
    valid_group_count = 0

    # Iterate over 3 individual growth periods
    for i, (period_name, period_slice) in enumerate([
        ('Growth Period 1', config.PERIODS['Growth Period 1']),
        ('Growth Period 2', config.PERIODS['Growth Period 2']),
        ('Growth Period 3', config.PERIODS['Growth Period 3'])
    ]):
        period_df = df.iloc[period_slice]
        X = period_df[valid_features].values
        y = period_df[config.TARGET_COL].values

        # Split training/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n🔍 Processing {period_name} (All Features-RF): {len(X)} samples, {len(valid_features)} features")

        # Train RF model
        if rf_params['best_n_estimators'] == 0:
            y_pred = np.full_like(y_test, np.mean(y_train))
            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)
        else:
            if USE_CUML:
                X_train_dev = to_device(X_train_scaled)
                y_train_dev = to_device(y_train)
                X_test_dev = to_device(X_test_scaled)
                model = cuRF(
                    n_estimators=int(rf_params['best_n_estimators']),
                    max_features=int(rf_params['best_mtry']),
                    random_state=config.RANDOM_STATE,
                    n_streams=4
                )
                model.fit(X_train_dev, y_train_dev)
                y_pred = to_numpy(model.predict(X_test_dev))
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=int(rf_params['best_n_estimators']),
                    max_features=int(rf_params['best_mtry']),
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"⚠️ No valid true/predicted values for {period_name} (All Features-RF), skipping")
            group_r2_list.append(np.nan)
            group_rmse_list.append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_group_labels.extend([config.GROUP_LABELS[i]] * len(y_test))
        group_r2_list.append(round(r2, 4))
        group_rmse_list.append(round(rmse, 4))
        valid_group_count += 1
        print(f"✅ {period_name} (All Features-RF) Processed: {len(y_test)} valid test samples, R²={r2:.4f}, RMSE={rmse:.4f}")

    # Process full growth period
    full_period_name = "Full Growth Period"
    full_period_df = df.iloc[config.PERIODS['Full Growth Period']]
    X_full = full_period_df[valid_features].values
    y_full = full_period_df[config.TARGET_COL].values

    # Split training/test sets
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, shuffle=True
    )

    # Standardization
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    print(f"\n🔍 Processing {full_period_name} (All Features-RF): {len(X_full)} samples, {len(valid_features)} features")

    # Train RF model
    if rf_params['best_n_estimators'] == 0:
        y_pred_full = np.full_like(y_test_full, np.mean(y_train_full))
        r2_full = calc_r2(y_test_full, y_pred_full)
        rmse_full = calc_rmse(y_test_full, y_pred_full)
    else:
        if USE_CUML:
            X_train_full_dev = to_device(X_train_full_scaled)
            y_train_full_dev = to_device(y_train_full)
            X_test_full_dev = to_device(X_test_full_scaled)
            model = cuRF(
                n_estimators=int(rf_params['best_n_estimators']),
                max_features=int(rf_params['best_mtry']),
                random_state=config.RANDOM_STATE,
                n_streams=4
            )
            model.fit(X_train_full_dev, y_train_full_dev)
            y_pred_full = to_numpy(model.predict(X_test_full_dev))
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=int(rf_params['best_n_estimators']),
                max_features=int(rf_params['best_mtry']),
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            model.fit(X_train_full_scaled, y_train_full)
            y_pred_full = model.predict(X_test_full_scaled)

        r2_full = calc_r2(y_test_full, y_pred_full)
        rmse_full = calc_rmse(y_test_full, y_pred_full)

    if len(y_test_full) == 0 or np.isnan(r2_full):
        print(f"⚠️ No valid true/predicted values for {full_period_name} (All Features-RF), skipping")
        group_r2_list.append(np.nan)
        group_rmse_list.append(np.nan)
    else:
        all_y_true.extend(y_test_full)
        all_y_pred.extend(y_pred_full)
        all_group_labels.extend([config.GROUP_LABELS[3]] * len(y_test_full))
        group_r2_list.append(round(r2_full, 4))
        group_rmse_list.append(round(rmse_full, 4))
        valid_group_count += 1
        print(
            f"✅ {full_period_name} (All Features-RF) Processed: {len(y_test_full)} valid test samples, R²={r2_full:.4f}, RMSE={rmse_full:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_group_labels = np.array(all_group_labels)

    if valid_group_count == 0 or len(all_y_true) == 0 or len(all_y_pred) == 0:
        print(f"\n⚠️ No valid true/predicted values for all groups (including full growth period) (All Features-RF), cannot generate visualization charts")
        return (all_y_true, all_y_pred, all_group_labels,
                group_r2_list, group_rmse_list,
                np.nan, np.nan, np.nan, np.nan)

    overall_a, overall_b = np.polyfit(all_y_true, all_y_pred, 1)
    total_r2 = calc_r2(all_y_true, all_y_pred)
    total_rmse = calc_rmse(all_y_true, all_y_pred)

    print(
        f"\n📊 Combined Statistics for All Valid Groups (Including Full Growth Period) (All Features-RF): {len(all_y_true)} total samples, Combined R²={total_r2:.4f}, Combined RMSE={total_rmse:.4f}")
    return (all_y_true, all_y_pred, all_group_labels,
            group_r2_list, group_rmse_list,
            total_r2, total_rmse, overall_a, overall_b)


def plot_rf_origin_style_results(y_true, y_pred, groups, group_r2_values, group_rmse_values,
                                 total_r2, total_rmse, overall_a, overall_b,
                                 save_name: str = "rf_origin_style_fitting_plot_with_all_periods"):
    """Plot RF Origin-style visualization (rf_ prefix file name)"""
    config = ExperimentConfig()

    if len(y_true) == 0 or len(y_pred) == 0 or np.isnan(total_r2) or np.isnan(total_rmse):
        print(f"⚠️ No valid data for plotting (All Features-RF), skipping Origin-style visualization chart generation")
        return

    fig, ax = plt.subplots(figsize=config.PLOT_FIG_SIZE)

    for i in range(4):
        group_id = config.GROUP_LABELS[i]
        mask = groups == group_id
        if not np.any(mask) or np.isnan(group_r2_values[i]) or np.isnan(group_rmse_values[i]):
            continue

        ax.scatter(
            y_true[mask], y_pred[mask],
            c=config.GROUP_COLORS[i], edgecolors='white', linewidth=0.8,
            s=60, alpha=0.8,
            label=f'{config.GROUP_NAMES[i]} (R²={group_r2_values[i]:.4f}, RMSE={group_rmse_values[i]:.4f})'
        )
        a_group, b_group = np.polyfit(y_true[mask], y_pred[mask], 1)
        x_range = np.linspace(y_true[mask].min(), y_true[mask].max(), 100)
        ax.plot(x_range, a_group * x_range + b_group, color=config.GROUP_COLORS[i], linestyle='--', linewidth=1.5)

    x_total_range = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(
        x_total_range, overall_a * x_total_range + overall_b,
        color='black', linestyle='-', linewidth=2.5,
        label=f'Overall Fitting Line (y={overall_a:.3f}x+{overall_b:.3f}, R²={total_r2:.4f}, RMSE={total_rmse:.4f})'
    )

    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')

    ax.set_xlabel(f'True_Value_{config.TARGET_COL}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Predicted_Value_{config.TARGET_COL}', fontsize=14, fontweight='bold')
    ax.set_title('RF Model True vs Predicted Value Fitting Results (Origin Style + All Features + Full Growth Period)', fontsize=16, fontweight='bold',
                 pad=20)

    ax.legend(
        loc='lower right', frameon=True, framealpha=0.9,
        edgecolor='gray', fontsize=10, labelspacing=0.8
    )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{save_name}.png")  # rf_ prefix file name
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f'\n✅ Origin-style Visualization Chart Including Full Growth Period Saved (All Features-RF): {save_path}')
    plt.show()
    plt.close()


# ---------------------- New: Single Feature Group RF Processing Function ----------------------
def collect_single_feature_group_rf_data(df, group_name, group_features, rf_params, baseline_rmse):
    """Collect RF true and predicted values for a single feature group"""
    config = ExperimentConfig()
    valid_features = [f for f in group_features if f in df.columns]
    if not valid_features:
        print(f"⚠️ No valid features in feature group {group_name}, skipping")
        return None

    group_data = {
        'y_true': [], 'y_pred': [], 'labels': [],
        'period_r2': [], 'period_rmse': [], 'valid_count': 0
    }

    for i, (period_name, period_slice) in enumerate(config.PERIODS.items()):
        period_df = df.iloc[period_slice]
        X = period_df[valid_features].values
        y = period_df[config.TARGET_COL].values

        # Split training/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   🔍 Processing {period_name} ({group_name}-RF): {len(X)} samples, {len(valid_features)} features")

        # Train RF model
        r2, rmse, y_pred = None, None, None
        if rf_params['best_n_estimators'] == 0:
            y_pred = np.full_like(y_test, np.mean(y_train))
            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)
        else:
            if USE_CUML:
                X_train_dev = to_device(X_train_scaled)
                y_train_dev = to_device(y_train)
                X_test_dev = to_device(X_test_scaled)
                model = cuRF(
                    n_estimators=int(rf_params['best_n_estimators']),
                    max_features=int(rf_params['best_mtry']),
                    random_state=config.RANDOM_STATE,
                    n_streams=4
                )
                model.fit(X_train_dev, y_train_dev)
                y_pred = to_numpy(model.predict(X_test_dev))
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=int(rf_params['best_n_estimators']),
                    max_features=int(rf_params['best_mtry']),
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=0
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"   ⚠️ No valid data for {period_name}, skipping")
            group_data['period_r2'].append(np.nan)
            group_data['period_rmse'].append(np.nan)
            continue

        group_data['y_true'].extend(y_test)
        group_data['y_pred'].extend(y_pred)
        group_data['labels'].extend([config.GROUP_LABELS[i]] * len(y_test))
        group_data['period_r2'].append(round(r2, 4))
        group_data['period_rmse'].append(round(rmse, 4))
        group_data['valid_count'] += 1
        print(f"   ✅ {period_name} Completed: R²={r2:.4f}, RMSE={rmse:.4f}")

    group_data['y_true'] = np.array(group_data['y_true'])
    group_data['y_pred'] = np.array(group_data['y_pred'])
    group_data['labels'] = np.array(group_data['labels'])

    if group_data['valid_count'] == 0:
        print(f"   ❌ No valid growth period data for feature group {group_name}, skipping visualization")
        return None

    try:
        overall_a, overall_b = np.polyfit(group_data['y_true'], group_data['y_pred'], 1)
        total_r2 = calc_r2(group_data['y_true'], group_data['y_pred'])
        total_rmse = calc_rmse(group_data['y_true'], group_data['y_pred'])
    except:
        print(f"   ❌ Overall fitting failed for feature group {group_name}, skipping visualization")
        return None

    group_data['overall_a'] = overall_a
    group_data['overall_b'] = overall_b
    group_data['total_r2'] = round(total_r2, 4)
    group_data['total_rmse'] = round(total_rmse, 4)

    return group_data


def plot_single_feature_group_rf_visualization(group_name, group_data):
    """Plot RF Origin-style fitting curve for a single feature group (rf_ prefix)"""
    config = ExperimentConfig()
    group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
    save_name = f"rf_{group_name}_fitting_plot".replace('/', '_').replace(':', '')  # rf_ prefix file name

    fig, ax = plt.subplots(figsize=config.SINGLE_GROUP_PLOT_SIZE)
    color_idx = list(ExperimentConfig.FEATURE_GROUPS.keys()).index(group_name)
    group_color = config.FEATURE_GROUP_PLOT_COLORS[color_idx]

    # Plot scatter points and fitting lines for each growth period
    for i in range(4):
        period_id = config.GROUP_LABELS[i]
        mask = group_data['labels'] == period_id
        if not np.any(mask) or np.isnan(group_data['period_r2'][i]):
            continue

        ax.scatter(
            group_data['y_true'][mask],
            group_data['y_pred'][mask],
            c=config.GROUP_COLORS[i],
            edgecolors='white',
            linewidth=0.8,
            s=60,
            alpha=0.8,
            label=f'{config.GROUP_NAMES[i]} (R²={group_data["period_r2"][i]:.4f}, RMSE={group_data["period_rmse"][i]:.4f})'
        )

        a_period, b_period = np.polyfit(group_data['y_true'][mask], group_data['y_pred'][mask], 1)
        x_range = np.linspace(group_data['y_true'][mask].min(), group_data['y_true'][mask].max(), 100)
        ax.plot(x_range, a_period * x_range + b_period, color=config.GROUP_COLORS[i], linestyle='--', linewidth=1.5)

    # Overall fitting line
    x_total = np.linspace(group_data['y_true'].min(), group_data['y_true'].max(), 100)
    ax.plot(
        x_total,
        group_data['overall_a'] * x_total + group_data['overall_b'],
        color=group_color,
        linestyle='-',
        linewidth=2.5,
        label=f'Overall Fitting Line (y={group_data["overall_a"]:.3f}x+{group_data["overall_b"]:.3f}, R²={group_data["total_r2"]:.4f})'
    )

    # Style settings
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(f'True_Value_{config.TARGET_COL}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted_Value_{config.TARGET_COL}', fontsize=12, fontweight='bold')
    ax.set_title(f'RF Fitting Results - {group_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    # Save
    plt.tight_layout()
    save_path = os.path.join(group_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f"   🖼️  RF Visualization Chart for Feature Group {group_name} Saved: {save_path}")
    plt.close()


# ==============================================
# 5. Data Processing Module (Consistent with original PLS)
# ==============================================
def load_and_preprocess_data():
    """Data Loading and Preprocessing"""
    print(f"\n📥 Loading Data: {ExperimentConfig.DATA_PATH}")
    try:
        df = pd.read_excel(ExperimentConfig.DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data file not found: {ExperimentConfig.DATA_PATH}")

    # Verify features and target variable
    missing_features = [col for col in all_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing the following feature columns in the data: {missing_features}\nPlease check if the feature list is consistent with the data column names!")

    if ExperimentConfig.TARGET_COL not in df.columns:
        raise ValueError(f"Missing target variable column in the data: {ExperimentConfig.TARGET_COL}\nPlease confirm if the target variable column name is correct!")

    # Extract features and target variable
    X = df[all_features].copy()
    y = df[ExperimentConfig.TARGET_COL].copy()
    print(f"Original Data: Feature matrix shape {X.shape}, target variable shape {y.shape}")

    # Missing value handling
    print(f"Feature matrix missing value statistics:\n{X.isnull().sum().sum()} missing values")
    if ExperimentConfig.MISSING_VALUE_HANDLER == 'drop':
        combined = pd.concat([X, y], axis=1)
        combined_clean = combined.dropna(axis=0)
        X = combined_clean[all_features]
        y = combined_clean[ExperimentConfig.TARGET_COL]
        df = combined_clean.reset_index(drop=True)
        print(f"After dropping missing values: Feature matrix shape {X.shape}, target variable shape {y.shape}")
    elif ExperimentConfig.MISSING_VALUE_HANDLER == 'fill':
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        df = pd.concat([X, y], axis=1).reset_index(drop=True)
        print("All missing values have been filled with the mean value")

    # Adaptive setting of MTRY grid
    p = len(all_features)
    if p <= 30:
        ExperimentConfig.MTRY_GRID = list(range(1, p + 1, 1))
    else:
        ExperimentConfig.MTRY_GRID = list(range(1, p + 1, max(2, p // 20)))
    print(
        f"Adaptive RF Parameter Grid: {len(ExperimentConfig.N_ESTIMATORS_GRID)} points for n_estimators, {len(ExperimentConfig.MTRY_GRID)} points for mtry")

    return df, X, y


# ==============================================
# 6. RF Grid Search (Replacing original PLS grid search)
# ==============================================
def run_rf_grid_search(X_scaled, y, baseline_rmse):
    """Execute RF Grid Search"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values,
        test_size=ExperimentConfig.TEST_SIZE,
        random_state=ExperimentConfig.RANDOM_STATE,
        shuffle=True
    )

    total_combinations = len(ExperimentConfig.N_ESTIMATORS_GRID) * len(ExperimentConfig.MTRY_GRID)
    print(f"\n🚀 Starting RF Multi-Parameter Grid Search (Total Combinations: {total_combinations})")

    search_results = []
    for idx, (n_est, mtry) in enumerate(itertools.product(
            ExperimentConfig.N_ESTIMATORS_GRID,
            ExperimentConfig.MTRY_GRID
    ), 1):
        if idx % 50 == 0 or idx == total_combinations:
            print(f"   Progress: {idx}/{total_combinations} (Current: n_est={n_est}, mtry={mtry})")

        # Train RF model
        r2, rm = train_rf_model(n_est, mtry, X_train, y_train, X_test, y_test, baseline_rmse)
        search_results.append({
            'n_estimators': int(n_est),
            'mtry': int(mtry),
            'r2': r2,
            'rmse': rm
        })

    # Save RF grid results (rf_ prefix)
    results_df = pd.DataFrame(search_results)
    full_save_path = os.path.join(ExperimentConfig.OUTPUT_ROOT, "rf_full_search_results.csv")  # rf_ prefix
    results_df.to_csv(full_save_path, index=False, encoding='utf-8-sig')

    # Statistics of valid results
    valid_count = results_df['rmse'].notna().sum()
    print(f"✅ RF Grid Search Completed: {valid_count}/{total_combinations} valid results")
    print(f"📄 RF Grid Results Saved: {full_save_path}")

    # Find optimal parameters
    valid_results = results_df.dropna(subset=['rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['rmse'].idxmin()
        best_params = {
            'best_n_estimators': int(valid_results.loc[best_idx, 'n_estimators']),
            'best_mtry': int(valid_results.loc[best_idx, 'mtry']),
            'best_rmse': valid_results.loc[best_idx, 'rmse'],
            'best_r2': valid_results.loc[best_idx, 'r2']
        }
        print(f"\n🏆 RF Optimal Parameters: n_estimators={best_params['best_n_estimators']}, mtry={best_params['best_mtry']}")
        print(f"   Optimal RMSE: {best_params['best_rmse']:.4f}, Optimal R²: {best_params['best_r2']:.4f}")
    else:
        best_params = {'best_n_estimators': 0, 'best_mtry': 1, 'best_rmse': baseline_rmse, 'best_r2': 0.0}
        print(f"\n⚠️ No valid RF grid results, using baseline parameters")

    return results_df, best_params


# ==============================================
# 7. RF 3D Surface Plotting (Matching example plot, rf_ prefix)
# ==============================================
def plot_rf_3d_surface_match_example(results_df):
    """Plot RF 3D surface plot matching the example plot (rf_ prefix)"""
    # Filter invalid values
    df_valid = results_df.dropna(subset=['rmse'])
    if len(df_valid) < 3:
        print(f"Insufficient valid data, skipping RF 3D surface plot generation")
        return

    # Extract data
    xs = df_valid['n_estimators'].values
    ys = df_valid['mtry'].values
    zs = df_valid['rmse'].values

    # Create 3D figure
    fig = plt.figure(figsize=ExperimentConfig.FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    # Generate regular grid
    xi = np.linspace(xs.min(), xs.max(), 100)
    yi = np.linspace(ys.min(), ys.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate to calculate z values
    zi = griddata((xs, ys), zs, (xi, yi), method='cubic', fill_value=np.nan)

    # Plot 3D surface (matching example plot style)
    surf = ax.plot_surface(
        xi, yi, zi,
        cmap=ExperimentConfig.SURFACE_CMAP,
        alpha=0.9,
        linewidth=0.5,
        edgecolor='k',
        antialiased=True
    )

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='RMSE')
    cbar.ax.tick_params(labelsize=10)

    # Match example plot viewing angle
    ax.view_init(elev=30, azim=135)
    ax.grid(True, alpha=0.3)

    # Chart style
    ax.set_xlabel('n_estimators', fontsize=12, labelpad=15)
    ax.set_ylabel('mtry', fontsize=12, labelpad=15)
    ax.set_zlabel('RMSE', fontsize=12, labelpad=15)
    ax.set_title('RF Performance Surface: n_estimators vs mtry → RMSE', fontsize=14, pad=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Save image (rf_ prefix)
    save_path = os.path.join(PLOTS_DIR, "rf_3d_rmse_match_example.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=ExperimentConfig.FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ RF Example-Matched 3D Surface Plot Saved: {save_path}")


# ==============================================
# 8. Main Function (Consistent with original PLS process, RF replacement + new function to save true/predicted CSV)
# ==============================================
def main():
    print("=" * 70)
    print("📌 RF Regression Complete Toolkit (GPU/CPU Adaptive, file names with rf_ prefix)")
    print("=" * 70)

    try:
        # Step 1: Data Preprocessing
        df, X, y = load_and_preprocess_data()

        # Step 2: Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Standardized feature matrix shape: {X_scaled.shape}")

        # Step 3: Baseline Model RMSE
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values,
            test_size=ExperimentConfig.TEST_SIZE,
            random_state=ExperimentConfig.RANDOM_STATE,
            shuffle=True
        )
        baseline_rmse = get_baseline_rmse(X_train, y_train, X_test, y_test)
        print(f"\nBaseline Model (Mean Prediction) RMSE: {baseline_rmse:.4f}")

        # Step 4: RF Grid Search
        rf_results_df, rf_best_params = run_rf_grid_search(X_scaled, y, baseline_rmse)

        # Step 5: Plot RF 3D Surface Plot (rf_ prefix)
        print(f"\n🎨 Starting to Plot RF 3D Performance Surface Plot (Matching Example Plot)")
        plot_rf_3d_surface_match_example(rf_results_df)

        # Step 6: Save RF Grid Data Table (rf_ prefix)
        save_rf_grid_table(rf_results_df, "grid_results")

        # Step 7: Generate RF Performance Statistics Table (rf_ prefix)
        print(f"\n📋 Starting to Generate RF Performance Statistics Table (Including Full Growth Period)")
        rf_performance_table = create_rf_performance_table(df, rf_best_params, baseline_rmse)
        save_rf_performance_table(rf_performance_table)

        # Step 8: All Features-RF Visualization + Save True/Predicted CSV
        print(f"\n🎨 Starting to Process Origin-style Visualization for All Features-RF")
        (y_true, y_pred, group_labels,
         group_r2, group_rmse,
         total_r2, total_rmse, overall_a, overall_b) = collect_rf_true_pred_data(df, rf_best_params, baseline_rmse)

        # New: Save All Features true/predicted data CSV
        if len(y_true) > 0 and len(y_pred) > 0 and not np.isnan(total_r2):
            # Save true/predicted CSV
            save_rf_true_pred_csv(
                y_true=y_true,
                y_pred=y_pred,
                group_labels=group_labels,
                save_name="all_features_true_pred_data",
                group_r2=group_r2,
                group_rmse=group_rmse,
                total_r2=total_r2,
                total_rmse=total_rmse
            )
            # Plot visualization chart
            plot_rf_origin_style_results(
                y_true=y_true,
                y_pred=y_pred,
                groups=group_labels,
                group_r2_values=group_r2,
                group_rmse_values=group_rmse,
                total_r2=total_r2,
                total_rmse=total_rmse,
                overall_a=overall_a,
                overall_b=overall_b
            )
        else:
            print(f"⚠️ No valid data for All Features-RF, skipping visualization and CSV saving")

        # Step 9: 6 Major Feature Groups-RF Independent Visualization + Save Each Feature Group's True/Predicted CSV
        print(f"\n🎨 Starting to Process Independent Fitting Visualization for 6 Major Feature Groups-RF")
        # For summarizing true/predicted data of all feature groups (optional)
        all_group_pred_data = []

        for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
            if group_name == "All Features":
                continue
            print(f"\n{'=' * 50}")
            print(f"Processing Feature Group: {group_name} (RF)")
            print(f"{'=' * 50}")
            group_data = collect_single_feature_group_rf_data(df, group_name, group_features, rf_best_params,
                                                              baseline_rmse)

            if group_data is not None:
                # New: Save single feature group true/predicted data CSV
                csv_save_name = f"{group_name.replace(' ', '_').replace('(', '').replace(')', '')}_true_pred_data"
                save_rf_true_pred_csv(
                    y_true=group_data['y_true'],
                    y_pred=group_data['y_pred'],
                    group_labels=group_data['labels'],
                    save_name=csv_save_name,
                    group_r2=group_data['period_r2'],
                    group_rmse=group_data['period_rmse'],
                    total_r2=group_data['total_r2'],
                    total_rmse=group_data['total_rmse']
                )
                # Plot visualization chart
                plot_single_feature_group_rf_visualization(group_name, group_data)

                # (Optional) Summarize into total data
                group_pred_df = pd.DataFrame({
                    'Feature_Group': group_name,
                    'True_Value_' + ExperimentConfig.TARGET_COL: group_data['y_true'],
                    'Predicted_Value_' + ExperimentConfig.TARGET_COL: group_data['y_pred'],
                    'Growth_Period_Label': group_data['labels'],
                    'Growth_Period_Name': [
                        ExperimentConfig.GROUP_NAMES[int(label) - 1] if label in ExperimentConfig.GROUP_LABELS
                        else 'Unknown' for label in group_data['labels']
                    ]
                })
                all_group_pred_data.append(group_pred_df)

        # (Optional) Save summary CSV of true/predicted data for all feature groups
        if all_group_pred_data:
            total_group_pred_df = pd.concat(all_group_pred_data, ignore_index=True)
            total_save_path = os.path.join(TABLES_DIR, "rf_all_feature_groups_true_pred_summary.csv")
            total_group_pred_df.to_csv(total_save_path, index=False, encoding='utf-8-sig')
            print(f"\n📄 Summary CSV of True/Predicted Data for All Feature Groups Saved: {total_save_path}")

        # Step 10: Output File List
        print(f"\n🎉 All RF Processes Executed Successfully!")
        print(f"\n【RF Output File List】")
        print(f"### Core Output Files (rf_ Prefix) ###")
        print(f"1. RF Grid Search Results: {os.path.join(ExperimentConfig.OUTPUT_ROOT, 'rf_full_search_results.csv')}")
        print(f"2. RF 3D Surface Plot: {os.path.join(PLOTS_DIR, 'rf_3d_rmse_match_example.png')}")
        print(f"3. RF Grid Data Table: {os.path.join(TABLES_DIR, 'rf_grid_results.csv')}")
        print(f"4. RF Performance Statistics Table (Excel): {os.path.join(TABLES_DIR, 'rf_Performance_Table_With_R2_RMSE_Header.xlsx')}")
        print(f"5. RF Performance Statistics Table (CSV): {os.path.join(TABLES_DIR, 'rf_Performance_Table_With_R2_RMSE_Header.csv')}")
        print(f"6. RF All Features True/Predicted Data: {os.path.join(TABLES_DIR, 'rf_all_features_true_pred_data.csv')}")
        print(f"7. RF True/Predicted Data for Each Feature Group: {TABLES_DIR} (File name format: rf_*_true_pred_data.csv)")
        print(
            f"8. RF All Features Visualization: {os.path.join(PLOTS_DIR, 'rf_origin_style_fitting_plot_with_all_periods.png')}")
        print(f"9. RF Feature Group Visualization: {SINGLE_GROUP_PLOTS_DIR}")
        if all_group_pred_data:
            print(
                f"10. RF True/Predicted Summary for All Feature Groups: {os.path.join(TABLES_DIR, 'rf_all_feature_groups_true_pred_summary.csv')}")

    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()