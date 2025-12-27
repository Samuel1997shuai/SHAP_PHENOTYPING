# -*- coding: utf-8 -*-
"""
Support Vector Regression (SVR) Complete Toolkit (GPU/CPU Adaptive)
Functions are fully consistent with the RF (Random Forest) version, and all file names are unified as svr_xxxxx
1. SVR Parameter Grid Search (log10(C) + log10(gamma), including exponential (e) conversion)
2. 3D Performance Surface Plot (log10(C) vs log10(gamma) → RMSE) + Corresponding Data Table
3. R²/RMSE Performance Statistics Table (including Full Growth Period) in both Excel and CSV formats
4. Origin-style Visualization (Fitting Curves for Each Growth Period + Full Growth Period)
5. Independent Fitting Visualization for 6 Major Feature Groups (New Function, Maintaining Compatibility)
6. Saving of True/Predicted Values (CSV format, stored in the svr_data_tables directory)
"""

# ==============================================
# 1. Dependency Library Import (Integrated GPU/CPU Adaptation + Original Functional Libraries)
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
from sklearn.svm import SVR

# GPU / CuPy Adaptive Configuration (Consistent with RF Settings)
GPU_AVAILABLE = False
USE_CUPY = False
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

# Feature Definition Import (Consistent with RF)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)
try:
    from config.feature_definitions import (
        Vegetation_Index,
        Color_Index,
        Texture_Feature,
        Meteorological_Factor,
        all_features,
        coverage
    )
except ImportError:
    # If feature configuration file does not exist, manually define sample features (ensure code can run independently)
    print("⚠️ Feature configuration file not found, using sample feature list")
    Vegetation_Index = ['NDVI', 'EVI', 'GNDVI']
    Color_Index = ['R/G', 'G/B', 'R/B']
    Texture_Feature = ['mean', 'var', 'entropy']
    Meteorological_Factor = ['temp', 'rainfall', 'humidity']
    coverage = ['coverage']
    all_features = Vegetation_Index + Color_Index + Texture_Feature + Meteorological_Factor + coverage

# Environment Configuration (Consistent with RF)
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Support English display (replace with SimHei for Chinese if needed)
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
print(f"GPU_AVAILABLE={GPU_AVAILABLE}, USE_CUPY={USE_CUPY}")


# ==============================================
# 2. Experiment Parameter Configuration (All file names start with svr_, consistent with RF functions)
# ==============================================
class ExperimentConfig:
    """SVR Experiment Parameter Configuration Class (Consistent with RF functions, prefix with svr_)"""
    # Data-related Settings (Consistent with RF)
    DATA_PATH = '../resource/data_all.xlsx'
    TARGET_COL = 'LAI'
    MISSING_VALUE_HANDLER = 'drop'

    # Dataset Splitting (Consistent with RF)
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_TEST_SAMPLES = 5

    # Output Configuration (All files/directories start with svr_, maintaining original structure)
    OUTPUT_ROOT = './svr_3d_surfaces_with_tables'  # Directory with svr_ prefix
    FIG_DPI = 300
    FIG_SIZE = (14, 10)
    SURFACE_CMAP = 'jet'  # Match RF example figure colormap
    PLOT_FIG_SIZE = (14, 10)
    SINGLE_GROUP_PLOT_SIZE = (12, 9)

    # SVR Core Parameter Grid (Replace RF's n_estimators+mtry, retain original SVR log10 scale)
    LOG10_C_GRID = np.linspace(-2, 3, num=20)  # log10(C) range: -2 ~ 3
    LOG10_GAMMA_GRID = np.linspace(-4, 0, num=20)  # log10(gamma) range: -4 ~ 0
    SVR_KERNEL = 'rbf'  # SVR kernel function
    SVR_MAX_ITER = 10000  # SVR maximum number of iterations
    INTERP_GRID_SIZE = 50
    Z_AXIS_LIMIT = None

    # Plotting Configuration (Consistent with RF)
    GROUP_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    GROUP_NAMES = ['Growth Period 1', 'Growth Period 2', 'Growth Period 3', 'Full Growth Period']
    GROUP_LABELS = [1, 2, 3, 4]
    FEATURE_GROUP_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Growth Period and Feature Group Configuration (Fully consistent with RF)
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
# 3. Initialize Output Directories (svr_ prefix, consistent with RF directory structure)
# ==============================================
os.makedirs(ExperimentConfig.OUTPUT_ROOT, exist_ok=True)
TABLES_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "svr_data_tables")  # Table directory with svr_ prefix
PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "svr_plots")  # Plot directory with svr_ prefix
SINGLE_GROUP_PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "svr_feature_group_plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SINGLE_GROUP_PLOTS_DIR, exist_ok=True)
for group_name in ExperimentConfig.FEATURE_GROUPS.keys():
    if group_name != "All Features":
        group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
        os.makedirs(group_dir, exist_ok=True)

print(f"📂 SVR Result Root Directory: {ExperimentConfig.OUTPUT_ROOT}")
print(f"📊 SVR Data Table Directory: {TABLES_DIR}")
print(f"🖼️  SVR Visualization Directory: {PLOTS_DIR}")
print(f"🖼️  SVR Feature Group Visualization Directory: {SINGLE_GROUP_PLOTS_DIR}")


# ==============================================
# 4. Utility Functions (Integrated GPU/CPU Adaptation + RF Utility Functions + SVR Exclusive Functions)
# ==============================================
# ---------------------- GPU/CPU Data Conversion Utility Functions (Consistent with RF) ----------------------
def to_device(arr):
    """Convert a NumPy array to a CuPy array (if GPU is available)"""
    if USE_CUPY and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_numpy(arr):
    """Convert a CuPy array to a NumPy array (if GPU is available)"""
    if USE_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------- SVR Exclusive Utility Functions (Retain original SVR core functions) ----------------------
def log10_to_exp(x: float) -> tuple[float, float]:
    """
    Convert log10-scaled parameters to exponential (e) form
    Formula: x = log10(val) → val = 10^x = e^(x * ln10)
    Returns: (Original value val, Exponential exponent exp_x)
    """
    ln10 = math.log(10)
    exp_x = x * ln10
    val = math.exp(exp_x)
    return val, exp_x


# ---------------------- Evaluation Metric Utility Functions (Consistent with RF) ----------------------
def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)"""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R²)"""
    return r2_score(y_true, y_pred)


# ---------------------- Surface Grid Generation Utility Functions (Consistent with RF) ----------------------
def create_surface_grid(x_param: np.ndarray, y_param: np.ndarray, z_values: np.ndarray, grid_size: int) -> tuple:
    """Generate smooth surface grid"""
    x_grid = np.linspace(x_param.min(), x_param.max(), grid_size)
    y_grid = np.linspace(y_param.min(), y_param.max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    points = np.vstack((x_param, y_param)).T
    z_mesh = griddata(points, z_values, (x_mesh, y_mesh), method='cubic', fill_value=np.nan)
    return x_mesh, y_mesh, z_mesh


# ---------------------- Baseline Model Utility Functions (Consistent with RF) ----------------------
def get_baseline_rmse(X_train, y_train, X_test, y_test):
    """Get RMSE of baseline model (mean prediction)"""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    return calc_rmse(y_test, y_pred_dummy)


# ---------------------- SVR Model Training Utility Functions (GPU/CPU Adaptive, replace RF training function) ----------------------
def train_svr_model(log10_C, log10_gamma, X_train, y_train, X_test, y_test, baseline_rmse):
    """Train SVR model and return R² and RMSE (compatible with baseline model marking)"""
    # If log10_C is a special value (simulate RF's n_estimators=0), use baseline model
    if log10_C == -999:
        return 0.0, baseline_rmse

    try:
        # Convert parameters (log10 → original value)
        C = 10 ** log10_C
        gamma = 10 ** log10_gamma

        # SVR only supports CPU computation, ensure data is NumPy array
        X_train_np = to_numpy(X_train)
        y_train_np = to_numpy(y_train)
        X_test_np = to_numpy(X_test)

        # Train SVR model
        model = SVR(
            C=float(C),
            gamma=float(gamma),
            kernel=ExperimentConfig.SVR_KERNEL,
            max_iter=ExperimentConfig.SVR_MAX_ITER,
            verbose=False
        )
        model.fit(X_train_np, y_train_np)
        y_pred = model.predict(X_test_np)

        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)
        return round(r2, 4), round(rmse, 4)

    except Exception as e:
        print(f"⚠️ SVR Training Error (log10_C={log10_C:.2f}, log10_gamma={log10_gamma:.2f}): {str(e)[:50]}...")
        return np.nan, np.nan


# ---------------------- Save SVR Grid Search Results (svr_ prefix, replace RF grid save function) ----------------------
def save_svr_grid_table(df_grid, save_name):
    """Save SVR grid search result table"""
    save_path = os.path.join(TABLES_DIR, f"svr_{save_name}.csv")  # File name with svr_ prefix
    df_grid.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"📄 SVR Grid Data Table Saved: {save_path} ({len(df_grid)} valid rows in total)")
    return df_grid


# ---------------------- Performance Statistics Table Utility Functions (Consistent with RF, svr_ prefix) ----------------------
def create_svr_performance_table(df, svr_params, baseline_rmse):
    """Generate SVR performance statistics table (replace RF model)"""
    table_data = []
    index_tuples = []
    target_col_count = len(ExperimentConfig.TABLE_COLUMNS)

    for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
        valid_features = [f for f in group_features if f in df.columns]
        if not valid_features:
            print(f"⚠️ Feature group {group_name} has no valid features, skipped")
            continue

        group_metrics = [np.nan] * target_col_count
        for period_idx, (period_name, period_slice) in enumerate(ExperimentConfig.PERIODS.items()):
            period_df = df.iloc[period_slice]
            X = period_df[valid_features].values
            y = period_df[ExperimentConfig.TARGET_COL].values

            # Split training/test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ExperimentConfig.TEST_SIZE,
                random_state=ExperimentConfig.RANDOM_STATE, shuffle=True
            )

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train SVR model (using optimal parameters)
            r2, rmse = train_svr_model(
                log10_C=svr_params['best_log10_C'],
                log10_gamma=svr_params['best_log10_gamma'],
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
        index_tuples.append((group_name, 'SVR'))

    svr_table = pd.DataFrame(
        table_data,
        index=pd.MultiIndex.from_tuples(index_tuples, names=['Feature Group', 'Model']),
        columns=ExperimentConfig.TABLE_COLUMNS
    )
    return svr_table


def save_svr_performance_table(svr_table):
    """Save SVR performance statistics table (file name with svr_ prefix)"""
    excel_path = os.path.join(TABLES_DIR, "svr_Performance_Table_With_R2_RMSE_Header.xlsx")
    csv_path = os.path.join(TABLES_DIR, "svr_Performance_Table_With_R2_RMSE_Header.csv")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        svr_table.to_excel(writer, sheet_name='SVR Performance Statistics (Including Full Growth Period)', index=True)
    svr_table.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n📄 SVR Performance Statistics Table Saved:")
    print(f"   - Excel Format: {excel_path}")
    print(f"   - CSV Format: {csv_path}")
    print(f"\n📋 SVR Performance Statistics Table Preview:")
    print(svr_table)
    return svr_table


# ---------------------- Save True/Predicted Data to CSV Utility Function (svr_ prefix, consistent with RF) ----------------------
def save_svr_true_pred_csv(y_true, y_pred, group_labels, save_name,
                          group_r2=None, group_rmse=None,
                          total_r2=None, total_rmse=None):
    """
    Save true and predicted values to CSV file (stored in svr_data_tables directory)
    Parameters:
        y_true: Array of true values
        y_pred: Array of predicted values
        group_labels: Group labels (growth period numbers)
        save_name: CSV file name prefix (no need to add svr_ prefix, automatically supplemented inside the function)
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

    # If evaluation metrics are provided, add them to the top of the DataFrame (as note rows)
    if group_r2 is not None and group_rmse is not None and total_r2 is not None and total_rmse is not None:
        # Construct note row
        note_row = {
            'True_Value_' + ExperimentConfig.TARGET_COL: 'Notes:',
            'Predicted_Value_' + ExperimentConfig.TARGET_COL: f'Growth Period 1_R²={group_r2[0]:.4f}, RMSE={group_rmse[0]:.4f}',
            'Growth_Period_Label': f'Growth Period 2_R²={group_r2[1]:.4f}, RMSE={group_rmse[1]:.4f}',
            'Growth_Period_Name': f'Growth Period 3_R²={group_r2[2]:.4f}, RMSE={group_rmse[2]:.4f}; Full Growth Period_R²={total_r2:.4f}, RMSE={total_rmse:.4f}'
        }
        # Insert note row at the first row of DataFrame
        pred_data = pd.concat([pd.DataFrame([note_row]), pred_data], ignore_index=True)

    # Define save path (stored in svr_data_tables directory, named with svr_ prefix)
    save_path = os.path.join(TABLES_DIR, f"svr_{save_name}.csv")
    # Save CSV (support Chinese, ignore index)
    pred_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    valid_sample_count = len(pred_data) - 1 if 'Notes:' in pred_data.iloc[0, 0] else len(pred_data)
    print(f"📄 True/Predicted Data CSV Saved: {save_path} ({valid_sample_count} valid samples in total)")
    return save_path


# ---------------------- Data Collection and Visualization Utility Functions (Consistent with RF, adapted for SVR) ----------------------
def collect_svr_true_pred_data(df, svr_params, baseline_rmse):
    """Collect SVR true and predicted values for All Features"""
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

        # Split training/test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n🔍 Processing {period_name} (All Features-SVR): {len(X)} samples, {len(valid_features)} features")

        # Train SVR model
        if svr_params['best_log10_C'] == -999:
            y_pred = np.full_like(y_test, np.mean(y_train))
            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)
        else:
            # SVR only supports CPU computation
            X_train_np = to_numpy(X_train_scaled)
            y_train_np = to_numpy(y_train)
            X_test_np = to_numpy(X_test_scaled)

            # Convert optimal parameters
            best_C = 10 ** svr_params['best_log10_C']
            best_gamma = 10 ** svr_params['best_log10_gamma']

            model = SVR(
                C=float(best_C),
                gamma=float(best_gamma),
                kernel=config.SVR_KERNEL,
                max_iter=config.SVR_MAX_ITER,
                verbose=False
            )
            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_test_np)

            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"⚠️ {period_name} (All Features-SVR) has no valid true/predicted values, skipped")
            group_r2_list.append(np.nan)
            group_rmse_list.append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_group_labels.extend([config.GROUP_LABELS[i]] * len(y_test))
        group_r2_list.append(round(r2, 4))
        group_rmse_list.append(round(rmse, 4))
        valid_group_count += 1
        print(f"✅ {period_name} (All Features-SVR) Processed: {len(y_test)} valid test samples, R²={r2:.4f}, RMSE={rmse:.4f}")

    # Process full growth period
    full_period_name = "Full Growth Period"
    full_period_df = df.iloc[config.PERIODS['Full Growth Period']]
    X_full = full_period_df[valid_features].values
    y_full = full_period_df[config.TARGET_COL].values

    # Split training/test set
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, shuffle=True
    )

    # Standardization
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    print(f"\n🔍 Processing {full_period_name} (All Features-SVR): {len(X_full)} samples, {len(valid_features)} features")

    # Train SVR model
    if svr_params['best_log10_C'] == -999:
        y_pred_full = np.full_like(y_test_full, np.mean(y_train_full))
        r2_full = calc_r2(y_test_full, y_pred_full)
        rmse_full = calc_rmse(y_test_full, y_pred_full)
    else:
        # SVR only supports CPU computation
        X_train_full_np = to_numpy(X_train_full_scaled)
        y_train_full_np = to_numpy(y_train_full)
        X_test_full_np = to_numpy(X_test_full_scaled)

        # Convert optimal parameters
        best_C = 10 ** svr_params['best_log10_C']
        best_gamma = 10 ** svr_params['best_log10_gamma']

        model = SVR(
            C=float(best_C),
            gamma=float(best_gamma),
            kernel=config.SVR_KERNEL,
            max_iter=config.SVR_MAX_ITER,
            verbose=False
        )
        model.fit(X_train_full_np, y_train_full_np)
        y_pred_full = model.predict(X_test_full_np)

        r2_full = calc_r2(y_test_full, y_pred_full)
        rmse_full = calc_rmse(y_test_full, y_pred_full)

    if len(y_test_full) == 0 or np.isnan(r2_full):
        print(f"⚠️ {full_period_name} (All Features-SVR) has no valid true/predicted values, skipped")
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
            f"✅ {full_period_name} (All Features-SVR) Processed: {len(y_test_full)} valid test samples, R²={r2_full:.4f}, RMSE={rmse_full:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_group_labels = np.array(all_group_labels)

    if valid_group_count == 0 or len(all_y_true) == 0 or len(all_y_pred) == 0:
        print(f"\n⚠️ All groups (including Full Growth Period) have no valid true/predicted values (All Features-SVR), cannot generate visualization charts")
        return (all_y_true, all_y_pred, all_group_labels,
                group_r2_list, group_rmse_list,
                np.nan, np.nan, np.nan, np.nan)

    overall_a, overall_b = np.polyfit(all_y_true, all_y_pred, 1)
    total_r2 = calc_r2(all_y_true, all_y_pred)
    total_rmse = calc_rmse(all_y_true, all_y_pred)

    print(
        f"\n📊 Combined Statistics for All Valid Groups (Including Full Growth Period) (All Features-SVR): {len(all_y_true)} total samples, Combined R²={total_r2:.4f}, Combined RMSE={total_rmse:.4f}")
    return (all_y_true, all_y_pred, all_group_labels,
            group_r2_list, group_rmse_list,
            total_r2, total_rmse, overall_a, overall_b)


def plot_svr_origin_style_results(y_true, y_pred, groups, group_r2_values, group_rmse_values,
                                 total_r2, total_rmse, overall_a, overall_b,
                                 save_name: str = "svr_origin_style_fitting_plot_with_all_periods"):
    """Plot SVR Origin-style visualization (file name with svr_ prefix)"""
    config = ExperimentConfig()

    if len(y_true) == 0 or len(y_pred) == 0 or np.isnan(total_r2) or np.isnan(total_rmse):
        print(f"⚠️ No valid data for plotting (All Features-SVR), skipped Origin-style visualization chart generation")
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

    ax.set_xlabel(f'True Value_{config.TARGET_COL}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Predicted Value_{config.TARGET_COL}', fontsize=14, fontweight='bold')
    ax.set_title('SVR Model True vs. Predicted Values Fitting Results (Origin-style + All Features + Full Growth Period)', fontsize=16, fontweight='bold',
                 pad=20)

    ax.legend(
        loc='lower right', frameon=True, framealpha=0.9,
        edgecolor='gray', fontsize=10, labelspacing=0.8
    )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{save_name}.png")  # File name with svr_ prefix
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f'\n✅ Origin-style Visualization Chart (Including Full Growth Period) Saved (All Features-SVR): {save_path}')
    plt.show()
    plt.close()


# ---------------------- Single Feature Group SVR Processing Function (Consistent with RF, adapted for SVR) ----------------------
def collect_single_feature_group_svr_data(df, group_name, group_features, svr_params, baseline_rmse):
    """Collect SVR true and predicted values for a single feature group"""
    config = ExperimentConfig()
    valid_features = [f for f in group_features if f in df.columns]
    if not valid_features:
        print(f"⚠️ Feature group {group_name} has no valid features, skipped")
        return None

    group_data = {
        'y_true': [], 'y_pred': [], 'labels': [],
        'period_r2': [], 'period_rmse': [], 'valid_count': 0
    }

    for i, (period_name, period_slice) in enumerate(config.PERIODS.items()):
        period_df = df.iloc[period_slice]
        X = period_df[valid_features].values
        y = period_df[config.TARGET_COL].values

        # Split training/test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   🔍 Processing {period_name} ({group_name}-SVR): {len(X)} samples, {len(valid_features)} features")

        # Train SVR model
        r2, rmse, y_pred = None, None, None
        if svr_params['best_log10_C'] == -999:
            y_pred = np.full_like(y_test, np.mean(y_train))
            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)
        else:
            # SVR only supports CPU computation
            X_train_np = to_numpy(X_train_scaled)
            y_train_np = to_numpy(y_train)
            X_test_np = to_numpy(X_test_scaled)

            # Convert optimal parameters
            best_C = 10 ** svr_params['best_log10_C']
            best_gamma = 10 ** svr_params['best_log10_gamma']

            model = SVR(
                C=float(best_C),
                gamma=float(best_gamma),
                kernel=config.SVR_KERNEL,
                max_iter=config.SVR_MAX_ITER,
                verbose=False
            )
            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_test_np)

            r2 = calc_r2(y_test, y_pred)
            rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"   ⚠️ {period_name} has no valid data, skipped")
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
        print(f"   ❌ Feature group {group_name} has no valid growth period data, skipped visualization")
        return None

    try:
        overall_a, overall_b = np.polyfit(group_data['y_true'], group_data['y_pred'], 1)
        total_r2 = calc_r2(group_data['y_true'], group_data['y_pred'])
        total_rmse = calc_rmse(group_data['y_true'], group_data['y_pred'])
    except:
        print(f"   ❌ Overall fitting failed for feature group {group_name}, skipped visualization")
        return None

    group_data['overall_a'] = overall_a
    group_data['overall_b'] = overall_b
    group_data['total_r2'] = round(total_r2, 4)
    group_data['total_rmse'] = round(total_rmse, 4)

    return group_data


def plot_single_feature_group_svr_visualization(group_name, group_data):
    """Plot Origin-style fitting curve for a single feature group's SVR results (svr_ prefix)"""
    config = ExperimentConfig()
    group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
    save_name = f"svr_{group_name}_fitting_plot".replace('/', '_').replace(':', '')  # File name with svr_ prefix

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
    ax.set_xlabel(f'True Value_{config.TARGET_COL}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted Value_{config.TARGET_COL}', fontsize=12, fontweight='bold')
    ax.set_title(f'SVR Fitting Results - {group_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    # Save
    plt.tight_layout()
    save_path = os.path.join(group_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f"   🖼️  SVR Visualization Chart for Feature Group {group_name} Saved: {save_path}")
    plt.close()


# ==============================================
# 5. Data Processing Module (Consistent with RF)
# ==============================================
def load_and_preprocess_data():
    """Data loading and preprocessing"""
    print(f"\n📥 Loading Data: {ExperimentConfig.DATA_PATH}")
    try:
        df = pd.read_excel(ExperimentConfig.DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data file not found: {ExperimentConfig.DATA_PATH}")

    # Verify features and target variable
    missing_features = [col for col in all_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing the following feature columns in data: {missing_features}\nPlease check if feature list matches data column names!")

    if ExperimentConfig.TARGET_COL not in df.columns:
        raise ValueError(f"Missing target variable column in data: {ExperimentConfig.TARGET_COL}\nPlease confirm target variable column name is correct!")

    # Extract features and target variable
    X = df[all_features].copy()
    y = df[ExperimentConfig.TARGET_COL].copy()
    print(f"Original Data: Feature matrix shape {X.shape}, Target variable shape {y.shape}")

    # Missing value handling
    print(f"Feature matrix missing value statistics:\n{X.isnull().sum().sum()} missing values")
    if ExperimentConfig.MISSING_VALUE_HANDLER == 'drop':
        combined = pd.concat([X, y], axis=1)
        combined_clean = combined.dropna(axis=0)
        X = combined_clean[all_features]
        y = combined_clean[ExperimentConfig.TARGET_COL]
        df = combined_clean.reset_index(drop=True)
        print(f"After dropping missing values: Feature matrix shape {X.shape}, Target variable shape {y.shape}")
    elif ExperimentConfig.MISSING_VALUE_HANDLER == 'fill':
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        df = pd.concat([X, y], axis=1).reset_index(drop=True)
        print("Filled all missing values with mean")

    # Print SVR parameter grid information
    print(
        f"Adaptive SVR parameter grid: {len(ExperimentConfig.LOG10_C_GRID)} points for log10_C, {len(ExperimentConfig.LOG10_GAMMA_GRID)} points for log10_gamma")

    return df, X, y


# ==============================================
# 6. SVR Grid Search (Replace RF grid search, retain original SVR core logic)
# ==============================================
def run_svr_grid_search(X_scaled, y, baseline_rmse):
    """Execute SVR grid search"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values,
        test_size=ExperimentConfig.TEST_SIZE,
        random_state=ExperimentConfig.RANDOM_STATE,
        shuffle=True
    )

    # Convert to NumPy array (SVR only supports CPU)
    X_train_np = to_numpy(X_train)
    X_test_np = to_numpy(X_test)
    y_train_np = to_numpy(y_train)
    y_test_np = to_numpy(y_test)

    total_combinations = len(ExperimentConfig.LOG10_C_GRID) * len(ExperimentConfig.LOG10_GAMMA_GRID)
    print(f"\n🚀 Starting SVR Multi-parameter Grid Search (Total combinations: {total_combinations})")

    search_results = []
    for idx, (log10_C, log10_gamma) in enumerate(itertools.product(
            ExperimentConfig.LOG10_C_GRID,
            ExperimentConfig.LOG10_GAMMA_GRID
    ), 1):
        if idx % 20 == 0 or idx == total_combinations:
            print(f"   Progress: {idx}/{total_combinations} (Current: log10_C={log10_C:.2f}, log10_gamma={log10_gamma:.2f})")

        # Train SVR model
        try:
            # Convert parameters (log10 → original value → exponential form)
            C = 10 ** log10_C
            gamma = 10 ** log10_gamma
            _, exp_C = log10_to_exp(log10_C)
            _, exp_gamma = log10_to_exp(log10_gamma)

            # Train SVR model
            svr_model = SVR(
                C=float(C),
                gamma=float(gamma),
                kernel=ExperimentConfig.SVR_KERNEL,
                max_iter=ExperimentConfig.SVR_MAX_ITER,
                verbose=False
            )
            svr_model.fit(X_train_np, y_train_np)

            # Predict and calculate metrics
            y_pred = svr_model.predict(X_test_np)
            r2 = calc_r2(y_test_np, y_pred)
            rmse = calc_rmse(y_test_np, y_pred)

            search_results.append({
                'log10_C': log10_C,
                'log10_gamma': log10_gamma,
                'C': C,
                'gamma': gamma,
                'exp_C': exp_C,
                'exp_gamma': exp_gamma,
                'r2': round(r2, 4),
                'rmse': round(rmse, 4)
            })
        except Exception as e:
            print(f"⚠️ SVR Training Error (log10_C={log10_C:.2f}, log10_gamma={log10_gamma:.2f}): {str(e)[:50]}...")
            _, exp_C = log10_to_exp(log10_C)
            _, exp_gamma = log10_to_exp(log10_gamma)
            search_results.append({
                'log10_C': log10_C,
                'log10_gamma': log10_gamma,
                'C': 10 ** log10_C,
                'gamma': 10 ** log10_gamma,
                'exp_C': exp_C,
                'exp_gamma': exp_gamma,
                'r2': np.nan,
                'rmse': np.nan
            })

    # Save SVR grid results (svr_ prefix)
    results_df = pd.DataFrame(search_results)
    full_save_path = os.path.join(ExperimentConfig.OUTPUT_ROOT, "svr_full_search_results.csv")  # svr_ prefix
    results_df.to_csv(full_save_path, index=False, encoding='utf-8-sig')

    # Statistics of valid results
    valid_count = results_df['rmse'].notna().sum()
    print(f"✅ SVR Grid Search Completed: {valid_count}/{total_combinations} valid results")
    print(f"📄 SVR Grid Results Saved: {full_save_path}")

    # Find optimal parameters
    valid_results = results_df.dropna(subset=['rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['rmse'].idxmin()
        best_params = {
            'best_log10_C': valid_results.loc[best_idx, 'log10_C'],
            'best_log10_gamma': valid_results.loc[best_idx, 'log10_gamma'],
            'best_C': valid_results.loc[best_idx, 'C'],
            'best_gamma': valid_results.loc[best_idx, 'gamma'],
            'best_exp_C': valid_results.loc[best_idx, 'exp_C'],
            'best_exp_gamma': valid_results.loc[best_idx, 'exp_gamma'],
            'best_rmse': valid_results.loc[best_idx, 'rmse'],
            'best_r2': valid_results.loc[best_idx, 'r2']
        }
        print(f"\n🏆 SVR Optimal Parameters:")
        print(f"   log10(C)={best_params['best_log10_C']:.2f}, log10(gamma)={best_params['best_log10_gamma']:.2f}")
        print(f"   C={best_params['best_C']:.2e}=e^({best_params['best_exp_C']:.2f}), gamma={best_params['best_gamma']:.2e}=e^({best_params['best_exp_gamma']:.2f})")
        print(f"   Optimal RMSE: {best_params['best_rmse']:.4f}, Optimal R²: {best_params['best_r2']:.4f}")
    else:
        best_params = {
            'best_log10_C': -999,
            'best_log10_gamma': -999,
            'best_C': np.nan,
            'best_gamma': np.nan,
            'best_exp_C': np.nan,
            'best_exp_gamma': np.nan,
            'best_rmse': baseline_rmse,
            'best_r2': 0.0
        }
        print(f"\n⚠️ No valid SVR grid results, using baseline parameters")

    return results_df, best_params


# ==============================================
# 7. SVR 3D Surface Plotting (Match RF example figure, svr_ prefix)
# ==============================================
def plot_svr_3d_surface_match_example(results_df):
    """Plot SVR 3D surface plot matching the example figure (svr_ prefix)"""
    # Filter invalid values
    df_valid = results_df.dropna(subset=['rmse'])
    if len(df_valid) < 3:
        print(f"Insufficient valid data, skipped SVR 3D surface plot generation")
        return

    # Extract data
    xs = df_valid['log10_C'].values
    ys = df_valid['log10_gamma'].values
    zs = df_valid['rmse'].values

    # Create 3D figure
    fig = plt.figure(figsize=ExperimentConfig.FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    # Generate regular grid
    xi = np.linspace(xs.min(), xs.max(), 100)
    yi = np.linspace(ys.min(), ys.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z values
    zi = griddata((xs, ys), zs, (xi, yi), method='cubic', fill_value=np.nan)

    # Plot 3D surface (matching example figure style)
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

    # Match example figure viewing angle
    ax.view_init(elev=30, azim=135)
    ax.grid(True, alpha=0.3)

    # Chart style
    ax.set_xlabel('log₁₀(C) (Regularization Parameter)', fontsize=12, labelpad=15)
    ax.set_ylabel('log₁₀(gamma) (Kernel Bandwidth Parameter)', fontsize=12, labelpad=15)
    ax.set_zlabel('RMSE', fontsize=12, labelpad=15)
    ax.set_title('SVR Performance Surface: log10(C) vs log10(gamma) → RMSE', fontsize=14, pad=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Mark optimal parameters
    best_idx = df_valid['rmse'].idxmin()
    best_log10_C = df_valid.loc[best_idx, 'log10_C']
    best_log10_gamma = df_valid.loc[best_idx, 'log10_gamma']
    best_rmse = df_valid.loc[best_idx, 'rmse']
    best_exp_C = df_valid.loc[best_idx, 'exp_C']
    best_exp_gamma = df_valid.loc[best_idx, 'exp_gamma']

    # Plot optimal value marker
    ax.scatter(
        [best_log10_C], [best_log10_gamma], [best_rmse],
        s=300, c='#d62728', marker='*',
        label=f'Optimal Combination\nRMSE = {best_rmse:.4f}',
        zorder=5
    )

    # Add parameter annotation
    annotation_text = (
        f'log10(C) = {best_log10_C:.2f}\n'
        f'log10(gamma) = {best_log10_gamma:.2f}\n'
        f'C = e^({best_exp_C:.2f})\n'
        f'gamma = e^({best_exp_gamma:.2f})'
    )
    ax.text(
        best_log10_C, best_log10_gamma, best_rmse + (best_rmse * 0.05 if best_rmse != 0 else 0.05),
        annotation_text, color='#d62728', fontsize=10,
        ha='center', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )

    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

    # Save image (svr_ prefix)
    save_path = os.path.join(PLOTS_DIR, "svr_3d_rmse_match_example.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=ExperimentConfig.FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ SVR Example-matched 3D Surface Plot Saved: {save_path}")


# ==============================================
# 8. Main Function (Consistent with RF process, SVR replacement + new exponential parameter output)
# ==============================================
def main():
    print("=" * 70)
    print("📌 SVR Regression Complete Toolkit (GPU/CPU Adaptive, Files prefixed with svr_)")
    print("=" * 70)

    try:
        # Step 1: Data Preprocessing
        df, X, y = load_and_preprocess_data()

        # Step 2: Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Standardized Feature Matrix Shape: {X_scaled.shape}")

        # Step 3: Baseline Model RMSE
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values,
            test_size=ExperimentConfig.TEST_SIZE,
            random_state=ExperimentConfig.RANDOM_STATE,
            shuffle=True
        )
        baseline_rmse = get_baseline_rmse(X_train, y_train, X_test, y_test)
        print(f"\nBaseline Model (Mean Prediction) RMSE: {baseline_rmse:.4f}")

        # Step 4: SVR Grid Search
        svr_results_df, svr_best_params = run_svr_grid_search(X_scaled, y, baseline_rmse)

        # Step 5: Plot SVR 3D Surface Plot (svr_ prefix)
        print(f"\n🎨 Starting to plot SVR 3D Performance Surface Plot (Matching Example Figure)")
        plot_svr_3d_surface_match_example(svr_results_df)

        # Step 6: Save SVR Grid Data Table (svr_ prefix)
        save_svr_grid_table(svr_results_df, "grid_results")

        # Step 7: Generate SVR Performance Statistics Table (svr_ prefix)
        print(f"\n📋 Starting to generate SVR Performance Statistics Table (Including Full Growth Period)")
        svr_performance_table = create_svr_performance_table(df, svr_best_params, baseline_rmse)
        save_svr_performance_table(svr_performance_table)

        # Step 8: All Features-SVR Visualization + Save True/Predicted CSV
        print(f"\n🎨 Starting to process Origin-style Visualization for All Features-SVR")
        (y_true, y_pred, group_labels,
         group_r2, group_rmse,
         total_r2, total_rmse, overall_a, overall_b) = collect_svr_true_pred_data(df, svr_best_params, baseline_rmse)

        # New: Save All Features true/predicted data CSV
        if len(y_true) > 0 and len(y_pred) > 0 and not np.isnan(total_r2):
            # Save true/predicted CSV
            save_svr_true_pred_csv(
                y_true=y_true,
                y_pred=y_pred,
                group_labels=group_labels,
                save_name="all_features_true_pred_data",
                group_r2=group_r2,
                group_rmse=group_rmse,
                total_r2=total_r2,
                total_rmse=total_rmse
            )
            # Plot visualization
            plot_svr_origin_style_results(
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
            print(f"⚠️ No valid data for All Features-SVR, skipped visualization and CSV saving")

        # Step 9: 6 Major Feature Groups-SVR Independent Visualization + Save Each Feature Group's True/Predicted CSV
        print(f"\n🎨 Starting to process Independent Fitting Visualization for 6 Major Feature Groups-SVR")
        # For summarizing all feature groups' true/predicted data (optional)
        all_group_pred_data = []

        for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
            if group_name == "All Features":
                continue
            print(f"\n{'=' * 50}")
            print(f"Processing Feature Group: {group_name} (SVR)")
            print(f"{'=' * 50}")
            group_data = collect_single_feature_group_svr_data(df, group_name, group_features, svr_best_params,
                                                              baseline_rmse)

            if group_data is not None:
                # New: Save single feature group's true/predicted data CSV
                csv_save_name = f"{group_name.replace(' ', '_').replace('(', '').replace(')', '')}_true_pred_data"
                save_svr_true_pred_csv(
                    y_true=group_data['y_true'],
                    y_pred=group_data['y_pred'],
                    group_labels=group_data['labels'],
                    save_name=csv_save_name,
                    group_r2=group_data['period_r2'],
                    group_rmse=group_data['period_rmse'],
                    total_r2=group_data['total_r2'],
                    total_rmse=group_data['total_rmse']
                )
                # Plot visualization
                plot_single_feature_group_svr_visualization(group_name, group_data)

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

        # (Optional) Save summary CSV of all feature groups' true/predicted data
        if all_group_pred_data:
            total_group_pred_df = pd.concat(all_group_pred_data, ignore_index=True)
            total_save_path = os.path.join(TABLES_DIR, "svr_all_feature_groups_true_pred_summary.csv")
            total_group_pred_df.to_csv(total_save_path, index=False, encoding='utf-8-sig')
            print(f"\n📄 Summary CSV of All Feature Groups' True/Predicted Data Saved: {total_save_path}")

        # Step 10: Output File List
        print(f"\n🎉 All SVR Processes Completed!")
        print(f"\n【SVR Output File List】")
        print(f"### Core Output Files (svr_ prefix) ###")
        print(f"1. SVR Grid Search Results: {os.path.join(ExperimentConfig.OUTPUT_ROOT, 'svr_full_search_results.csv')}")
        print(f"2. SVR 3D Surface Plot: {os.path.join(PLOTS_DIR, 'svr_3d_rmse_match_example.png')}")
        print(f"3. SVR Grid Data Table: {os.path.join(TABLES_DIR, 'svr_grid_results.csv')}")
        print(f"4. SVR Performance Statistics Table (Excel): {os.path.join(TABLES_DIR, 'svr_Performance_Table_With_R2_RMSE_Header.xlsx')}")
        print(f"5. SVR Performance Statistics Table (CSV): {os.path.join(TABLES_DIR, 'svr_Performance_Table_With_R2_RMSE_Header.csv')}")
        print(f"6. SVR All Features True/Predicted Data: {os.path.join(TABLES_DIR, 'svr_all_features_true_pred_data.csv')}")
        print(f"7. SVR Each Feature Group True/Predicted Data: {TABLES_DIR} (File name format: svr_*_true_pred_data.csv)")
        print(
            f"8. SVR All Features Visualization: {os.path.join(PLOTS_DIR, 'svr_origin_style_fitting_plot_with_all_periods.png')}")
        print(f"9. SVR Feature Group Visualizations: {SINGLE_GROUP_PLOTS_DIR}")
        if all_group_pred_data:
            print(
                f"10. SVR All Feature Groups True/Predicted Summary: {os.path.join(TABLES_DIR, 'svr_all_feature_groups_true_pred_summary.csv')}")

    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()