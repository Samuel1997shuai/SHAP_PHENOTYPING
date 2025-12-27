# -*- coding: utf-8 -*-
"""
Complete Toolkit for Extreme Gradient Boosting (XGBoost) Regression (GPU/CPU Adaptive)
Functionality is fully consistent with the original PLS/RF versions, and all file names are uniformly prefixed with "xgb_"

1. XGBoost Parameter Grid Search (n_estimators + max_depth + learning_rate)
2. 3D Performance Surface Plot (Core Parameter Combinations → RMSE) with Corresponding Data Table
3. R²/RMSE Performance Statistics Table Including All Growth Stages (Dual formats: Excel + CSV)
4. Origin-style Visualization (Fitting Curves for Each Growth Stage + All Growth Stages)
5. Independent Fitting Visualization for 6 Major Feature Groups (Newly added function with backward compatibility)
6. Saving of True/Predicted Values (CSV format, stored in the xgb_data_tables directory)
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

# GPU / CuPy / XGBoost GPU Adaptive Configuration
GPU_AVAILABLE = False
USE_CUPY = False
USE_CUML_XGB = False
USE_XGB_GPU = False

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

# Import XGBoost (CPU/GPU)
try:
    import xgboost as xgb
    # Detect XGBoost GPU Availability
    if GPU_AVAILABLE:
        try:
            # Test GPU Availability
            dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
            params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            xgb.train(params, dtrain, num_boost_round=1)
            USE_XGB_GPU = True
        except Exception:
            USE_XGB_GPU = False
    print(f"XGBoost GPU Availability Status: {USE_XGB_GPU}")
except Exception as e:
    raise ImportError(f"❌ XGBoost Library Not Installed: {e}")

# Import cuml XGBoost (Optional)
try:
    import cuml
    from cuml import XGBoostRegressor as cuXGB

    USE_CUML_XGB = True
except Exception:
    USE_CUML_XGB = False

# Import Feature Definitions
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # Support for Chinese character display
plt.rcParams['axes.unicode_minus'] = False  # Fix the issue of negative sign display
print(f"GPU_AVAILABLE={GPU_AVAILABLE}, USE_CUPY={USE_CUPY}, USE_CUML_XGB={USE_CUML_XGB}, USE_XGB_GPU={USE_XGB_GPU}")


# ==============================================
# 2. Experiment Parameter Configuration (All file names start with "xgb_", consistent with original RF functionality)
# ==============================================
class ExperimentConfig:
    """XGBoost Experiment Parameter Configuration Class (Consistent with original RF functionality, naming prefix: xgb_)"""
    # Data Related
    DATA_PATH = '../resource/data_all.xlsx'
    TARGET_COL = 'LAI'  # Leaf Area Index
    MISSING_VALUE_HANDLER = 'drop'

    # Dataset Splitting
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_TEST_SAMPLES = 5

    # Output Configuration (All files/directories start with "xgb_", maintaining the original structure)
    OUTPUT_ROOT = './xgb_3d_surfaces_with_tables'  # Directory with xgb_ prefix
    FIG_DPI = 300
    FIG_SIZE = (14, 10)
    SURFACE_CMAP = 'jet'  # Match the color map of XGBoost example plots
    PLOT_FIG_SIZE = (14, 10)
    SINGLE_GROUP_PLOT_SIZE = (12, 9)

    # XGBoost Core Parameter Grid (Replaces original RF parameters)
    N_ESTIMATORS_GRID = list(range(50, 801, 50))  # Number of boosting rounds
    MAX_DEPTH_GRID = list(range(1, 11))  # Maximum tree depth (XGBoost core parameter)
    LEARNING_RATE_GRID = [0.01, 0.05, 0.1, 0.2, 0.3]  # Learning rate (XGBoost core parameter)
    MAX_ITER = np.arange(50, 351, 50)  # Retained for compatibility, not actually used by XGBoost
    INTERP_GRID_SIZE = 50
    Z_AXIS_LIMIT = None

    # Plot Configuration (Consistent with original RF)
    GROUP_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    GROUP_NAMES = ['Growth Stage 1', 'Growth Stage 2', 'Growth Stage 3', 'All Growth Stages']
    GROUP_LABELS = [1, 2, 3, 4]
    FEATURE_GROUP_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Growth Stage and Feature Group Configuration (Fully consistent with original RF)
    PERIODS = {
        'Growth Stage 1': slice(0, 60),
        'Growth Stage 2': slice(60, 120),
        'Growth Stage 3': slice(120, 180),
        'All Growth Stages': slice(None)
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
        'Growth Stage 1_R²', 'Growth Stage 1_RMSE',
        'Growth Stage 2_R²', 'Growth Stage 2_RMSE',
        'Growth Stage 3_R²', 'Growth Stage 3_RMSE',
        'All Growth Stages_R²', 'All Growth Stages_RMSE'
    ]


# ==============================================
# 3. Initialize Output Directories (Prefixed with "xgb_", consistent with original RF directory structure)
# ==============================================
os.makedirs(ExperimentConfig.OUTPUT_ROOT, exist_ok=True)
TABLES_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "xgb_data_tables")  # Table directory with xgb_ prefix
PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "xgb_plots")  # Plot directory with xgb_ prefix
SINGLE_GROUP_PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "xgb_feature_group_plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SINGLE_GROUP_PLOTS_DIR, exist_ok=True)
for group_name in ExperimentConfig.FEATURE_GROUPS.keys():
    if group_name != "All Features":
        group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
        os.makedirs(group_dir, exist_ok=True)

print(f"📂 XGBoost Result Root Directory: {ExperimentConfig.OUTPUT_ROOT}")
print(f"📊 XGBoost Data Table Directory: {TABLES_DIR}")
print(f"🖼️  XGBoost Visualization Directory: {PLOTS_DIR}")
print(f"🖼️  XGBoost Feature Group Visualization Directory: {SINGLE_GROUP_PLOTS_DIR}")


# ==============================================
# 4. Utility Functions (Integrated GPU/CPU Adaptation + Original RF Utility Functions + New CSV Saving for True/Predicted Values)
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


# ---------------------- Evaluation Metric Utility Functions (Consistent with original RF) ----------------------
def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE (Root Mean Squared Error)"""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (Coefficient of Determination)"""
    return r2_score(y_true, y_pred)


# ---------------------- Surface Grid Generation Utility Function (Consistent with original RF) ----------------------
def create_surface_grid(x_param: np.ndarray, y_param: np.ndarray, z_values: np.ndarray, grid_size: int) -> tuple:
    """Generate smooth surface grid"""
    x_grid = np.linspace(x_param.min(), x_param.max(), grid_size)
    y_grid = np.linspace(y_param.min(), y_param.max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    points = np.vstack((x_param, y_param)).T
    z_mesh = griddata(points, z_values, (x_mesh, y_mesh), method='cubic', fill_value=np.nan)
    return x_mesh, y_mesh, z_mesh


# ---------------------- Baseline Model Utility Function (XGBoost Exclusive) ----------------------
def get_baseline_rmse(X_train, y_train, X_test, y_test):
    """Get RMSE of the baseline model (mean prediction)"""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    return calc_rmse(y_test, y_pred_dummy)


# ---------------------- XGBoost Model Training Utility Function (GPU/CPU Adaptive) ----------------------
def train_xgb_model(n_estimators, max_depth, learning_rate, X_train, y_train, X_test, y_test, baseline_rmse):
    """Train XGBoost model and return R² and RMSE"""
    try:
        if USE_CUML_XGB:
            # GPU Version of XGBoost (cuml)
            X_train_dev = to_device(X_train)
            y_train_dev = to_device(y_train)
            X_test_dev = to_device(X_test)

            model = cuXGB(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                random_state=ExperimentConfig.RANDOM_STATE,
                tree_method='gpu_hist'
            )
            model.fit(X_train_dev, y_train_dev)
            y_pred = to_numpy(model.predict(X_test_dev))

        else:
            # CPU/GPU Version of XGBoost (Native xgboost)
            model = xgb.XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                random_state=ExperimentConfig.RANDOM_STATE,
                n_jobs=-1,
                tree_method='gpu_hist' if USE_XGB_GPU else 'auto',
                gpu_id=0 if USE_XGB_GPU else -1,
                verbosity=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)
        return round(r2, 4), round(rmse, 4)

    except Exception as e:
        print(f"⚠️ XGBoost Training Error (n_est={n_estimators}, max_depth={max_depth}, lr={learning_rate}): {str(e)[:50]}...")
        return np.nan, np.nan


# ---------------------- Save XGBoost Grid Search Results (Prefixed with "xgb_") ----------------------
def save_xgb_grid_table(df_grid, save_name):
    """Save XGBoost grid search result table"""
    save_path = os.path.join(TABLES_DIR, f"xgb_{save_name}.csv")  # File name with xgb_ prefix
    df_grid.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"📄 XGBoost Grid Data Table Saved: {save_path} ({len(df_grid)} valid rows in total)")
    return df_grid


# ---------------------- Performance Statistics Table Utility Functions (Consistent with original RF, prefixed with "xgb_") ----------------------
def create_xgb_performance_table(df, xgb_params, baseline_rmse):
    """Generate XGBoost performance statistics table (replaces original RF model)"""
    table_data = []
    index_tuples = []
    target_col_count = len(ExperimentConfig.TABLE_COLUMNS)

    for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
        valid_features = [f for f in group_features if f in df.columns]
        if not valid_features:
            print(f"⚠️ No valid features in feature group {group_name}, skipped")
            continue

        group_metrics = [np.nan] * target_col_count
        for period_idx, (period_name, period_slice) in enumerate(ExperimentConfig.PERIODS.items()):
            period_df = df.iloc[period_slice]
            X = period_df[valid_features].values
            y = period_df[ExperimentConfig.TARGET_COL].values

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ExperimentConfig.TEST_SIZE,
                random_state=ExperimentConfig.RANDOM_STATE, shuffle=True
            )

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train XGBoost model (using optimal parameters)
            r2, rmse = train_xgb_model(
                n_estimators=xgb_params['best_n_estimators'],
                max_depth=xgb_params['best_max_depth'],
                learning_rate=xgb_params['best_learning_rate'],
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
        index_tuples.append((group_name, 'XGBoost'))

    xgb_table = pd.DataFrame(
        table_data,
        index=pd.MultiIndex.from_tuples(index_tuples, names=['Feature Group', 'Model']),
        columns=ExperimentConfig.TABLE_COLUMNS
    )
    return xgb_table


def save_xgb_performance_table(xgb_table):
    """Save XGBoost performance statistics table (file name prefixed with "xgb_")"""
    excel_path = os.path.join(TABLES_DIR, "xgb_Performance_Table_With_R2_RMSE_Header.xlsx")
    csv_path = os.path.join(TABLES_DIR, "xgb_Performance_Table_With_R2_RMSE_Header.csv")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        xgb_table.to_excel(writer, sheet_name='XGBoost Performance Statistics (Including All Growth Stages)', index=True)
    xgb_table.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n📄 XGBoost Performance Statistics Table Saved:")
    print(f"   - Excel Format: {excel_path}")
    print(f"   - CSV Format: {csv_path}")
    print(f"\n📋 XGBoost Performance Statistics Table Preview:")
    print(xgb_table)
    return xgb_table


# ---------------------- New: Utility Function to Save True/Predicted Data to CSV (Prefixed with "xgb_") ----------------------
def save_xgb_true_pred_csv(y_true, y_pred, group_labels, save_name,
                          group_r2=None, group_rmse=None,
                          total_r2=None, total_rmse=None):
    """
    Save true and predicted values to CSV file (stored in xgb_data_tables directory)
    Parameters:
        y_true: Array of true values
        y_pred: Array of predicted values
        group_labels: Group labels (growth stage numbers)
        save_name: CSV file name prefix (no need to add "xgb_" prefix, automatically supplemented inside the function)
        group_r2: R² of each growth stage (optional, stored in CSV as remarks)
        group_rmse: RMSE of each growth stage (optional, stored in CSV as remarks)
        total_r2: Overall R² (optional, stored in CSV as remarks)
        total_rmse: Overall RMSE (optional, stored in CSV as remarks)
    """
    # Organize true and predicted data into DataFrame
    pred_data = pd.DataFrame({
        'True_Value_' + ExperimentConfig.TARGET_COL: y_true,
        'Predicted_Value_' + ExperimentConfig.TARGET_COL: y_pred,
        'Growth_Stage_Label': group_labels,  # 1=Growth Stage 1, 2=Growth Stage 2, 3=Growth Stage 3, 4=All Growth Stages
        'Growth_Stage_Name': [
            ExperimentConfig.GROUP_NAMES[int(label) - 1] if label in ExperimentConfig.GROUP_LABELS
            else 'Unknown' for label in group_labels
        ]
    })

    # If evaluation metrics are provided, add them to the top of the DataFrame (as remark rows)
    if group_r2 is not None and group_rmse is not None and total_r2 is not None and total_rmse is not None:
        # Construct remark row
        note_row = {
            'True_Value_' + ExperimentConfig.TARGET_COL: 'Remarks:',
            'Predicted_Value_' + ExperimentConfig.TARGET_COL: f"Growth Stage 1_R²={group_r2[0]:.4f}, RMSE={group_rmse[0]:.4f}",
            'Growth_Stage_Label': f"Growth Stage 2_R²={group_r2[1]:.4f}, RMSE={group_rmse[1]:.4f}",
            'Growth_Stage_Name': f"Growth Stage 3_R²={group_r2[2]:.4f}, RMSE={group_rmse[2]:.4f}; All Growth Stages_R²={total_r2:.4f}, RMSE={total_rmse:.4f}"
        }
        # Insert remark row at the first row of DataFrame
        pred_data = pd.concat([pd.DataFrame([note_row]), pred_data], ignore_index=True)

    # Define save path (stored in xgb_data_tables directory, named with "xgb_" prefix)
    save_path = os.path.join(TABLES_DIR, f"xgb_{save_name}.csv")
    # Save CSV (support Chinese, ignore index)
    pred_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    valid_sample_count = len(pred_data) - 1 if 'Remarks:' in pred_data.iloc[0, 0] else len(pred_data)
    print(f"📄 True/Predicted Data CSV Saved: {save_path} ({valid_sample_count} valid samples in total)")
    return save_path


# ---------------------- Data Collection and Visualization Utility Functions (Consistent with original RF, prefixed with "xgb_") ----------------------
def collect_xgb_true_pred_data(df, xgb_params, baseline_rmse):
    """Collect true and predicted values of XGBoost for All Features"""
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

    # Iterate through 3 individual growth stages
    for i, (period_name, period_slice) in enumerate([
        ('Growth Stage 1', config.PERIODS['Growth Stage 1']),
        ('Growth Stage 2', config.PERIODS['Growth Stage 2']),
        ('Growth Stage 3', config.PERIODS['Growth Stage 3'])
    ]):
        period_df = df.iloc[period_slice]
        X = period_df[valid_features].values
        y = period_df[config.TARGET_COL].values

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n🔍 Processing {period_name} (All Features-XGBoost): {len(X)} samples, {len(valid_features)} features")

        # Train XGBoost model
        if USE_CUML_XGB:
            X_train_dev = to_device(X_train_scaled)
            y_train_dev = to_device(y_train)
            X_test_dev = to_device(X_test_scaled)
            model = cuXGB(
                n_estimators=int(xgb_params['best_n_estimators']),
                max_depth=int(xgb_params['best_max_depth']),
                learning_rate=float(xgb_params['best_learning_rate']),
                random_state=config.RANDOM_STATE,
                tree_method='gpu_hist'
            )
            model.fit(X_train_dev, y_train_dev)
            y_pred = to_numpy(model.predict(X_test_dev))
        else:
            model = xgb.XGBRegressor(
                n_estimators=int(xgb_params['best_n_estimators']),
                max_depth=int(xgb_params['best_max_depth']),
                learning_rate=float(xgb_params['best_learning_rate']),
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                tree_method='gpu_hist' if USE_XGB_GPU else 'auto',
                gpu_id=0 if USE_XGB_GPU else -1,
                verbosity=0
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"⚠️ No valid true/predicted values for {period_name} (All Features-XGBoost), skipped")
            group_r2_list.append(np.nan)
            group_rmse_list.append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_group_labels.extend([config.GROUP_LABELS[i]] * len(y_test))
        group_r2_list.append(round(r2, 4))
        group_rmse_list.append(round(rmse, 4))
        valid_group_count += 1
        print(f"✅ {period_name} (All Features-XGBoost) Processed: {len(y_test)} valid test samples, R²={r2:.4f}, RMSE={rmse:.4f}")

    # Process all growth stages
    full_period_name = "All Growth Stages"
    full_period_df = df.iloc[config.PERIODS['All Growth Stages']]
    X_full = full_period_df[valid_features].values
    y_full = full_period_df[config.TARGET_COL].values

    # Split into training and test sets
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, shuffle=True
    )

    # Standardization
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_full_scaled = scaler.transform(X_test_full)

    print(f"\n🔍 Processing {full_period_name} (All Features-XGBoost): {len(X_full)} samples, {len(valid_features)} features")

    # Train XGBoost model
    if USE_CUML_XGB:
        X_train_full_dev = to_device(X_train_full_scaled)
        y_train_full_dev = to_device(y_train_full)
        X_test_full_dev = to_device(X_test_full_scaled)
        model = cuXGB(
            n_estimators=int(xgb_params['best_n_estimators']),
            max_depth=int(xgb_params['best_max_depth']),
            learning_rate=float(xgb_params['best_learning_rate']),
            random_state=config.RANDOM_STATE,
            tree_method='gpu_hist'
        )
        model.fit(X_train_full_dev, y_train_full_dev)
        y_pred_full = to_numpy(model.predict(X_test_full_dev))
    else:
        model = xgb.XGBRegressor(
            n_estimators=int(xgb_params['best_n_estimators']),
            max_depth=int(xgb_params['best_max_depth']),
            learning_rate=float(xgb_params['best_learning_rate']),
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method='gpu_hist' if USE_XGB_GPU else 'auto',
            gpu_id=0 if USE_XGB_GPU else -1,
            verbosity=0
        )
        model.fit(X_train_full_scaled, y_train_full)
        y_pred_full = model.predict(X_test_full_scaled)

    r2_full = calc_r2(y_test_full, y_pred_full)
    rmse_full = calc_rmse(y_test_full, y_pred_full)

    if len(y_test_full) == 0 or np.isnan(r2_full):
        print(f"⚠️ No valid true/predicted values for {full_period_name} (All Features-XGBoost), skipped")
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
            f"✅ {full_period_name} (All Features-XGBoost) Processed: {len(y_test_full)} valid test samples, R²={r2_full:.4f}, RMSE={rmse_full:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_group_labels = np.array(all_group_labels)

    if valid_group_count == 0 or len(all_y_true) == 0 or len(all_y_pred) == 0:
        print(f"\n⚠️ No valid true/predicted values for all groups (including all growth stages) (All Features-XGBoost), cannot generate visualization charts")
        return (all_y_true, all_y_pred, all_group_labels,
                group_r2_list, group_rmse_list,
                np.nan, np.nan, np.nan, np.nan)

    overall_a, overall_b = np.polyfit(all_y_true, all_y_pred, 1)
    total_r2 = calc_r2(all_y_true, all_y_pred)
    total_rmse = calc_rmse(all_y_true, all_y_pred)

    print(
        f"\n📊 Combined Statistics for All Valid Groups (Including All Growth Stages) (All Features-XGBoost): {len(all_y_true)} total samples, Combined R²={total_r2:.4f}, Combined RMSE={total_rmse:.4f}")
    return (all_y_true, all_y_pred, all_group_labels,
            group_r2_list, group_rmse_list,
            total_r2, total_rmse, overall_a, overall_b)


def plot_xgb_origin_style_results(y_true, y_pred, groups, group_r2_values, group_rmse_values,
                                 total_r2, total_rmse, overall_a, overall_b,
                                 save_name: str = "xgb_origin_style_fitting_plot_with_all_periods"):
    """Plot XGBoost Origin-style visualization (file name prefixed with "xgb_")"""
    config = ExperimentConfig()

    if len(y_true) == 0 or len(y_pred) == 0 or np.isnan(total_r2) or np.isnan(total_rmse):
        print(f"⚠️ No valid data for plotting (All Features-XGBoost), skipped Origin-style visualization chart generation")
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
    ax.set_title('XGBoost Model True vs. Predicted Value Fitting Results (Origin Style + All Features + All Growth Stages)', fontsize=16, fontweight='bold',
                 pad=20)

    ax.legend(
        loc='lower right', frameon=True, framealpha=0.9,
        edgecolor='gray', fontsize=10, labelspacing=0.8
    )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{save_name}.png")  # File name with xgb_ prefix
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f'\n✅ Origin-style visualization chart including all growth stages saved to (All Features-XGBoost): {save_path}')
    plt.show()
    plt.close()


# ---------------------- New: XGBoost Processing Function for Single Feature Group ----------------------
def collect_single_feature_group_xgb_data(df, group_name, group_features, xgb_params, baseline_rmse):
    """Collect true and predicted values of XGBoost for a single feature group"""
    config = ExperimentConfig()
    valid_features = [f for f in group_features if f in df.columns]
    if not valid_features:
        print(f"⚠️ No valid features in feature group {group_name}, skipped")
        return None

    group_data = {
        'y_true': [], 'y_pred': [], 'labels': [],
        'period_r2': [], 'period_rmse': [], 'valid_count': 0
    }

    for i, (period_name, period_slice) in enumerate(config.PERIODS.items()):
        period_df = df.iloc[period_slice]
        X = period_df[valid_features].values
        y = period_df[config.TARGET_COL].values

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, shuffle=True
        )

        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"   🔍 Processing {period_name} ({group_name}-XGBoost): {len(X)} samples, {len(valid_features)} features")

        # Train XGBoost model
        r2, rmse, y_pred = None, None, None
        if USE_CUML_XGB:
            X_train_dev = to_device(X_train_scaled)
            y_train_dev = to_device(y_train)
            X_test_dev = to_device(X_test_scaled)
            model = cuXGB(
                n_estimators=int(xgb_params['best_n_estimators']),
                max_depth=int(xgb_params['best_max_depth']),
                learning_rate=float(xgb_params['best_learning_rate']),
                random_state=config.RANDOM_STATE,
                tree_method='gpu_hist'
            )
            model.fit(X_train_dev, y_train_dev)
            y_pred = to_numpy(model.predict(X_test_dev))
        else:
            model = xgb.XGBRegressor(
                n_estimators=int(xgb_params['best_n_estimators']),
                max_depth=int(xgb_params['best_max_depth']),
                learning_rate=float(xgb_params['best_learning_rate']),
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                tree_method='gpu_hist' if USE_XGB_GPU else 'auto',
                gpu_id=0 if USE_XGB_GPU else -1,
                verbosity=0
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"   ⚠️ No valid data for {period_name}, skipped")
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
        print(f"   ❌ No valid growth stage data for feature group {group_name}, skipped visualization")
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


def plot_single_feature_group_xgb_visualization(group_name, group_data):
    """Plot Origin-style fitting curve for XGBoost of a single feature group (prefixed with "xgb_")"""
    config = ExperimentConfig()
    group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
    save_name = f"xgb_{group_name}_fitting_plot".replace('/', '_').replace(':', '')  # File name with xgb_ prefix

    fig, ax = plt.subplots(figsize=config.SINGLE_GROUP_PLOT_SIZE)
    color_idx = list(ExperimentConfig.FEATURE_GROUPS.keys()).index(group_name)
    group_color = config.FEATURE_GROUP_PLOT_COLORS[color_idx]

    # Plot scatter points and fitting lines for each growth stage
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
    ax.set_title(f'XGBoost Fitting Results - {group_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    # Save
    plt.tight_layout()
    save_path = os.path.join(group_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f"   🖼️  XGBoost visualization chart for feature group {group_name} saved: {save_path}")
    plt.close()


# ==============================================
# 5. Data Processing Module (Consistent with original RF)
# ==============================================
def load_and_preprocess_data():
    """Data reading and preprocessing"""
    print(f"\n📥 Reading Data: {ExperimentConfig.DATA_PATH}")
    try:
        df = pd.read_excel(ExperimentConfig.DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data File Not Found: {ExperimentConfig.DATA_PATH}")

    # Verify features and target variable
    missing_features = [col for col in all_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following feature columns are missing in the data: {missing_features}\nPlease check if the feature list is consistent with the data column names!")

    if ExperimentConfig.TARGET_COL not in df.columns:
        raise ValueError(f"Target variable column is missing in the data: {ExperimentConfig.TARGET_COL}\nPlease confirm if the target variable column name is correct!")

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
        print("All missing values filled with mean")

    # Print XGBoost parameter grid information
    print(
        f"XGBoost Parameter Grid: {len(ExperimentConfig.N_ESTIMATORS_GRID)} points for n_estimators, {len(ExperimentConfig.MAX_DEPTH_GRID)} points for max_depth, {len(ExperimentConfig.LEARNING_RATE_GRID)} points for learning_rate")

    return df, X, y


# ==============================================
# 6. XGBoost Grid Search (Replaces original RF grid search)
# ==============================================
def run_xgb_grid_search(X_scaled, y, baseline_rmse):
    """Execute XGBoost grid search"""
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values,
        test_size=ExperimentConfig.TEST_SIZE,
        random_state=ExperimentConfig.RANDOM_STATE,
        shuffle=True
    )

    # Select two core parameters for 3D visualization (n_estimators + max_depth)
    param_combinations = list(itertools.product(
        ExperimentConfig.N_ESTIMATORS_GRID,
        ExperimentConfig.MAX_DEPTH_GRID,
        ExperimentConfig.LEARNING_RATE_GRID
    ))
    total_combinations = len(param_combinations)
    print(f"\n🚀 Starting XGBoost Multi-Parameter Grid Search (Total Combinations: {total_combinations})")

    search_results = []
    for idx, (n_est, max_depth, lr) in enumerate(param_combinations, 1):
        if idx % 50 == 0 or idx == total_combinations:
            print(f"   Progress: {idx}/{total_combinations} (Current: n_est={n_est}, max_depth={max_depth}, lr={lr})")

        # Train XGBoost model
        r2, rm = train_xgb_model(n_est, max_depth, lr, X_train, y_train, X_test, y_test, baseline_rmse)
        search_results.append({
            'n_estimators': int(n_est),
            'max_depth': int(max_depth),
            'learning_rate': float(lr),
            'r2': r2,
            'rmse': rm
        })

    # Save XGBoost grid results (prefixed with "xgb_")
    results_df = pd.DataFrame(search_results)
    full_save_path = os.path.join(ExperimentConfig.OUTPUT_ROOT, "xgb_full_search_results.csv")  # Prefix with xgb_
    results_df.to_csv(full_save_path, index=False, encoding='utf-8-sig')

    # Statistics of valid results
    valid_count = results_df['rmse'].notna().sum()
    print(f"✅ XGBoost Grid Search Completed: {valid_count}/{total_combinations} valid results")
    print(f"📄 XGBoost Grid Results Saved: {full_save_path}")

    # Find optimal parameters
    valid_results = results_df.dropna(subset=['rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['rmse'].idxmin()
        best_params = {
            'best_n_estimators': int(valid_results.loc[best_idx, 'n_estimators']),
            'best_max_depth': int(valid_results.loc[best_idx, 'max_depth']),
            'best_learning_rate': float(valid_results.loc[best_idx, 'learning_rate']),
            'best_rmse': valid_results.loc[best_idx, 'rmse'],
            'best_r2': valid_results.loc[best_idx, 'r2']
        }
        print(f"\n🏆 XGBoost Optimal Parameters: n_estimators={best_params['best_n_estimators']}, max_depth={best_params['best_max_depth']}, learning_rate={best_params['best_learning_rate']}")
        print(f"   Optimal RMSE: {best_params['best_rmse']:.4f}, Optimal R²: {best_params['best_r2']:.4f}")
    else:
        best_params = {'best_n_estimators': 100, 'best_max_depth': 3, 'best_learning_rate': 0.1, 'best_rmse': baseline_rmse, 'best_r2': 0.0}
        print(f"\n⚠️ No valid XGBoost grid results, using default parameters")

    return results_df, best_params


# ==============================================
# 7. XGBoost 3D Surface Plotting (Matches example plot, prefixed with "xgb_")
# ==============================================
def plot_xgb_3d_surface_match_example(results_df):
    """Plot XGBoost 3D surface plot matching the example (prefixed with "xgb_")"""
    # Filter invalid values
    df_valid = results_df.dropna(subset=['rmse'])
    if len(df_valid) < 3:
        print(f"Insufficient valid data, skipped XGBoost 3D surface plot generation")
        return

    # Extract data (use n_estimators and max_depth as X/Y axes, RMSE as Z axis)
    xs = df_valid['n_estimators'].values
    ys = df_valid['max_depth'].values
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
    ax.set_ylabel('max_depth', fontsize=12, labelpad=15)
    ax.set_zlabel('RMSE', fontsize=12, labelpad=15)
    ax.set_title('XGBoost Performance Surface: n_estimators vs max_depth → RMSE', fontsize=14, pad=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Save image (prefixed with "xgb_")
    save_path = os.path.join(PLOTS_DIR, "xgb_3d_rmse_match_example.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=ExperimentConfig.FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ XGBoost Example-Matched 3D Surface Plot Saved: {save_path}")


# ==============================================
# 8. Main Function (Consistent with original RF process, XGBoost replacement + new CSV saving for true/predicted values)
# ==============================================
def main():
    print("=" * 70)
    print("📌 Complete XGBoost Regression Toolkit (GPU/CPU Adaptive, File Names Prefixed with 'xgb_')")
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

        # Step 4: XGBoost Grid Search
        xgb_results_df, xgb_best_params = run_xgb_grid_search(X_scaled, y, baseline_rmse)

        # Step 5: Plot XGBoost 3D Surface Plot (prefixed with "xgb_")
        print(f"\n🎨 Starting to plot XGBoost 3D Performance Surface Plot (matching example)")
        plot_xgb_3d_surface_match_example(xgb_results_df)

        # Step 6: Save XGBoost Grid Data Table (prefixed with "xgb_")
        save_xgb_grid_table(xgb_results_df, "grid_results")

        # Step 7: Generate XGBoost Performance Statistics Table (prefixed with "xgb_")
        print(f"\n📋 Starting to generate XGBoost Performance Statistics Table (including all growth stages)")
        xgb_performance_table = create_xgb_performance_table(df, xgb_best_params, baseline_rmse)
        save_xgb_performance_table(xgb_performance_table)

        # Step 8: All Features-XGBoost Visualization + Save True/Predicted Data to CSV
        print(f"\n🎨 Starting to process Origin-style visualization for All Features-XGBoost")
        (y_true, y_pred, group_labels,
         group_r2, group_rmse,
         total_r2, total_rmse, overall_a, overall_b) = collect_xgb_true_pred_data(df, xgb_best_params, baseline_rmse)

        # New: Save All Features true/predicted data to CSV
        if len(y_true) > 0 and len(y_pred) > 0 and not np.isnan(total_r2):
            # Save true/predicted CSV
            save_xgb_true_pred_csv(
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
            plot_xgb_origin_style_results(
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
            print(f"⚠️ No valid data for All Features-XGBoost, skipped visualization and CSV saving")

        # Step 9: 6 Major Feature Groups-XGBoost Independent Visualization + Save True/Predicted Data for Each Feature Group to CSV
        print(f"\n🎨 Starting to process independent fitting visualization for 6 major feature groups-XGBoost")
        # For summarizing true/predicted data of all feature groups (optional)
        all_group_pred_data = []

        for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
            if group_name == "All Features":
                continue
            print(f"\n{'=' * 50}")
            print(f"Processing Feature Group: {group_name} (XGBoost)")
            print(f"{'=' * 50}")
            group_data = collect_single_feature_group_xgb_data(df, group_name, group_features, xgb_best_params,
                                                              baseline_rmse)

            if group_data is not None:
                # New: Save true/predicted data of single feature group to CSV
                csv_save_name = f"{group_name.replace(' ', '_').replace('(', '').replace(')', '')}_true_pred_data"
                save_xgb_true_pred_csv(
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
                plot_single_feature_group_xgb_visualization(group_name, group_data)

                # (Optional) Summarize into total data
                group_pred_df = pd.DataFrame({
                    'Feature_Group': group_name,
                    'True_Value_' + ExperimentConfig.TARGET_COL: group_data['y_true'],
                    'Predicted_Value_' + ExperimentConfig.TARGET_COL: group_data['y_pred'],
                    'Growth_Stage_Label': group_data['labels'],
                    'Growth_Stage_Name': [
                        ExperimentConfig.GROUP_NAMES[int(label) - 1] if label in ExperimentConfig.GROUP_LABELS
                        else 'Unknown' for label in group_data['labels']
                    ]
                })
                all_group_pred_data.append(group_pred_df)

        # (Optional) Save summarized true/predicted data of all feature groups to CSV
        if all_group_pred_data:
            total_group_pred_df = pd.concat(all_group_pred_data, ignore_index=True)
            total_save_path = os.path.join(TABLES_DIR, "xgb_all_feature_groups_true_pred_summary.csv")
            total_group_pred_df.to_csv(total_save_path, index=False, encoding='utf-8-sig')
            print(f"\n📄 Summarized True/Predicted Data CSV of All Feature Groups Saved: {total_save_path}")

        # Step 10: Output File List
        print(f"\n🎉 All XGBoost Processes Completed!")
        print(f"\n【XGBoost Output File List】")
        print(f"### Core Output Files (Prefixed with 'xgb_') ###")
        print(f"1. XGBoost Grid Search Results: {os.path.join(ExperimentConfig.OUTPUT_ROOT, 'xgb_full_search_results.csv')}")
        print(f"2. XGBoost 3D Surface Plot: {os.path.join(PLOTS_DIR, 'xgb_3d_rmse_match_example.png')}")
        print(f"3. XGBoost Grid Data Table: {os.path.join(TABLES_DIR, 'xgb_grid_results.csv')}")
        print(f"4. XGBoost Performance Statistics Table (Excel): {os.path.join(TABLES_DIR, 'xgb_Performance_Table_With_R2_RMSE_Header.xlsx')}")
        print(f"5. XGBoost Performance Statistics Table (CSV): {os.path.join(TABLES_DIR, 'xgb_Performance_Table_With_R2_RMSE_Header.csv')}")
        print(f"6. XGBoost All Features True/Predicted Data: {os.path.join(TABLES_DIR, 'xgb_all_features_true_pred_data.csv')}")
        print(f"7. XGBoost True/Predicted Data for Each Feature Group: {TABLES_DIR} (File name format: xgb_*_true_pred_data.csv)")
        print(
            f"8. XGBoost All Features Visualization: {os.path.join(PLOTS_DIR, 'xgb_origin_style_fitting_plot_with_all_periods.png')}")
        print(f"9. XGBoost Feature Group Visualization: {SINGLE_GROUP_PLOTS_DIR}")
        if all_group_pred_data:
            print(
                f"10. XGBoost Summarized True/Predicted Data of All Feature Groups: {os.path.join(TABLES_DIR, 'xgb_all_feature_groups_true_pred_summary.csv')}")

    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()