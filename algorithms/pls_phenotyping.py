
# -*- coding: utf-8 -*-
"""
PLS Regression Complete Toolkit (GPU/CPU Adaptive + Fully Consistent with RF Functionality)
1. PLS parameter grid search (n_components + scaling + max_iter)
2. 3D performance surface plot (n_components vs scaling → RMSE) + corresponding data table
3. R²/RMSE performance statistics table including full growth period (Excel + CSV dual format)
4. Origin-style visualization (fitting curves for each growth period + full growth period)
5. Independent fitting visualization for 6 major feature groups (maintaining compatibility)
6. True/predicted value data saving (CSV format, stored in pls_data_tables directory)
"""

# ==============================================
# 1. Dependency Import (Integrated GPU/CPU Adaptive + Original Functional Libraries)
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
from sklearn.cross_decomposition import PLSRegression

# GPU / CuPy / cuML Adaptive Configuration (PLS Exclusive)
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
    # Import cuML PLS Regression
    import cuml
    from cuml.cross_decomposition import PLSRegression as cuPLS
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
plt.rcParams['font.sans-serif'] = ['Arial']  # English font support
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
print(f"GPU_AVAILABLE={GPU_AVAILABLE}, USE_CUPY={USE_CUPY}, USE_CUML={USE_CUML}")

# ==============================================
# 2. Experiment Parameter Configuration (File Naming Unified with pls_ Prefix, Consistent with RF)
# ==============================================
class ExperimentConfig:
    """PLS Experiment Parameter Configuration Class (Consistent with RF Functionality, pls_ Prefix)"""
    # Data Related
    DATA_PATH = '../resource/data_all.xlsx'
    TARGET_COL = 'LAI'
    MISSING_VALUE_HANDLER = 'drop'

    # Dataset Splitting
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    MIN_TEST_SAMPLES = 5

    # Output Configuration (All Files/Directories with pls_ Prefix, Maintaining RF Structure)
    OUTPUT_ROOT = './pls_3d_surfaces_with_tables'
    FIG_DPI = 300
    FIG_SIZE = (14, 10)
    SURFACE_CMAP = 'plasma'  # PLS Exclusive Colormap
    PLOT_FIG_SIZE = (14, 10)
    SINGLE_GROUP_PLOT_SIZE = (12, 9)

    # PLS Core Parameter Grid (Exclusive Tuning Settings, Replacing RF Parameters)
    N_COMPONENTS = None  # Number of principal components, dynamically adaptive
    SCALING = [True, False]  # Whether to standardize
    MAX_ITER = np.arange(50, 351, 50)  # Maximum number of iterations
    INTERP_GRID_SIZE = 50
    Z_AXIS_LIMIT = None

    # Plot Configuration (Fully Consistent with RF)
    GROUP_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    GROUP_NAMES = ['Growth Period 1', 'Growth Period 2', 'Growth Period 3', 'Full Growth Period']
    GROUP_LABELS = [1, 2, 3, 4]
    FEATURE_GROUP_PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Growth Period and Feature Group Configuration (Fully Consistent with RF)
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
# 3. Initialize Output Directories (pls_ Prefix, Consistent with RF Directory Structure)
# ==============================================
os.makedirs(ExperimentConfig.OUTPUT_ROOT, exist_ok=True)
TABLES_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "pls_data_tables")
PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "pls_plots")
SINGLE_GROUP_PLOTS_DIR = os.path.join(ExperimentConfig.OUTPUT_ROOT, "pls_feature_group_plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SINGLE_GROUP_PLOTS_DIR, exist_ok=True)
for group_name in ExperimentConfig.FEATURE_GROUPS.keys():
    if group_name != "All Features":
        group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
        os.makedirs(group_dir, exist_ok=True)

print(f"📂 PLS Result Root Directory: {ExperimentConfig.OUTPUT_ROOT}")
print(f"📊 PLS Data Table Directory: {TABLES_DIR}")
print(f"🖼️  PLS Visualization Directory: {PLOTS_DIR}")
print(f"🖼️  PLS Feature Group Visualization Directory: {SINGLE_GROUP_PLOTS_DIR}")

# ==============================================
# 4. Utility Functions (Integrated GPU/CPU Adaptive + RF Same Functions + PLS Exclusive Logic)
# ==============================================
# ---------------------- GPU/CPU Data Conversion Utility Functions (Consistent with RF) ----------------------
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

# ---------------------- Evaluation Metric Utility Functions (Consistent with RF) ----------------------
def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE"""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R²"""
    return r2_score(y_true, y_pred)

# ---------------------- Surface Grid Generation Utility Functions (Consistent with RF) ----------------------
def create_surface_grid(x_param: np.ndarray, y_param: np.ndarray, z_values: np.ndarray, grid_size: int) -> tuple:
    """Generate smooth surface grid"""
    x_grid = np.linspace(x_param.min(), x_param.max(), grid_size)
    y_grid = np.linspace(y_param.min(), y_param.max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    points = np.vstack((x_param, y_param)).T
    z_mesh = griddata(points, z_values, (x_mesh, y_mesh), method='linear')
    return x_mesh, y_mesh, z_mesh

def bool_to_int(bool_val: bool) -> int:
    """Boolean to integer conversion (PLS Exclusive Auxiliary Function)"""
    return 1 if bool_val else 0

# ---------------------- PLS Auxiliary Utility Functions (Exclusive Logic) ----------------------
def get_valid_n_components(X: np.ndarray, y: np.ndarray, input_n_components: int) -> int:
    """Get valid PLS number of principal components (PLS Exclusive)"""
    if X.size == 0 or y.size == 0:
        return 1
    n_samples, n_features = X.shape
    max_valid = min(n_samples - 1, n_features, input_n_components)
    return max(1, max_valid)

# ---------------------- Save PLS Grid Search Results (pls_ Prefix, Consistent with RF) ----------------------
def save_pls_grid_table(df_grid, save_name):
    """Save PLS grid search result table"""
    save_path = os.path.join(TABLES_DIR, f"pls_{save_name}.csv")
    df_grid.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"📄 PLS Grid Data Table Saved: {save_path} ({len(df_grid)} valid rows)")
    return df_grid

def save_surface_table(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                       x_name: str, y_name: str, z_name: str, save_name: str):
    """Save PLS surface plot corresponding table (pls_ Prefix)"""
    table_df = pd.DataFrame({
        x_name: x_data,
        y_name: y_data,
        z_name: z_data
    })
    if 'scaling' in table_df.columns:
        table_df['scaling'] = table_df['scaling'].apply(bool_to_int)
    table_df = table_df.dropna(subset=[z_name])
    save_path = os.path.join(TABLES_DIR, f"pls_{save_name}.csv")
    table_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"📄 PLS Surface Data Table Saved: {save_path} ({len(table_df)} valid rows)")
    return table_df

# ---------------------- PLS Model Training Utility Function (GPU/CPU Adaptive + Exclusive Logic) ----------------------
def train_pls_model(X, y, pls_params: dict, return_true_pred: bool = False) -> tuple:
    """Train PLS model (GPU/CPU adaptive, maintaining consistent input/output with RF)"""
    if X.size == 0 or y.size == 0:
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan
    if len(X) < 5:
        # 修复f-string语法错误：将 (only len(X)) 改为 only {len(X)}
        print(f"⚠️ Insufficient sample size (only {len(X)} samples), cannot split train/test sets, skipping")
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan
    if X.shape[1] < 1:
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan

    # Adjust test set ratio
    test_size = ExperimentConfig.TEST_SIZE
    min_test_samples = 5
    while int(len(X) * test_size) < min_test_samples and test_size > 0.1:
        test_size += 0.05
    if int(len(X) * test_size) < min_test_samples:
        print(f"⚠️ Test set samples still insufficient ({min_test_samples} required) even after adjusting ratio, skipping")
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan

    # Split train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=ExperimentConfig.RANDOM_STATE, shuffle=True
    )

    if len(X_test) < min_test_samples or len(y_test) < min_test_samples:
        print(f"⚠️ Insufficient test set samples after splitting ({min_test_samples} required), skipping")
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan

    # Standardization processing
    X_train_scaled, X_test_scaled = X_train, X_test
    if pls_params['scaling']:
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"⚠️ Standardization failed: {str(e)[:30]}, skipping")
            if return_true_pred:
                return np.nan, np.nan, np.array([]), np.array([])
            return np.nan, np.nan

    # Correct number of principal components
    current_n_components = get_valid_n_components(X_train_scaled, y_train, pls_params['n_components'])
    if len(X_train_scaled) < current_n_components + 1:
        print(f"⚠️ Insufficient training samples ({current_n_components + 1} required, only {len(X_train_scaled)} available), skipping")
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan

    try:
        if USE_CUML:
            # GPU version PLS
            X_train_dev = to_device(X_train_scaled)
            y_train_dev = to_device(y_train).reshape(-1, 1)
            X_test_dev = to_device(X_test_scaled)

            model = cuPLS(
                n_components=int(current_n_components),
                scale=False,  # Already standardized in advance, set to False here
                max_iter=int(pls_params['max_iter']),
                tol=1e-06
            )
            model.fit(X_train_dev, y_train_dev)
            y_pred = to_numpy(model.predict(X_test_dev))
        else:
            # CPU version PLS (sklearn)
            model = PLSRegression(
                n_components=int(current_n_components),
                scale=False,  # Already standardized in advance, set to False here
                max_iter=int(pls_params['max_iter']),
                tol=1e-06,
                copy=True
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        # Data format sorting
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()

        # Calculate evaluation metrics
        r2 = calc_r2(y_test, y_pred)
        rmse = calc_rmse(y_test, y_pred)
        if not (-1 <= r2 <= 1) or rmse < 0:
            if return_true_pred:
                return np.nan, np.nan, np.array([]), np.array([])
            return np.nan, np.nan

        if return_true_pred:
            return round(r2, 4), round(rmse, 4), y_test, y_pred
        return round(r2, 4), round(rmse, 4)

    except Exception as e:
        print(f"⚠️ PLS model training failed: {str(e)[:30]}, skipping")
        if return_true_pred:
            return np.nan, np.nan, np.array([]), np.array([])
        return np.nan, np.nan

# ---------------------- PLS Performance Statistics Table Utility Functions (Consistent with RF, pls_ Prefix) ----------------------
def create_pls_performance_table(df, pls_params):
    """Generate PLS performance statistics table (replacing RF model, maintaining consistent format)"""
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

            # Train PLS model (using optimal parameters)
            r2, rmse = train_pls_model(X, y, pls_params, return_true_pred=False)

            r2_col_idx = period_idx * 2
            rmse_col_idx = period_idx * 2 + 1
            if r2_col_idx < target_col_count and not np.isnan(r2):
                group_metrics[r2_col_idx] = r2
            if rmse_col_idx < target_col_count and not np.isnan(rmse):
                group_metrics[rmse_col_idx] = rmse

        table_data.append(group_metrics)
        index_tuples.append((group_name, 'PLSR'))

    pls_table = pd.DataFrame(
        table_data,
        index=pd.MultiIndex.from_tuples(index_tuples, names=['Feature Group', 'Model']),
        columns=ExperimentConfig.TABLE_COLUMNS
    )
    return pls_table

def save_pls_performance_table(pls_table):
    """Save PLS performance statistics table (pls_ prefix filename, consistent with RF format)"""
    excel_path = os.path.join(TABLES_DIR, "pls_Performance_Table_With_R2_RMSE_Header.xlsx")
    csv_path = os.path.join(TABLES_DIR, "pls_Performance_Table_With_R2_RMSE_Header.csv")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pls_table.to_excel(writer, sheet_name='PLS Performance Statistics (Including Full Growth Period)', index=True)
    pls_table.reset_index().to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\n📄 PLS Performance Statistics Table Saved:")
    print(f"   - Excel Format: {excel_path}")
    print(f"   - CSV Format: {csv_path}")
    print(f"\n📋 PLS Performance Statistics Table Preview:")
    print(pls_table)
    return pls_table

# ---------------------- Save True/Predicted Data to CSV Utility Function (PLS Exclusive, Consistent with RF Functionality) ----------------------
def save_pls_true_pred_csv(y_true, y_pred, group_labels, save_name,
                          group_r2=None, group_rmse=None,
                          total_r2=None, total_rmse=None):
    """
    Save PLS true/predicted data to CSV file (stored in pls_data_tables directory)
    Fully consistent with RF homonymous function in functionality and format
    """
    # Organize true/predicted data into DataFrame
    pred_data = pd.DataFrame({
        'True_Value_' + ExperimentConfig.TARGET_COL: y_true,
        'Predicted_Value_' + ExperimentConfig.TARGET_COL: y_pred,
        'Growth_Period_Label': group_labels,
        'Growth_Period_Name': [
            ExperimentConfig.GROUP_NAMES[int(label) - 1] if label in ExperimentConfig.GROUP_LABELS
            else 'Unknown' for label in group_labels
        ]
    })

    # Add evaluation metric note row
    if group_r2 is not None and group_rmse is not None and total_r2 is not None and total_rmse is not None:
        note_row = {
            'True_Value_' + ExperimentConfig.TARGET_COL: 'Note:',
            'Predicted_Value_' + ExperimentConfig.TARGET_COL: f'Growth Period 1_R²={group_r2[0]:.4f}, RMSE={group_rmse[0]:.4f}',
            'Growth_Period_Label': f'Growth Period 2_R²={group_r2[1]:.4f}, RMSE={group_rmse[1]:.4f}',
            'Growth_Period_Name': f'Growth Period 3_R²={group_r2[2]:.4f}, RMSE={group_rmse[2]:.4f}; Full Growth Period_R²={total_r2:.4f}, RMSE={total_rmse:.4f}'
        }
        pred_data = pd.concat([pd.DataFrame([note_row]), pred_data], ignore_index=True)

    # Define save path (pls_ prefix)
    save_path = os.path.join(TABLES_DIR, f"pls_{save_name}.csv")
    pred_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    valid_sample_count = len(pred_data) - 1 if 'Note:' in pred_data.iloc[0, 0] else len(pred_data)
    print(f"📄 PLS True/Predicted Data CSV Saved: {save_path} ({valid_sample_count} valid samples)")
    return save_path

# ---------------------- Data Collection and Visualization Utility Functions (Consistent with RF, PLS Exclusive Logic) ----------------------
def collect_pls_true_pred_data(df, pls_params):
    """Collect PLS true and predicted values for All Features (consistent with RF process)"""
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

        print(f"\n🔍 Processing {period_name} (All Features-PLS): {len(X)} samples, {len(valid_features)} features")
        r2, rmse, y_test, y_pred = train_pls_model(X, y, pls_params, return_true_pred=True)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"⚠️ No valid true/predicted values for {period_name} (All Features-PLS), skipping")
            group_r2_list.append(np.nan)
            group_rmse_list.append(np.nan)
            continue

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_group_labels.extend([config.GROUP_LABELS[i]] * len(y_test))
        group_r2_list.append(r2)
        group_rmse_list.append(rmse)
        valid_group_count += 1
        print(f"✅ {period_name} (All Features-PLS) processed: {len(y_test)} valid test samples, R²={r2:.4f}, RMSE={rmse:.4f}")

    # Process full growth period
    full_period_name = "Full Growth Period"
    full_period_df = df.iloc[config.PERIODS['Full Growth Period']]
    X_full = full_period_df[valid_features].values
    y_full = full_period_df[config.TARGET_COL].values
    print(f"\n🔍 Processing {full_period_name} (All Features-PLS): {len(X_full)} samples, {len(valid_features)} features")

    r2_full, rmse_full, y_test_full, y_pred_full = train_pls_model(
        X_full, y_full, pls_params, return_true_pred=True
    )

    if len(y_test_full) == 0 or np.isnan(r2_full):
        print(f"⚠️ No valid true/predicted values for {full_period_name} (All Features-PLS), skipping")
        group_r2_list.append(np.nan)
        group_rmse_list.append(np.nan)
    else:
        all_y_true.extend(y_test_full)
        all_y_pred.extend(y_pred_full)
        all_group_labels.extend([config.GROUP_LABELS[3]] * len(y_test_full))
        group_r2_list.append(r2_full)
        group_rmse_list.append(rmse_full)
        valid_group_count += 1
        print(f"✅ {full_period_name} (All Features-PLS) processed: {len(y_test_full)} valid test samples, R²={r2_full:.4f}, RMSE={rmse_full:.4f}")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_group_labels = np.array(all_group_labels)

    if valid_group_count == 0 or len(all_y_true) == 0 or len(all_y_pred) == 0:
        print(f"\n⚠️ No valid true/predicted values for all groups (including full growth period) (All Features-PLS), cannot generate visualization charts")
        return (all_y_true, all_y_pred, all_group_labels,
                group_r2_list, group_rmse_list,
                np.nan, np.nan, np.nan, np.nan)

    overall_a, overall_b = np.polyfit(all_y_true, all_y_pred, 1)
    total_r2 = calc_r2(all_y_true, all_y_pred)
    total_rmse = calc_rmse(all_y_true, all_y_pred)

    print(
        f"\n📊 Combined statistics for all valid groups (including full growth period) (All Features-PLS): {len(all_y_true)} total samples, combined R²={total_r2:.4f}, combined RMSE={total_rmse:.4f}")
    return (all_y_true, all_y_pred, all_group_labels,
            group_r2_list, group_rmse_list,
            total_r2, total_rmse, overall_a, overall_b)

def plot_pls_origin_style_results(y_true, y_pred, groups, group_r2_values, group_rmse_values,
                                 total_r2, total_rmse, overall_a, overall_b,
                                 save_name: str = "pls_origin_style_fitting_plot_with_all_periods"):
    """Plot PLS Origin-style visualization (pls_ prefix filename, consistent with RF style)"""
    config = ExperimentConfig()

    if len(y_true) == 0 or len(y_pred) == 0 or np.isnan(total_r2) or np.isnan(total_rmse):
        print(f"⚠️ No valid data for plotting (All Features-PLS), skipping Origin-style visualization chart generation")
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
    ax.set_title('PLS Model True vs Predicted Value Fitting Results (Origin Style + All Features + Full Growth Period)', fontsize=16, fontweight='bold',
                 pad=20)

    ax.legend(
        loc='lower right', frameon=True, framealpha=0.9,
        edgecolor='gray', fontsize=10, labelspacing=0.8
    )

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{save_name}.png")
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f'\n✅ Origin-style visualization chart including full growth period saved (All Features-PLS): {save_path}')
    plt.show()
    plt.close()

# ---------------------- Single Feature Group PLS Processing Function (Consistent with RF) ----------------------
def collect_single_feature_group_pls_data(df, group_name, group_features, pls_params):
    """Collect PLS true and predicted values for single feature group (consistent with RF process)"""
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

        print(f"   🔍 Processing {period_name} ({group_name}-PLS): {len(X)} samples, {len(valid_features)} features")
        r2, rmse, y_test, y_pred = train_pls_model(X, y, pls_params, return_true_pred=True)

        if len(y_test) == 0 or np.isnan(r2):
            print(f"   ⚠️ No valid data for {period_name}, skipping")
            group_data['period_r2'].append(np.nan)
            group_data['period_rmse'].append(np.nan)
            continue

        group_data['y_true'].extend(y_test)
        group_data['y_pred'].extend(y_pred)
        group_data['labels'].extend([config.GROUP_LABELS[i]] * len(y_test))
        group_data['period_r2'].append(r2)
        group_data['period_rmse'].append(rmse)
        group_data['valid_count'] += 1
        print(f"   ✅ {period_name} completed: R²={r2:.4f}, RMSE={rmse:.4f}")

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

def plot_single_feature_group_pls_visualization(group_name, group_data):
    """Plot PLS Origin-style fitting curve for single feature group (pls_ prefix)"""
    config = ExperimentConfig()
    group_dir = os.path.join(SINGLE_GROUP_PLOTS_DIR, group_name.replace('/', '_').replace(':', ''))
    save_name = f"pls_{group_name}_fitting_plot".replace('/', '_').replace(':', '')

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
    ax.set_title(f'PLS Fitting Results - {group_name}', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    # Save
    plt.tight_layout()
    save_path = os.path.join(group_dir, f"{save_name}.png")
    plt.savefig(save_path, dpi=config.FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f"   🖼️  PLS visualization chart for feature group {group_name} saved: {save_path}")
    plt.close()

# ==============================================
# 5. Data Processing Module (Consistent with RF, PLS Exclusive Adaptive Principal Component Setting)
# ==============================================
def load_and_preprocess_data():
    """Data loading and preprocessing (consistent with RF process, PLS exclusive adaptive principal component)"""
    print(f"\n📥 Loading Data: {ExperimentConfig.DATA_PATH}")
    try:
        df = pd.read_excel(ExperimentConfig.DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Data file not found: {ExperimentConfig.DATA_PATH}")

    # Verify features and target variable
    missing_features = [col for col in all_features if col not in df.columns]
    if missing_features:
        print(f"⚠️ Some features in all_features are missing: {missing_features}, these invalid features will be skipped")

    if ExperimentConfig.TARGET_COL not in df.columns:
        raise ValueError(f"❌ Target variable missing: {ExperimentConfig.TARGET_COL}")

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
        print("Filled all missing values with mean")

    # PLS Exclusive: Dynamically set principal component number range
    n_samples, n_features = X.shape
    max_n_components = min(n_samples - 1, n_features, 49)
    ExperimentConfig.N_COMPONENTS = np.arange(2, max_n_components + 1, 1)
    print(f"Adaptive PLS parameter grid: {len(ExperimentConfig.N_COMPONENTS)} points for n_components, {len(ExperimentConfig.SCALING)} values for scaling, {len(ExperimentConfig.MAX_ITER)} points for max_iter")

    return df, X, y

# ==============================================
# 6. PLS Grid Search (Exclusive Logic, Consistent with RF Process)
# ==============================================
def run_pls_grid_search(X, y):
    """Execute PLS grid search (replacing RF grid search, maintaining consistent process)"""
    X_scaled = StandardScaler().fit_transform(X) if ExperimentConfig.SCALING[0] else X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values,
        test_size=ExperimentConfig.TEST_SIZE,
        random_state=ExperimentConfig.RANDOM_STATE,
        shuffle=True
    )

    X_train = to_numpy(X_train)
    X_test = to_numpy(X_test)
    y_train = to_numpy(y_train).reshape(-1, 1)
    y_test = to_numpy(y_test).reshape(-1, 1)

    total_combinations = len(ExperimentConfig.N_COMPONENTS) * \
                         len(ExperimentConfig.SCALING) * \
                         len(ExperimentConfig.MAX_ITER)
    print(f"\n🚀 Starting PLS Multi-Parameter Grid Search (Total Combinations: {total_combinations})")

    search_results = []
    for idx, (n_comp, scaling, max_iter) in enumerate(itertools.product(
            ExperimentConfig.N_COMPONENTS,
            ExperimentConfig.SCALING,
            ExperimentConfig.MAX_ITER
    ), 1):
        if idx % 50 == 0 or idx == total_combinations:
            print(f"   Progress: {idx}/{total_combinations} (Current: n_comp={n_comp}, scaling={scaling}, max_iter={max_iter})")

        try:
            current_n_comp = get_valid_n_components(X_train, y_train, n_comp)
            if current_n_comp != n_comp:
                continue

            if USE_CUML:
                # GPU version PLS
                X_train_dev = to_device(X_train)
                y_train_dev = to_device(y_train)
                X_test_dev = to_device(X_test)

                model = cuPLS(
                    n_components=int(current_n_comp),
                    scale=scaling,
                    max_iter=int(max_iter),
                    tol=1e-06
                )
                model.fit(X_train_dev, y_train_dev)
                y_pred = to_numpy(model.predict(X_test_dev))
            else:
                # CPU version PLS
                model = PLSRegression(
                    n_components=int(current_n_comp),
                    scale=scaling,
                    max_iter=int(max_iter),
                    tol=1e-06,
                    copy=True
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            rmse = calc_rmse(y_test, y_pred)
            search_results.append({
                'n_components': int(current_n_comp),
                'scaling': scaling,
                'max_iter': int(max_iter),
                'rmse': rmse
            })

        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            print(f"   ⚠️ Combination {idx} failed: {error_msg}")
            continue

    # Save PLS grid results (pls_ prefix)
    results_df = pd.DataFrame(search_results)
    full_save_path = os.path.join(ExperimentConfig.OUTPUT_ROOT, "pls_full_search_results.csv")
    results_df.to_csv(full_save_path, index=False, encoding='utf-8-sig')

    # Statistics of valid results
    valid_count = len(results_df)
    print(f"✅ PLS Grid Search Completed: {valid_count}/{total_combinations} valid results")
    print(f"📄 PLS Grid Results Saved: {full_save_path}")

    # Find optimal parameters
    valid_results = results_df.dropna(subset=['rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['rmse'].idxmin()
        best_params = {
            'n_components': int(valid_results.loc[best_idx, 'n_components']),
            'scaling': valid_results.loc[best_idx, 'scaling'],
            'max_iter': int(valid_results.loc[best_idx, 'max_iter']),
            'best_rmse': valid_results.loc[best_idx, 'rmse']
        }
        print(f"\n🏆 PLS Optimal Parameters: n_components={best_params['n_components']}, scaling={best_params['scaling']}, max_iter={best_params['max_iter']}")
        print(f"   Optimal RMSE: {best_params['best_rmse']:.4f}")
    else:
        best_params = {'n_components': 2, 'scaling': True, 'max_iter': 50, 'best_rmse': np.nan}
        print(f"\n⚠️ No valid PLS grid results, using default parameters")

    return results_df, best_params

# ==============================================
# 7. PLS 3D Surface Plotting (Exclusive Style, Consistent with RF Process)
# ==============================================
def plot_pls_3d_surface(results_df):
    """Plot PLS 3D performance surface plot (pls_ prefix, exclusive style)"""
    valid_df = results_df.dropna(subset=['rmse']).copy()
    if len(valid_df) < 10:
        print("⚠️ Insufficient data, skipping PLS 3D surface plot")
        return

    # Calculate average RMSE by grouping with n_components and scaling
    grouped = valid_df.groupby(['n_components', 'scaling'])['rmse'].mean().reset_index()
    x_data = grouped['n_components'].values
    y_data = grouped['scaling'].values
    z_data = grouped['rmse'].values

    # Save surface table (pls_ prefix)
    save_surface_table(
        x_data=x_data,
        y_data=y_data,
        z_data=z_data,
        x_name='n_components',
        y_name='scaling',
        z_name='rmse',
        save_name='table_1_components_vs_scaling'
    )

    # Generate regular grid
    x_mesh, y_mesh, z_mesh = create_surface_grid(x_data, y_data, z_data, ExperimentConfig.INTERP_GRID_SIZE)

    # Create 3D figure
    fig = plt.figure(figsize=ExperimentConfig.FIG_SIZE)
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D surface (PLS exclusive style)
    surf = ax.plot_surface(
        x_mesh, y_mesh, z_mesh,
        cmap=ExperimentConfig.SURFACE_CMAP,
        alpha=0.8, edgecolor='k', linewidth=0.2
    )

    # Mark optimal parameters
    best_idx = np.argmin(z_data)
    best_comp = x_data[best_idx]
    best_scaling = bool_to_int(y_data[best_idx])
    best_rmse = z_data[best_idx]

    ax.scatter(
        [best_comp], [best_scaling], [best_rmse],
        s=500, c='red', marker='*',
        label=f'Optimal Parameters\nNumber of Components: {int(best_comp)}\nScaling: {best_scaling} (1=Yes,0=No)\nRMSE: {best_rmse:.4f}',
        zorder=5
    )

    ax.text(
        best_comp, best_scaling, best_rmse + 0.02,
        f'({int(best_comp)}, {best_scaling})',
        color='red', fontsize=12, fontweight='bold'
    )

    # Chart style settings
    ax.set_xlabel('Number of Components (n_components)', fontsize=14, labelpad=15)
    ax.set_ylabel('Scaling (1=Yes, 0=No)', fontsize=14, labelpad=15)
    ax.set_zlabel('RMSE (Lower is Better)', fontsize=14, labelpad=15)
    ax.set_title('PLS Performance Surface: Number of Components vs Scaling', fontsize=16, pad=20, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.view_init(elev=25, azim=45)

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
    cbar.set_label('RMSE', fontsize=12, labelpad=10)

    # Save image (pls_ prefix)
    save_path = os.path.join(PLOTS_DIR, "pls_3d_rmse_surface.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=ExperimentConfig.FIG_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ PLS 3D Surface Plot Saved: {save_path}")

# ==============================================
# 8. Main Function (Fully Consistent with RF Process, PLS Exclusive Logic)
# ==============================================
def main():
    print("=" * 70)
    print("📌 PLS Regression Complete Toolkit (GPU/CPU Adaptive, pls_ Prefix for File Naming)")
    print("=" * 70)

    try:
        # Step 1: Data Preprocessing
        df, X, y = load_and_preprocess_data()

        # Step 2: PLS Grid Search
        pls_results_df, pls_best_params = run_pls_grid_search(X, y)

        # Step 3: Plot PLS 3D Surface Plot (pls_ prefix)
        print(f"\n🎨 Starting to Plot PLS 3D Performance Surface Plot")
        plot_pls_3d_surface(pls_results_df)

        # Step 4: Save PLS Grid Data Table (pls_ prefix)
        save_pls_grid_table(pls_results_df, "grid_results")

        # Step 5: Generate PLS Performance Statistics Table (pls_ prefix)
        print(f"\n📋 Starting to Generate PLS Performance Statistics Table (Including Full Growth Period)")
        pls_performance_table = create_pls_performance_table(df, pls_best_params)
        save_pls_performance_table(pls_performance_table)

        # Step 6: All Features-PLS Visualization + Save True/Predicted CSV
        print(f"\n🎨 Starting to Process Origin-style Visualization for All Features-PLS")
        (y_true, y_pred, group_labels,
         group_r2, group_rmse,
         total_r2, total_rmse, overall_a, overall_b) = collect_pls_true_pred_data(df, pls_best_params)

        # Save All Features true/predicted CSV
        if len(y_true) > 0 and len(y_pred) > 0 and not np.isnan(total_r2):
            # Save true/predicted CSV
            save_pls_true_pred_csv(
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
            plot_pls_origin_style_results(
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
            print(f"⚠️ No valid data for All Features-PLS, skipping visualization and CSV saving")

        # Step 7: 6 Major Feature Groups-PLS Independent Visualization + Save Each Feature Group's True/Predicted CSV
        print(f"\n🎨 Starting to Process Independent Fitting Visualization for 6 Major Feature Groups-PLS")
        all_group_pred_data = []

        for group_name, group_features in ExperimentConfig.FEATURE_GROUPS.items():
            if group_name == "All Features":
                continue
            print(f"\n{'=' * 50}")
            print(f"Processing Feature Group: {group_name} (PLS)")
            print(f"{'=' * 50}")
            group_data = collect_single_feature_group_pls_data(df, group_name, group_features, pls_best_params)

            if group_data is not None:
                # Save single feature group true/predicted data CSV
                csv_save_name = f"{group_name.replace(' ', '_').replace('(', '').replace(')', '')}_true_pred_data"
                save_pls_true_pred_csv(
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
                plot_single_feature_group_pls_visualization(group_name, group_data)

                # Summarize into total data
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

        # Save summary CSV of all feature groups' true/predicted data
        if all_group_pred_data:
            total_group_pred_df = pd.concat(all_group_pred_data, ignore_index=True)
            total_save_path = os.path.join(TABLES_DIR, "pls_all_feature_groups_true_pred_summary.csv")
            total_group_pred_df.to_csv(total_save_path, index=False, encoding='utf-8-sig')
            print(f"\n📄 Summary CSV of All Feature Groups' True/Predicted Data Saved: {total_save_path}")

        # Step 8: Output File List
        print(f"\n🎉 All PLS Processes Executed Successfully!")
        print(f"\n【PLS Output File List】")
        print(f"### Core Output Files (pls_ Prefix) ###")
        print(f"1. PLS Grid Search Results: {os.path.join(ExperimentConfig.OUTPUT_ROOT, 'pls_full_search_results.csv')}")
        print(f"2. PLS 3D Surface Plot: {os.path.join(PLOTS_DIR, 'pls_3d_rmse_surface.png')}")
        print(f"3. PLS Grid Data Table: {os.path.join(TABLES_DIR, 'pls_grid_results.csv')}")
        print(f"4. PLS Surface Data Table: {os.path.join(TABLES_DIR, 'pls_table_1_components_vs_scaling.csv')}")
        print(f"5. PLS Performance Statistics Table (Excel): {os.path.join(TABLES_DIR, 'pls_Performance_Table_With_R2_RMSE_Header.xlsx')}")
        print(f"6. PLS Performance Statistics Table (CSV): {os.path.join(TABLES_DIR, 'pls_Performance_Table_With_R2_RMSE_Header.csv')}")
        print(f"7. PLS All Features True/Predicted Data: {os.path.join(TABLES_DIR, 'pls_all_features_true_pred_data.csv')}")
        print(f"8. PLS Each Feature Group True/Predicted Data: {TABLES_DIR} (File name format: pls_*_true_pred_data.csv)")
        print(f"9. PLS All Features Visualization: {os.path.join(PLOTS_DIR, 'pls_origin_style_fitting_plot_with_all_periods.png')}")
        print(f"10. PLS Feature Group Visualization: {SINGLE_GROUP_PLOTS_DIR}")
        if all_group_pred_data:
            print(f"11. PLS All Feature Groups True/Predicted Summary: {os.path.join(TABLES_DIR, 'pls_all_feature_groups_true_pred_summary.csv')}")

    except Exception as e:
        print(f"\n❌ Execution Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()