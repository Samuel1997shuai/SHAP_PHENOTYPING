import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import Feature Definitions
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
except ImportError as e:
    print(f"Failed to import feature definitions: {e}")
    print("Verify config/feature_definitions.py exists in project root")
    sys.exit(1)

# SHAP Import & Validation
try:
    import shap
    from shap.utils import hclust, hclust_ordering
    from shap.plots import colors

    # Verify all required SHAP components
    required_components = [
        'TreeExplainer', 'LinearExplainer', 'KernelExplainer',
        'summary_plot', 'plots', 'utils', 'Explanation', 'force_plot'
    ]
    missing_components = [comp for comp in required_components if not hasattr(shap, comp)]
    if missing_components:
        raise ImportError(
            f"Missing required SHAP components: {missing_components}. Update SHAP with 'pip install shap --upgrade'"
        )
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"SHAP library import failed: {e}")
    print("Install SHAP first: 'pip install shap --upgrade'")
    SHAP_AVAILABLE = False

# Configure plot settings (SCI journal standard)
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette (consistent with academic standards)
COLOR_PALETTE = {
    'positive': '#FF6B6B',  # Red for positive impact
    'negative': '#4ECDC4',  # Teal for negative impact
    'neutral': '#95A5A6',  # Gray for baseline
    'heatmap': 'viridis',  # Viridis for heatmaps
    'bar': '#3498DB'  # Blue for bar plots
}

# ----------------------
# Dataset & Feature Config
# ----------------------
# Only retain the feature set corresponding to the target feature group
vegetation_features = Vegetation_Index
color_features = Color_Index
texture_features = Texture_Feature
climate_features = Meteorological_Factor
coverage_feature = coverage

# Only retain the Four_Features_Plus_Coverage feature group (the only feature group)
TARGET_FEATURE_GROUP_NAME = 'Four_Features_Plus_Coverage'
target_features = vegetation_features + color_features + texture_features + climate_features + coverage_feature
feature_groups = {
    TARGET_FEATURE_GROUP_NAME: target_features
}

# Only retain the Random Forest algorithm configuration
RF_ALGORITHM = {
    'name': 'Random_Forest',
    'code': 1,
    'model_type': 'tree'  # Tree-based
}

# Global constants
TARGET_COL = 'LAI'  # Target variable (Leaf Area Index)
DATA_PATH = '../resource/data_all.xlsx'
OUTPUT_ROOT = 'analysis_shap_rf'
RANDOM_STATE = 42
TOP_N_FEATURES = 20  # Modified: Display the top 20 features


# ----------------------
# Core Utility Functions
# ----------------------
def create_output_directory(feature_group: str, algorithm: str) -> str:
    """Create output directory (folder name unchanged as required)"""
    dir_name = f"{feature_group}+{algorithm}"
    dir_path = os.path.join(OUTPUT_ROOT, dir_name)
    try:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    except Exception as e:
        print(f"Failed to create directory {dir_path}: {e}")
        sys.exit(1)


def load_and_preprocess_data() -> tuple:
    """Load and preprocess data (handle missing values, scaling, train-test split)"""
    # Load data
    try:
        raw_data = pd.read_excel(DATA_PATH)
        print(f"Data loaded successfully (shape: {raw_data.shape})")
    except Exception as e:
        print(f"Data loading failed: {e}")
        print(f"Verify data path: {DATA_PATH}")
        sys.exit(1)

    # Check required columns (only check features of the target feature group)
    required_cols = target_features + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in raw_data.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # Clean data (remove missing values)
    data = raw_data[required_cols].copy()
    initial_shape = data.shape
    data = data.dropna()
    print(f"Data cleaned: {initial_shape} → {data.shape} (removed missing values)")

    # Split features and target
    X = data[target_features]
    y = data[TARGET_COL]

    # Scale features (retained for compatibility)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=target_features)

    # Train-test split (30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, X_scaled, scaler, target_features


def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                   y_test: pd.Series) -> tuple:
    """Only train the Random Forest model and return evaluation metrics"""
    try:
        # Initialize Random Forest model
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        )

        # Train and predict
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Evaluate
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Print metrics
        print(f"\nRandom Forest Model Performance:")
        print(f"Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.4f}")
        print(f"Test R²: {test_r2:.4f} | Test RMSE: {test_rmse:.4f}")

        return model, train_r2, test_r2, train_rmse, test_rmse, y_train_pred, y_test_pred

    except Exception as e:
        print(f"Random Forest model training failed: {e}")
        sys.exit(1)


def get_shap_explainer(model, model_type: str, X_background: pd.DataFrame):
    """Create SHAP explainer based on model type (only for tree-based RF)"""
    try:
        if model_type == 'tree':
            return shap.TreeExplainer(model, data=X_background)
        else:
            raise ValueError(f"Unsupported model type for RF: {model_type}")
    except Exception as e:
        print(f"SHAP explainer creation failed: {e}")
        return None


def get_most_accurate_sample_idx(y_true: pd.Series, y_pred: np.ndarray) -> int:
    """Get index of sample with smallest prediction error (most accurate)"""
    errors = np.abs(y_pred - y_true.values)
    return np.argmin(errors)


def get_top_n_feature_indices(shap_expl: shap.Explanation) -> np.ndarray:
    """Get the indices of the top N important features (sorted by mean of absolute SHAP values)"""
    feature_importance = np.abs(shap_expl.values).mean(axis=0)
    top_n_indices = np.argsort(feature_importance)[::-1][:TOP_N_FEATURES]
    return top_n_indices


def save_shap_data(shap_expl: shap.Explanation, X: pd.DataFrame, y_true: pd.Series,
                   y_pred: np.ndarray, output_dir: str, algorithm: str):
    """Save SHAP values and related data to files (retain tabular data)"""
    # 1. Full SHAP values (CSV)
    shap_df = pd.DataFrame(shap_expl.values, columns=X.columns)
    shap_df['sample_index'] = X.index
    shap_df['true_value'] = y_true.values
    shap_df['predicted_value'] = y_pred
    shap_df['prediction_error'] = np.abs(y_pred - y_true.values)
    shap_df.to_csv(os.path.join(output_dir, f"shap_values_{algorithm}.csv"), index=False)

    # 2. Feature importance summary (Excel) - Include all features and mark the top 20 features
    top_n_indices = get_top_n_feature_indices(shap_expl)
    top_feature_flags = [1 if i in top_n_indices else 0 for i in range(len(X.columns))]
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.mean(np.abs(shap_expl.values), axis=0),
        'max_shap': np.max(shap_expl.values, axis=0),
        'min_shap': np.min(shap_expl.values, axis=0),
        'median_shap': np.median(shap_expl.values, axis=0),
        'is_top_20': top_feature_flags  # Modified: Mark the top 20 features
    }).sort_values('mean_abs_shap', ascending=False)

    # 3. Model performance
    train_r2 = r2_score(y_train, y_train_pred) if 'y_train' in locals() else None
    test_r2 = r2_score(y_test, y_test_pred) if 'y_test' in locals() else None
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) if 'y_train' in locals() else None
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) if 'y_test' in locals() else None

    performance_df = pd.DataFrame({
        'metric': ['train_r2', 'test_r2', 'train_rmse', 'test_rmse'],
        'value': [train_r2, test_r2, train_rmse, test_rmse]
    })

    # Save to Excel
    with pd.ExcelWriter(os.path.join(output_dir, f"shap_analysis_summary_{algorithm}.xlsx"),
                        engine='openpyxl') as writer:
        importance_df.to_excel(writer, sheet_name='feature_importance', index=False)
        performance_df.to_excel(writer, sheet_name='model_performance', index=False)
        shap_df.head(100).to_excel(writer, sheet_name='top_100_samples', index=False)

    print(f"SHAP data saved for {algorithm} (with top {TOP_N_FEATURES} features marked)")


# ----------------------
# Retained Plot Functions
# ----------------------
def plot_model_builtin_importance(model, features: list, output_dir: str) -> str:
    """1. Model's built-in feature importance plot (RF's feature_importances_) - Top 20 features"""
    plot_path = os.path.join(output_dir, "model_builtin_importance.png")
    try:
        # RF is a tree-based model with feature_importances_ attribute
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:TOP_N_FEATURES]  # Top 20 features
        plt.figure(figsize=(10, 9))  # Appropriately adjust figure height to accommodate 20 features
        plt.barh(range(len(idx)), importances[idx], color=COLOR_PALETTE['bar'])
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.xlabel("Feature Importance")
        plt.gca().invert_yaxis()  # Top feature at top
        plt.savefig(plot_path)
        plt.close()
        print(f"Random Forest built-in importance plot (top {TOP_N_FEATURES} features) saved")
        return plot_path
    except Exception as e:
        print(f"Model importance plot failed: {e}")
        return ""


def plot_shap_summary_beeswarm(shap_expl: shap.Explanation, output_dir: str) -> str:
    """2. SHAP Summary Plot (Beeswarm type) - Original color scheme + Top 20 features"""
    plot_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    try:
        plt.figure(figsize=(12, 11))  # Appropriately adjust figure height to accommodate 20 features
        # Remove color parameter to use SHAP's original color scheme; set max_display to limit to top 20 features
        shap.plots.beeswarm(
            shap_expl,
            order=shap_expl.abs.mean(0),
            max_display=TOP_N_FEATURES,  # Top 20 features
            show=False
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"SHAP summary beeswarm plot (original color, top {TOP_N_FEATURES} features) saved")
        return plot_path
    except Exception as e:
        print(f"SHAP beeswarm plot failed: {e}")
        return ""


def plot_shap_waterfall(shap_expl: shap.Explanation, y_true: pd.Series, y_pred: np.ndarray,
                        output_dir: str) -> str:
    """3. SHAP Waterfall Plot - Single sample explanation (MOST ACCURATE) - Top 20 features"""
    plot_path = os.path.join(output_dir, "shap_waterfall.png")
    try:
        top_n_indices = get_top_n_feature_indices(shap_expl)
        shap_expl_top = shap_expl[:, top_n_indices]

        # Get most accurate sample (smallest prediction error)
        most_accurate_idx = get_most_accurate_sample_idx(y_true, y_pred)

        # Plot waterfall (show top 20 features)
        plt.figure(figsize=(12, 11))  # Appropriately adjust figure height to accommodate 20 features
        shap.plots.waterfall(
            shap_expl_top[most_accurate_idx],
            max_display=TOP_N_FEATURES,  # Top 20 features
            show=False
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"SHAP waterfall plot (most accurate sample, top {TOP_N_FEATURES} features) saved")
        return plot_path, shap_expl_top, most_accurate_idx  # Return parameters for the 4th plot
    except Exception as e:
        print(f"Waterfall plot failed: {e}")
        return "", None, None


def plot_shap_force_plot_matplotlib(explainer, shap_expl_top: shap.Explanation,
                                    most_accurate_idx: int, X_test: pd.DataFrame,
                                    output_dir: str) -> str:
    """4. SHAP Force Plot - Generate static PNG with matplotlib only, display values to 3 decimal places (fixed rc parameter error)"""
    plot_path = os.path.join(output_dir, "shap_force_plot.png")
    try:
        # Get core parameters of a single sample and round to 3 decimal places uniformly (ensure values themselves are 3 decimal places)
        base_value = np.round(explainer.expected_value, 3)
        shap_values = np.round(shap_expl_top.values[most_accurate_idx], 3)
        features = np.round(X_test[shap_expl_top.feature_names].iloc[most_accurate_idx].values, 3)
        feature_names = shap_expl_top.feature_names
        out_name = TARGET_COL

        # Create figure and set size
        plt.figure(figsize=(25, 3))  # Appropriately widen the figure to accommodate 20 features

        # Get current axes and set axis formatter (ensure display to 3 decimal places without invalid rc parameters)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        # Call shap.plots.force to generate static plot with matplotlib mode
        shap.plots.force(
            base_value=base_value,
            shap_values=shap_values,
            features=features,
            feature_names=feature_names,
            out_names=out_name,
            link='identity',
            plot_cmap='RdBu',
            matplotlib=True,
            show=False,
            figsize=(25, 3),
            text_rotation=0,
            contribution_threshold=0.05
        )

        # Save plot
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(
            f"SHAP force plot (matplotlib static PNG, 3 decimal places, top {TOP_N_FEATURES} features) saved successfully")
        return plot_path
    except Exception as e:
        print(f"SHAP force plot (matplotlib) failed: {e}")
        plt.close()
        return ""


# ----------------------
# Main SHAP Analysis Pipeline
# ----------------------
def run_shap_analysis(feature_group: str, features: list, X_train: pd.DataFrame,
                      X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                      y_train_pred: np.ndarray, y_test_pred: np.ndarray) -> dict:
    """Run simplified SHAP analysis (only retained plots)"""
    if not SHAP_AVAILABLE:
        print("SHAP not available; skipping SHAP analysis")
        return {}

    # Get RF algorithm information
    algo_name = RF_ALGORITHM['name']
    algo_type = RF_ALGORITHM['model_type']

    # Create output directory
    output_dir = create_output_directory(feature_group, algo_name)
    plot_info = {}  # Only for recording generated plots

    # Step 1: Train RF model and create SHAP explainer
    model, _, _, _, _, _, _ = train_rf_model(X_train[features], y_train, X_test[features], y_test)
    explainer = get_shap_explainer(model, algo_type, X_train[features])
    if explainer is None:
        print(f"SHAP explainer not created for {algo_name}; skipping analysis")
        return {}

    # Step 2: Compute SHAP values (both numpy array and Explanation object)
    print(f"\nComputing SHAP values for {algo_name}...")
    try:
        shap_vals_np = explainer.shap_values(X_test[features])
        # Handle SHAP value format differences
        if isinstance(shap_vals_np, list) and len(shap_vals_np) == 1:
            shap_vals_np = shap_vals_np[0]
        shap_vals_np = np.array(shap_vals_np)
        if shap_vals_np.ndim == 3:
            shap_vals_np = shap_vals_np.mean(axis=1)  # Handle multi-output

        # Create Explanation object
        shap_expl = shap.Explanation(
            values=shap_vals_np,
            base_values=explainer.expected_value,
            data=X_test[features].values,
            feature_names=features
        )

    except Exception as e:
        print(f"SHAP value computation failed: {e}")
        return {}

    # Step 3: Generate retained plots (top 20 features)
    print(f"\nGenerating retained SHAP plots for {algo_name} (top {TOP_N_FEATURES} features)...")

    # 1. Model built-in feature importance
    plot1 = plot_model_builtin_importance(model, features, output_dir)
    if plot1:
        plot_info[os.path.basename(plot1)] = f"Model built-in feature importance plot (top {TOP_N_FEATURES} features)"

    # 2. SHAP Summary Beeswarm Plot
    plot2 = plot_shap_summary_beeswarm(shap_expl, output_dir)
    if plot2:
        plot_info[os.path.basename(plot2)] = f"SHAP summary beeswarm plot (original color, top {TOP_N_FEATURES} features)"

    # 3. SHAP Waterfall Plot (get additional parameters for the 4th plot)
    plot3, shap_expl_top, most_accurate_idx = plot_shap_waterfall(shap_expl, y_test, y_test_pred, output_dir)
    if plot3:
        plot_info[os.path.basename(plot3)] = f"SHAP waterfall plot (most accurate sample, top {TOP_N_FEATURES} features)"

    # 4. SHAP Force Plot (4th plot, static PNG with matplotlib only, 3 decimal places)
    if shap_expl_top is not None and most_accurate_idx is not None and explainer is not None:
        plot4 = plot_shap_force_plot_matplotlib(explainer, shap_expl_top, most_accurate_idx, X_test, output_dir)
        if plot4:
            plot_info[os.path.basename(plot4)] = f"SHAP Force Plot (matplotlib static plot, 3 decimal places, top {TOP_N_FEATURES} features)"

    # Step 4: Save SHAP data (retain tabular data)
    save_shap_data(shap_expl, X_test[features], y_test, y_test_pred, output_dir, algo_name)

    return plot_info


# ----------------------
# Main Workflow
# ----------------------
def main():
    print("=" * 60)
    print("Simplified SHAP Analysis for Random Forest (RF)")
    print(f"Target Feature Group: {TARGET_FEATURE_GROUP_NAME}")
    print(f"Display Features: Top {TOP_N_FEATURES} Important Features")
    print(
        "Retained Plots: model_builtin_importance, shap_summary_beeswarm, shap_waterfall, shap_force_plot (matplotlib, 3 decimals)")
    print("=" * 60)

    # Load and preprocess data
    X_train_all, X_test_all, y_train, y_test, X_scaled_all, scaler, all_features = load_and_preprocess_data()

    # Create root output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Only process the Four_Features_Plus_Coverage feature group (no loop needed)
    feature_group = TARGET_FEATURE_GROUP_NAME
    features = target_features
    print(f"\n{'=' * 80}")
    print(f"Processing Feature Group: {feature_group}")
    print(f"{'=' * 80}")

    # Filter features (ensure consistency)
    valid_features = [f for f in features if f in X_train_all.columns]
    if len(valid_features) != len(features):
        print(f"Warning: {len(features) - len(valid_features)} features not found in data")
        features = valid_features

    # Split data for current feature group
    X_train = X_train_all[features]
    X_test = X_test_all[features]

    # Train Random Forest base model (for performance metrics)
    model, train_r2, test_r2, train_rmse, test_rmse, y_train_pred, y_test_pred = train_rf_model(
        X_train, y_train, X_test, y_test
    )

    # Run simplified SHAP analysis (only retained plots)
    run_shap_analysis(
        feature_group=feature_group,
        features=features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred
    )

    print(f"\n{'=' * 60}")
    print("Random Forest SHAP Analysis Completed!")
    print(f"All results saved to: {os.path.abspath(OUTPUT_ROOT)}")
    print(f"Target Feature Group Folder: {feature_group}+{RF_ALGORITHM['name']}")
    print(f"Key Settings:")
    print(f"  - Display: Top {TOP_N_FEATURES} features (sorted by mean absolute SHAP)")
    print(f"  - Beeswarm Plot: Original SHAP color scheme")
    print(
        f"  - Retained Plots: model_builtin_importance, shap_summary_beeswarm, shap_waterfall, shap_force_plot (matplotlib)")
    print(f"  - Force Plot: 3 decimal places for all numerical values, static PNG (no HTML)")
    print(f"  - Table data (CSV/Excel) are preserved")
    print("=" * 60)


if __name__ == "__main__":
    main()