import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import numpy as np
import re
import warnings
import sys

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
sys.path.append(project_root_dir)

from config.feature_definitions import all_features, feature_types
warnings.filterwarnings('ignore')

# =========================
# CUDA / CuPy Adaptive Configuration
# =========================
try:
    import cupy as cp

    _gpu_ok = True
    try:
        _ = cp.zeros((1,))
        _gpu_count = cp.cuda.runtime.getDeviceCount()
        _gpu_ok = _gpu_count > 0
        cp.cuda.set_pinned_memory_allocator(None)
    except Exception:
        _gpu_ok = False
        _gpu_count = 0
except Exception:
    cp = None
    _gpu_ok = False
    _gpu_count = 0


def _xp():
    """Return computation backend: use CuPy if CUDA is available, otherwise NumPy."""
    return cp if _gpu_ok else np


def _to_xp(arr):
    xp = _xp()
    if xp is np:
        if isinstance(arr, np.ndarray):
            return arr
        elif cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        else:
            return np.asarray(arr)
    else:
        if cp is not None and isinstance(arr, cp.ndarray):
            return arr
        elif isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        else:
            return cp.asarray(arr)


def _to_numpy(arr):
    """Ensure to return NumPy array for external use (pandas/plotting/saving)."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _maybe_print_backend(prefix=""):
    if _gpu_ok:
        print(f"{prefix}✅ Using CUDA acceleration (CuPy), number of detected GPUs: {_gpu_count}")
    else:
        print(f"{prefix}⚠️ No available CUDA detected, falling back to NumPy (CPU) execution")


_maybe_print_backend()

DATA_PATH = '../resource/data_all.xlsx'
TARGET_COL = 'LAI'
SIGNIFICANCE_LEVEL = 0.05
OUTPUT_ROOT_DIR = './correlation_analysis'

# ----------------- GPU-Accelerated Correlation Calculation Function -----------------
def calculate_pearson_correlation_gpu(df, features, target_col=TARGET_COL):
    """GPU-accelerated Pearson correlation calculation."""
    xp = _xp()
    results = []

    for feature in features:
        if feature not in df.columns:
            continue

        # Get valid data and convert to GPU array
        valid_data = df[[feature, target_col]].dropna()
        if len(valid_data) < 2:
            continue

        x_data = _to_xp(valid_data[feature].values)
        y_data = _to_xp(valid_data[target_col].values)

        # Calculate correlation coefficient
        x_mean = xp.mean(x_data)
        y_mean = xp.mean(y_data)
        numerator = xp.sum((x_data - x_mean) * (y_data - y_mean))
        denominator = xp.sqrt(xp.sum((x_data - x_mean) ** 2) * xp.sum((y_data - y_mean) ** 2))

        if float(_to_numpy(denominator)) == 0:
            corr = 0.0
        else:
            corr = float(_to_numpy(numerator / denominator))

        # Calculate p-value (using CPU)
        x_np = _to_numpy(x_data)
        y_np = _to_numpy(y_data)
        _, p_value = stats.pearsonr(x_np, y_np)

        results.append({
            'Feature': feature,
            'Correlation_Coefficient': corr,
            'P_Value': p_value,
            'Significance': p_value < SIGNIFICANCE_LEVEL,
            'Feature_Type': feature_types.get(feature, 'Unknown_Type')
        })

    correlation_df = pd.DataFrame(results).sort_values(
        by='Correlation_Coefficient',
        key=lambda x: abs(x),
        ascending=False
    )
    return correlation_df


# ----------------- Plotting Functions -----------------
def clean_filename(filename):
    """Clean illegal characters in filename and replace with underscores."""
    illegal_chars = r'[<>:"/\\|?*()（）]'
    cleaned = re.sub(illegal_chars, '_', filename)
    cleaned = cleaned.strip('_')
    return cleaned


def save_heatmap(df, features, title, folder):
    """Save correlation heatmap and corresponding data table."""
    selected_cols = [col for col in features + [TARGET_COL] if col in df.columns]
    corr_matrix = df[selected_cols].dropna().corr()

    # Save data table
    corr_matrix.to_excel(os.path.join(folder, 'heatmap_table.xlsx'))

    # Save heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show lower triangle only
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        mask=mask,
        cbar_kws={"shrink": .8}
    )
    plt.title(f'{title} - Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'heatmap.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    return corr_matrix


def save_barplot(corr_df, title, folder):
    """Save correlation bar plot and corresponding data table."""
    colors_map = {
        'Vegetation Index': '#1f77b4',
        'Color Index': '#ff7f0e',
        'Coverage Index': '#FAF9DE',
        'Texture Feature': '#2ca02c',
        'Meteorological Factor': '#d62728'
    }
    colors = [colors_map.get(ft, 'gray') for ft in corr_df['Feature_Type']]

    plt.figure(figsize=(12, 10))
    bars = plt.barh(
        corr_df['Feature'],
        corr_df['Correlation_Coefficient'],
        color=colors,
        alpha=0.8
    )

    # Add value labels
    for bar, corr in zip(bars, corr_df['Correlation_Coefficient']):
        plt.text(
            corr + 0.01 if corr >= 0 else corr - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{corr:.3f}',
            va='center',
            ha='left' if corr >= 0 else 'right',
            fontsize=9,
            fontweight='bold'
        )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=ftype, alpha=0.8)
        for ftype, color in colors_map.items()
        if ftype in corr_df['Feature_Type'].values
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.title(f'{title} - Full Feature Correlation Bar Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'barplot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save data table
    corr_df.to_excel(
        os.path.join(folder, 'barplot_table.xlsx'),
        index=False
    )


def save_type_mean_plot(corr_df, title, folder):
    """Save feature type mean correlation plot and corresponding data table."""
    type_corr = corr_df.groupby('Feature_Type')['Correlation_Coefficient'].agg(
        ['mean', 'std', 'count']
    ).sort_values(
        by='mean',
        key=lambda x: abs(x),
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.barh(
        type_corr.index,
        type_corr['mean'],
        xerr=type_corr['std'],
        color=colors[:len(type_corr)],
        alpha=0.8,
        capsize=5
    )

    # Add value labels
    for i, (idx, row) in enumerate(type_corr.iterrows()):
        plt.text(
            row['mean'] + 0.01 if row['mean'] >= 0 else row['mean'] - 0.01,
            i,
            f'{row["mean"]:.3f} ± {row["std"]:.3f}',
            va='center',
            ha='left' if row['mean'] >= 0 else 'right',
            fontsize=11,
            fontweight='bold'
        )

    plt.title(f'{title} - Mean Correlation by Feature Type', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Correlation Coefficient ± Standard Deviation', fontsize=14)
    plt.ylabel('Feature Type', fontsize=14)
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'type_mean_plot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save data table
    type_corr.to_excel(os.path.join(folder, 'type_mean_table.xlsx'))


def save_pairplot(df, features, title, folder):
    """Save pairwise scatter matrix and corresponding correlation table."""
    # Limit number of features for readability
    selected_cols = [col for col in features[:8] + [TARGET_COL] if col in df.columns]

    # Save correlation table
    corr_matrix = df[selected_cols].dropna().corr()
    corr_matrix.to_excel(os.path.join(folder, 'pairplot_corr_table.xlsx'))

    # Create pair grid
    g = sns.PairGrid(df[selected_cols].dropna())
    g.map_upper(sns.scatterplot, alpha=0.6, s=20)
    g.map_lower(sns.scatterplot, alpha=0.6, s=20)
    g.map_diag(plt.hist, alpha=0.7)

    # Add correlation coefficients to upper triangle
    for i in range(len(selected_cols)):
        for j in range(i + 1, len(selected_cols)):
            corr_val = corr_matrix.iloc[i, j]
            g.axes[i, j].annotate(
                f'r = {corr_val:.2f}',
                xy=(0.5, 0.5),
                xycoords='axes fraction',
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )

    plt.suptitle(f'{title} - Pairwise Scatter Matrix', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'pairplot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def save_histogram_corr(corr_df, folder):
    """Save correlation coefficient distribution histogram and statistics table."""
    plt.figure(figsize=(10, 6))
    sns.histplot(
        corr_df['Correlation_Coefficient'],
        bins=20,
        kde=True,
        color='skyblue',
        alpha=0.7,
        edgecolor='black'
    )

    # Add statistical information
    mean_corr = corr_df['Correlation_Coefficient'].mean()
    median_corr = corr_df['Correlation_Coefficient'].median()
    plt.axvline(mean_corr, color='red', linestyle='--', label=f'Mean: {mean_corr:.3f}')
    plt.axvline(median_corr, color='green', linestyle='--', label=f'Median: {median_corr:.3f}')

    plt.title('Distribution Histogram of Correlation Coefficients', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'correlation_histogram.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save statistics table
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Standard_Deviation', 'Minimum', 'Maximum', 'Num_Significant_Features', 'Total_Features'],
        'Value': [
            mean_corr,
            median_corr,
            corr_df['Correlation_Coefficient'].std(),
            corr_df['Correlation_Coefficient'].min(),
            corr_df['Correlation_Coefficient'].max(),
            corr_df['Significance'].sum(),
            len(corr_df)
        ]
    })
    stats_df.to_excel(
        os.path.join(folder, 'correlation_histogram_stats.xlsx'),
        index=False
    )
    corr_df.to_excel(
        os.path.join(folder, 'correlation_histogram_table.xlsx'),
        index=False
    )


def save_scatter_fit(df, features, folder):
    """Save scatter plot with regression fit and regression parameters table."""
    scatter_data = []

    for feature in features:
        if feature not in df.columns:
            continue

        cleaned_feature = clean_filename(feature)
        valid_data = df[[feature, TARGET_COL]].dropna()

        if len(valid_data) < 2:
            continue

        # Calculate regression parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data[feature], valid_data[TARGET_COL]
        )

        # Save regression data
        scatter_data.append({
            'Feature': feature,
            'Slope': slope,
            'Intercept': intercept,
            'R_Squared': r_value ** 2,
            'P_Value': p_value,
            'Standard_Error': std_err
        })

        # Create scatter plot with regression line
        plt.figure(figsize=(8, 6))
        sns.regplot(
            x=feature,
            y=TARGET_COL,
            data=valid_data,
            scatter_kws={'alpha': 0.6, 's': 30},
            line_kws={'color': 'red', 'linewidth': 2}
        )

        # Add regression equation and R²
        equation = f'y = {slope:.3f}x + {intercept:.3f}'
        plt.text(
            0.05, 0.95,
            f'{equation}\nR² = {r_value ** 2:.3f}\np = {p_value:.3e}',
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.title(f'{feature} vs {TARGET_COL} Regression Plot', fontsize=14, fontweight='bold')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel(TARGET_COL, fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(folder, f'scatter_fit_{cleaned_feature}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # Save all regression parameters
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        scatter_df.to_excel(
            os.path.join(folder, 'all_regression_params.xlsx'),
            index=False
        )


def save_top_features_plot(corr_df, title, folder, top_n=10):
    """Save top N positive and negative correlation features plot and table."""
    top_pos = corr_df.nlargest(top_n, 'Correlation_Coefficient')
    top_neg = corr_df.nsmallest(top_n, 'Correlation_Coefficient')

    plt.figure(figsize=(12, 8))

    # Positive correlation features
    bars1 = plt.barh(
        range(len(top_pos)),
        top_pos['Correlation_Coefficient'],
        color='#ff7f0e',
        alpha=0.8,
        label='Positive Correlation'
    )

    # Negative correlation features
    bars2 = plt.barh(
        range(len(top_pos), len(top_pos) + len(top_neg)),
        top_neg['Correlation_Coefficient'],
        color='#1f77b4',
        alpha=0.8,
        label='Negative Correlation'
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01 if width >= 0 else width - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                va='center',
                ha='left' if width >= 0 else 'right',
                fontsize=10,
                fontweight='bold'
            )

    # Set y-axis labels
    all_labels = list(top_pos['Feature']) + list(top_neg['Feature'])
    plt.yticks(range(len(all_labels)), all_labels)

    plt.title(f'{title} - Top {top_n} Positive and Negative Correlation Features', fontsize=16, fontweight='bold')
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.axvline(0, color='black', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'top_features_plot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save data table
    top_features_df = pd.concat([top_pos, top_neg])
    top_features_df.to_excel(
        os.path.join(folder, 'top_features_table.xlsx'),
        index=False
    )


def save_significance_plot(corr_df, title, folder):
    """Save significance analysis plot and corresponding data table."""
    sig_df = corr_df.groupby('Feature_Type')['Significance'].agg(['sum', 'count']).reset_index()
    sig_df['Proportion'] = sig_df['sum'] / sig_df['count'] * 100

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(
        sig_df['Feature_Type'],
        sig_df['Proportion'],
        color=colors[:len(sig_df)],
        alpha=0.8
    )

    # Add value labels
    for bar, ratio in zip(bars, sig_df['Proportion']):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{ratio:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.title(f'{title} - Proportion of Significant Features', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Type', fontsize=14)
    plt.ylabel('Proportion of Significant Features (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(folder, 'significance_plot.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # Save data table
    sig_df.to_excel(
        os.path.join(folder, 'significance_table.xlsx'),
        index=False
    )


# ----------------- Main Program -----------------
if __name__ == '__main__':
    # Set font for English display (no Chinese required)
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    # sns.set_style("whitegrid")

    print("Starting to load data...")
    df = pd.read_excel(DATA_PATH)
    print(f"Data loaded successfully, shape: {df.shape}")

    # Growth periods (corresponding to jointing, heading, filling stages)
    periods = {
        'Jointing_Stage': df.iloc[0:60, :],
        'Heading_Stage': df.iloc[60:120, :],
        'Filling_Stage': df.iloc[120:180, :],
        'Overall': df
    }

    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    for period_name, period_df in periods.items():
        print(f"\n=== Processing {period_name} ===")
        period_dir = os.path.join(OUTPUT_ROOT_DIR, period_name)
        os.makedirs(period_dir, exist_ok=True)

        # Calculate Pearson correlation with GPU acceleration
        print("Calculating Pearson correlation...")
        corr_df = calculate_pearson_correlation_gpu(period_df, all_features)
        corr_df.to_excel(
            os.path.join(period_dir, 'full_correlation_table.xlsx'),
            index=False
        )

        # Generate various plots
        print("Generating correlation heatmap...")
        save_heatmap(period_df, all_features, period_name, period_dir)

        print("Generating correlation bar plot...")
        save_barplot(corr_df, period_name, period_dir)

        print("Generating feature type mean correlation plot...")
        save_type_mean_plot(corr_df, period_name, period_dir)

        print("Generating pairwise scatter matrix...")
        save_pairplot(period_df, all_features, period_name, period_dir)

        print("Generating correlation coefficient histogram...")
        save_histogram_corr(corr_df, period_dir)

        print("Generating scatter plots with regression fits...")
        save_scatter_fit(period_df, all_features, period_dir)

        print("Generating top features plot...")
        save_top_features_plot(corr_df, period_name, period_dir, top_n=8)

        print("Generating significance analysis plot...")
        save_significance_plot(corr_df, period_name, period_dir)

        print(f"{period_name} processing completed")

    print(f"\n✅ All plots and corresponding tables have been saved to: {OUTPUT_ROOT_DIR}")