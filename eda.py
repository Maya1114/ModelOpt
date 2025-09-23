import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
import json
from scipy import stats

def generate_data_profile(df):
    """Generate comprehensive data profiling information"""
    profile = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }

    # Basic statistics for numeric columns
    if profile['numeric_columns']:
        profile['numeric_stats'] = df[profile['numeric_columns']].describe().to_dict()

    # Categorical value counts
    if profile['categorical_columns']:
        profile['categorical_stats'] = {}
        for col in profile['categorical_columns']:
            profile['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }

    return profile

def create_missing_values_heatmap(df):
    """Create heatmap showing missing values pattern"""
    missing_data = df.isnull()

    if missing_data.sum().sum() == 0:
        return None

    fig = px.imshow(
        missing_data.T,
        color_continuous_scale=['lightblue', 'red'],
        labels=dict(x="Row Index", y="Columns", color="Missing"),
        title="Missing Values Pattern"
    )

    fig.update_layout(
        height=400,
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_distribution_plots(df, max_cols=6):
    """Create distribution plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return None

    # Limit to first max_cols columns to avoid overcrowding
    cols_to_plot = numeric_cols[:max_cols]
    n_cols = len(cols_to_plot)

    # Calculate subplot layout
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    n_subplot_cols = min(3, n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_subplot_cols,
        subplot_titles=cols_to_plot,
        vertical_spacing=0.1
    )

    for i, col in enumerate(cols_to_plot):
        row = i // 3 + 1
        col_pos = i % 3 + 1

        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                showlegend=False,
                marker_color='lightblue'
            ),
            row=row,
            col=col_pos
        )

    fig.update_layout(
        title="Distribution of Numeric Features",
        height=300 * n_rows,
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        return None

    corr_matrix = numeric_df.corr()

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="Correlation Matrix",
        labels=dict(color="Correlation")
    )

    fig.update_layout(
        height=500,
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_categorical_plots(df, max_cols=4):
    """Create bar plots for categorical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) == 0:
        return None

    cols_to_plot = categorical_cols[:max_cols]
    n_cols = len(cols_to_plot)

    if n_cols == 1:
        n_rows, n_subplot_cols = 1, 1
    else:
        n_rows = (n_cols + 1) // 2  # 2 columns per row
        n_subplot_cols = min(2, n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_subplot_cols,
        subplot_titles=cols_to_plot,
        vertical_spacing=0.15
    )

    for i, col in enumerate(cols_to_plot):
        row = i // 2 + 1
        col_pos = i % 2 + 1

        value_counts = df[col].value_counts().head(10)  # Top 10 values

        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col,
                showlegend=False,
                marker_color='lightgreen'
            ),
            row=row,
            col=col_pos
        )

    fig.update_layout(
        title="Distribution of Categorical Features",
        height=300 * n_rows,
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def detect_outliers(df, method='iqr'):
    """Detect outliers using IQR or Z-score method"""
    numeric_df = df.select_dtypes(include=[np.number])
    outliers_info = {}

    for col in numeric_df.columns:
        outlier_indices = []

        if method == 'iqr':
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = numeric_df[(numeric_df[col] < lower_bound) |
                                       (numeric_df[col] > upper_bound)].index.tolist()

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df[col].dropna()))
            outlier_indices = numeric_df[z_scores > 3].index.tolist()

        outliers_info[col] = {
            'count': len(outlier_indices),
            'percentage': len(outlier_indices) / len(df) * 100,
            'indices': outlier_indices[:10]  # Show first 10 outlier indices
        }

    return outliers_info

def create_outlier_plots(df, max_cols=6):
    """Create box plots to visualize outliers"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return None

    cols_to_plot = numeric_cols[:max_cols]
    n_cols = len(cols_to_plot)

    n_rows = (n_cols + 2) // 3  # 3 columns per row
    n_subplot_cols = min(3, n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_subplot_cols,
        subplot_titles=cols_to_plot,
        vertical_spacing=0.1
    )

    for i, col in enumerate(cols_to_plot):
        row = i // 3 + 1
        col_pos = i % 3 + 1

        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                showlegend=False,
                marker_color='orange'
            ),
            row=row,
            col=col_pos
        )

    fig.update_layout(
        title="Outlier Detection (Box Plots)",
        height=300 * n_rows,
        template='plotly_white'
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def analyze_target_relationships(df, target_column, max_features=8):
    """Analyze relationships between features and target variable"""
    if target_column not in df.columns:
        return None

    target = df[target_column]
    features = df.drop(columns=[target_column])

    # Determine if target is categorical or numeric
    is_target_categorical = target.dtype == 'object' or target.nunique() <= 10

    relationships = {}

    if is_target_categorical:
        # For categorical target, show distributions by target class
        numeric_features = features.select_dtypes(include=[np.number]).columns[:max_features]

        if len(numeric_features) > 0:
            fig = make_subplots(
                rows=(len(numeric_features) + 1) // 2,
                cols=2,
                subplot_titles=numeric_features,
                vertical_spacing=0.1
            )

            for i, feature in enumerate(numeric_features):
                row = i // 2 + 1
                col = i % 2 + 1

                for target_val in target.unique():
                    subset = df[df[target_column] == target_val][feature]

                    fig.add_trace(
                        go.Histogram(
                            x=subset,
                            name=f'{target_column}={target_val}',
                            opacity=0.7,
                            showlegend=(i == 0)  # Only show legend for first subplot
                        ),
                        row=row,
                        col=col
                    )

            fig.update_layout(
                title=f"Feature Distributions by {target_column}",
                height=300 * ((len(numeric_features) + 1) // 2),
                template='plotly_white'
            )

            relationships['categorical_target'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    else:
        # For numeric target, show scatter plots
        numeric_features = features.select_dtypes(include=[np.number]).columns[:max_features]

        if len(numeric_features) > 0:
            fig = make_subplots(
                rows=(len(numeric_features) + 1) // 2,
                cols=2,
                subplot_titles=numeric_features,
                vertical_spacing=0.1
            )

            for i, feature in enumerate(numeric_features):
                row = i // 2 + 1
                col = i % 2 + 1

                fig.add_trace(
                    go.Scatter(
                        x=df[feature],
                        y=target,
                        mode='markers',
                        name=feature,
                        showlegend=False,
                        opacity=0.6
                    ),
                    row=row,
                    col=col
                )

            fig.update_layout(
                title=f"Feature vs {target_column} Relationships",
                height=300 * ((len(numeric_features) + 1) // 2),
                template='plotly_white'
            )

            relationships['numeric_target'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return relationships