from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
import json
import pickle
import os
import uuid
from eda import (generate_data_profile, create_missing_values_heatmap,
                 create_distribution_plots, create_correlation_heatmap,
                 create_categorical_plots, detect_outliers, create_outlier_plots,
                 analyze_target_relationships)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

def preprocess_data(df, target_column):
    """Basic preprocessing for the dataset"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables in features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Determine if this is classification or regression
    is_classification = False
    target_classes = None
    
    if y.dtype == 'object':
        # String target - definitely classification
        is_classification = True
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        target_classes = le_target.classes_
    elif y.nunique() <= 20 and y.dtype in ['int64', 'int32']:
        # Small number of unique integers - likely classification
        is_classification = True
        target_classes = sorted(y.unique())
    else:
        # Continuous values - regression
        is_classification = False
    
    # Handle missing values (simple approach)
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)  # For any remaining non-numeric columns
    
    return X, y, target_classes, is_classification

def create_learning_curve_plot(model, X, y, model_name, is_classification):
    """Create learning curve visualization"""
    scoring = 'accuracy' if is_classification else 'r2'
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=scoring, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, visible=True)
    ))

    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(type='data', array=val_std, visible=True)
    ))

    fig.update_layout(
        title=f'Learning Curve - {model_name}',
        xaxis_title='Training Set Size',
        yaxis_title=scoring.title(),
        template='plotly_white',
        height=400
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_feature_importance_plot(model, feature_names, model_name):
    """Create feature importance plot for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features

        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in indices],
                y=importances[indices],
                marker_color='lightblue'
            )
        ])

        fig.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Features',
            yaxis_title='Importance',
            template='plotly_white',
            height=400,
            xaxis_tickangle=-45
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return None

def create_confusion_matrix_plot(y_true, y_pred, target_classes, model_name):
    """Create confusion matrix heatmap for classification"""
    cm = confusion_matrix(y_true, y_pred)

    if target_classes is not None and len(target_classes) <= 20:  # Only for reasonable number of classes
        labels = [str(cls) for cls in target_classes]
    else:
        labels = [str(i) for i in range(len(np.unique(y_true)))]

    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True
    )

    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        template='plotly_white',
        height=400
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_residual_plot(y_true, y_pred, model_name):
    """Create residual plot for regression"""
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Predicted vs Actual', 'Residual Plot')
    )

    # Predicted vs Actual
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=y_true,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ),
        row=1, col=1
    )

    # Perfect prediction line
    min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Residuals
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', opacity=0.6)
        ),
        row=1, col=2
    )

    # Zero line for residuals
    fig.add_trace(
        go.Scatter(
            x=[min(y_pred), max(y_pred)], y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
    fig.update_yaxes(title_text="Actual Values", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)

    fig.update_layout(
        title=f'Regression Analysis - {model_name}',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_model_comparison_plot(results, is_classification):
    """Create model comparison bar chart"""
    model_names = list(results.keys())
    scores = [results[name]['score'] for name in model_names]

    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=scores,
            marker_color=['gold' if score == max(scores) else 'lightblue' for score in scores],
            text=[f'{score:.3f}' for score in scores],
            textposition='auto'
        )
    ])

    metric = 'Accuracy' if is_classification else 'R² Score'
    fig.update_layout(
        title=f'Model Performance Comparison ({metric})',
        xaxis_title='Models',
        yaxis_title=metric,
        template='plotly_white',
        height=400
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def train_models(X, y, is_classification):
    """Train three different models with hyperparameter tuning and return results"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    if is_classification:
        # Classification models

        # Logistic Regression (with basic hyperparameter tuning)
        lr_params = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
        lr = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_score = accuracy_score(y_test, lr_pred)

        models['Logistic Regression'] = (lr.best_estimator_, scaler)
        results['Logistic Regression'] = {
            'score': lr_score,
            'metric': 'accuracy',
            'predictions': lr_pred,
            'best_params': lr.best_params_
        }

        # Random Forest Classifier
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = accuracy_score(y_test, rf_pred)

        models['Random Forest'] = rf.best_estimator_
        results['Random Forest'] = {
            'score': rf_score,
            'metric': 'accuracy',
            'predictions': rf_pred,
            'best_params': rf.best_params_
        }

        # XGBoost Classifier
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_model = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            xgb_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_score = accuracy_score(y_test, xgb_pred)

        models['XGBoost'] = xgb_model.best_estimator_
        results['XGBoost'] = {
            'score': xgb_score,
            'metric': 'accuracy',
            'predictions': xgb_pred,
            'best_params': xgb_model.best_params_
        }

    else:
        # Regression models

        # Linear Regression (with basic regularization options)
        lr_params = {
            'fit_intercept': [True, False]
        }
        lr = GridSearchCV(
            LinearRegression(),
            lr_params,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)

        models['Linear Regression'] = (lr.best_estimator_, scaler)
        results['Linear Regression'] = {
            'score': lr_score,
            'metric': 'R²',
            'predictions': lr_pred,
            'mse': mean_squared_error(y_test, lr_pred),
            'best_params': lr.best_params_
        }

        # Random Forest Regressor
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)

        models['Random Forest'] = rf.best_estimator_
        results['Random Forest'] = {
            'score': rf_score,
            'metric': 'R²',
            'predictions': rf_pred,
            'mse': mean_squared_error(y_test, rf_pred),
            'best_params': rf.best_params_
        }

        # XGBoost Regressor
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_model = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params,
            cv=3,
            scoring='r2',
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_score = r2_score(y_test, xgb_pred)

        models['XGBoost'] = xgb_model.best_estimator_
        results['XGBoost'] = {
            'score': xgb_score,
            'metric': 'R²',
            'predictions': xgb_pred,
            'mse': mean_squared_error(y_test, xgb_pred),
            'best_params': xgb_model.best_params_
        }

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['score'])

    return models, results, best_model_name, y_test

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.csv'):
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and analyze the CSV
        try:
            df = pd.read_csv(filepath)
            # Redirect to data exploration page
            return redirect(url_for('explore_data', filename=filename))
        except Exception as e:
            return f"Error reading CSV: {str(e)}"
    
    return "Please upload a CSV file"

@app.route('/train', methods=['POST'])
def train():
    filename = request.form['filename']
    target_column = request.form['target_column']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Load data
        df = pd.read_csv(filepath)
        
        # Preprocess
        X, y, target_classes, is_classification = preprocess_data(df, target_column)
        
        # Train models
        models, results, best_model_name, y_test = train_models(X, y, is_classification)

        # Save the best model
        best_model = models[best_model_name]
        model_filename = f"best_model_{filename.split('_')[0]}.pkl"
        model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # Generate visualizations
        visualizations = {}
        feature_names = X.columns.tolist()

        # Model comparison plot
        visualizations['comparison'] = create_model_comparison_plot(results, is_classification)

        # Generate visualizations for each model
        for model_name, model in models.items():
            model_viz = {}

            # Get the actual model (handle tuples for scaled models)
            actual_model = model[0] if isinstance(model, tuple) else model

            # Learning curve
            try:
                model_viz['learning_curve'] = create_learning_curve_plot(
                    actual_model, X, y, model_name, is_classification
                )
            except Exception as e:
                print(f"Error creating learning curve for {model_name}: {e}")
                model_viz['learning_curve'] = None

            # Feature importance (for tree-based models)
            model_viz['feature_importance'] = create_feature_importance_plot(
                actual_model, feature_names, model_name
            )

            # Model-specific predictions visualization
            y_pred = results[model_name]['predictions']

            if is_classification:
                # Confusion matrix for classification
                model_viz['confusion_matrix'] = create_confusion_matrix_plot(
                    y_test, y_pred, target_classes, model_name
                )
            else:
                # Residual plot for regression
                model_viz['residual_plot'] = create_residual_plot(
                    y_test, y_pred, model_name
                )

            visualizations[model_name] = model_viz

        return render_template('results.html',
                             results=results,
                             best_model=best_model_name,
                             model_filename=model_filename,
                             target_classes=target_classes,
                             data_info={'shape': df.shape, 'target': target_column},
                             visualizations=visualizations,
                             is_classification=is_classification)
        
    except Exception as e:
        return f"Error during training: {str(e)}"

@app.route('/explore/<filename>')
def explore_data(filename):
    """Data exploration and profiling page"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Load data
        df = pd.read_csv(filepath)

        # Generate data profile
        profile = generate_data_profile(df)

        # Create visualizations
        visualizations = {}

        # Missing values heatmap
        visualizations['missing_heatmap'] = create_missing_values_heatmap(df)

        # Distribution plots
        visualizations['distributions'] = create_distribution_plots(df)

        # Correlation heatmap
        visualizations['correlation'] = create_correlation_heatmap(df)

        # Categorical plots
        visualizations['categorical'] = create_categorical_plots(df)

        # Outlier detection
        outliers_info = detect_outliers(df)
        visualizations['outliers'] = create_outlier_plots(df)

        return render_template('explore.html',
                             filename=filename,
                             profile=profile,
                             outliers_info=outliers_info,
                             visualizations=visualizations,
                             data_sample=df.head(10).to_html(classes='table table-striped table-sm'))

    except Exception as e:
        return f"Error during data exploration: {str(e)}"

@app.route('/select_target/<filename>')
def select_target(filename):
    """Target selection page after data exploration"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        df = pd.read_csv(filepath)
        columns = df.columns.tolist()
        return render_template('select_target.html',
                             columns=columns,
                             filename=filename,
                             data_shape=df.shape,
                             data_head=df.head().to_html(classes='table table-striped'),
                             from_explore=True)
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

@app.route('/download_model/<filename>')
def download_model(filename):
    model_path = os.path.join(app.config['MODELS_FOLDER'], filename)
    return send_file(model_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

# For Vercel
application = app