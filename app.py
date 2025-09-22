from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
import uuid

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

def train_models(X, y, is_classification):
    """Train three different models and return results"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    if is_classification:
        # Classification models
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_score = accuracy_score(y_test, lr_pred)
        
        models['Logistic Regression'] = (lr, scaler)
        results['Logistic Regression'] = {
            'score': lr_score,
            'metric': 'accuracy',
            'predictions': lr_pred
        }
        
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = accuracy_score(y_test, rf_pred)
        
        models['Random Forest'] = rf
        results['Random Forest'] = {
            'score': rf_score,
            'metric': 'accuracy',
            'predictions': rf_pred
        }
        
        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_score = accuracy_score(y_test, xgb_pred)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = {
            'score': xgb_score,
            'metric': 'accuracy',
            'predictions': xgb_pred
        }
        
    else:
        # Regression models
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_score = r2_score(y_test, lr_pred)
        
        models['Linear Regression'] = (lr, scaler)
        results['Linear Regression'] = {
            'score': lr_score,
            'metric': 'R²',
            'predictions': lr_pred,
            'mse': mean_squared_error(y_test, lr_pred)
        }
        
        # Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)
        
        models['Random Forest'] = rf
        results['Random Forest'] = {
            'score': rf_score,
            'metric': 'R²',
            'predictions': rf_pred,
            'mse': mean_squared_error(y_test, rf_pred)
        }
        
        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_score = r2_score(y_test, xgb_pred)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = {
            'score': xgb_score,
            'metric': 'R²',
            'predictions': xgb_pred,
            'mse': mean_squared_error(y_test, xgb_pred)
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
            columns = df.columns.tolist()
            return render_template('select_target.html', 
                                 columns=columns, 
                                 filename=filename,
                                 data_shape=df.shape,
                                 data_head=df.head().to_html(classes='table table-striped'))
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
        
        return render_template('results.html',
                             results=results,
                             best_model=best_model_name,
                             model_filename=model_filename,
                             target_classes=target_classes,
                             data_info={'shape': df.shape, 'target': target_column})
        
    except Exception as e:
        return f"Error during training: {str(e)}"

@app.route('/download_model/<filename>')
def download_model(filename):
    model_path = os.path.join(app.config['MODELS_FOLDER'], filename)
    return send_file(model_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

# For Vercel
application = app