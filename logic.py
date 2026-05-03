import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATASET_PATH = "datasets"
MODEL_PATH = "models"

def load_data(disease):
    path_map = {
        "diabetes": "diabetes.csv",
        "heart": "heart_disease.csv",
        "liver": "liver_disease.csv"
    }
    file_path = os.path.join(DATASET_PATH, path_map[disease])
    df = pd.read_csv(file_path)

    if disease == "diabetes":
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = \
            df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        df.fillna(df.mean(), inplace=True)

    elif disease == "heart":
        df.replace('?', np.nan, inplace=True)
        df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']] = \
            df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']].apply(pd.to_numeric, errors='coerce')
        for col in ['ca', 'thal']:
            df[col].fillna(df[col].mode()[0], inplace=True)
        df['target'] = (df['target'] > 0).astype(int)

    elif disease == "liver":
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df.rename(columns={'Dataset': 'target'}, inplace=True)
        df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

    return df

def get_config():
    return {
        "diabetes": {
            "features": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            "target": "Outcome",
            "model": LogisticRegression(max_iter=1000)
        },
        "heart": {
            "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                         'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                         'ca', 'thal'],
            "target": "target",
            "model": RandomForestClassifier(n_estimators=100, random_state=42)
        },
        "liver": {
            "features": ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                         'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                         'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                         'Albumin_and_Globulin_Ratio'],
            "target": "target",
            "model": LogisticRegression(max_iter=1000)
        }
    }

def train_model(disease):
    df = load_data(disease)
    config = get_config()[disease]
    X = df[config["features"]]
    y = df[config["target"]]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    model = config["model"]
    model.fit(X_train, y_train)

    # Save the model and preprocessing tools
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    joblib.dump(model, f"{MODEL_PATH}/{disease}_model.pkl")
    joblib.dump(imputer, f"{MODEL_PATH}/{disease}_imputer.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}/{disease}_scaler.pkl")

    # Evaluate (optional print)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{disease.capitalize()}] Model trained with accuracy: {acc:.2f}")

def ensure_models_trained():
    for disease in ['diabetes', 'heart', 'liver']:
        model_path = f"{MODEL_PATH}/{disease}_model.pkl"
        if not os.path.exists(model_path):
            print(f"Training model for {disease}...")
            train_model(disease)

def predict_disease_web(disease, input_list):
    model = joblib.load(f"{MODEL_PATH}/{disease}_model.pkl")
    imputer = joblib.load(f"{MODEL_PATH}/{disease}_imputer.pkl")
    scaler = joblib.load(f"{MODEL_PATH}/{disease}_scaler.pkl")

    input_array = np.array(input_list).reshape(1, -1).astype(float)
    input_array = imputer.transform(input_array)
    input_array = scaler.transform(input_array)

    prediction = model.predict(input_array)[0]
    confidence = None
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(input_array)[0][1]

    return prediction, round(confidence * 100, 1) if confidence is not None else None
