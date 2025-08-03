import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Setup folders
base_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_path, 'data')
model_dir = os.path.join(base_path, 'models')
os.makedirs(model_dir, exist_ok=True)

# Diameter and targets
diameters = ['10', '12', '16']
targets = ['QUALITY1', 'QUALITY2']

for d in diameters:
    file_path = os.path.join(data_dir, f'Diameter_{d}.xlsx')
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        continue

    print(f"📄 Reading data from Diameter_{d}.xlsx...")
    df = pd.read_excel(file_path)

    # DateTime feature engineering
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
        df['HOUR'] = df['DATE_TIME'].dt.hour

    # Encode categorical GRADE
    if 'GRADE' in df.columns:
        df['GRADE'] = LabelEncoder().fit_transform(df['GRADE'].astype(str))

    # One-hot encode other categorical variables if needed
    df = pd.get_dummies(df)

    for target in targets:
        if target not in df.columns:
            print(f"⚠️ Skipping {target} for diameter {d} (not in data)")
            continue

        print(f"🔧 Training model for {target} on diameter {d}...")

        # Prepare data
        temp_df = df.dropna(subset=[target])
        y = temp_df[target]
        X = temp_df.drop(columns=['ID', 'DATE_TIME', 'QUALITY1', 'QUALITY2'], errors='ignore')

        # Clean missing values
        X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X), columns=X.columns)
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_filename = f"{target.lower()}_d{d}.pkl"
        model_path = os.path.join(model_dir, model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"✅ Saved model: {model_filename}")

print("🎉 All models trained and saved!")
