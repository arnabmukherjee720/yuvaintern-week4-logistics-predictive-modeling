"""Week 4: Predictive Modeling and Optimization in Logistics Systems.

Run:
    pip install -r requirements.txt
    python week4_predictive_model.py

The project uses a simulated logistics dataset for demonstration.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "week4_logistics_prediction_simulated.csv"
TARGET = "Delivery_Days"


def main():
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    numeric_features = ["Distance_km", "Weight_kg", "Shipment_Volume_units"]
    categorical_features = ["Region", "Traffic_Level", "Weather"]

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()

    print("WEEK 4: LOGISTICS DELIVERY-TIME PREDICTION")
    print(f"Dataset records: {len(data)}")
    print(f"Training records: {len(X_train)} | Testing records: {len(X_test)}")
    print(f"Random Forest MAE: {mae:.3f} days")
    print(f"Random Forest RMSE: {rmse:.3f} days")
    print(f"Random Forest R-squared: {r2:.3f}")
    print(f"5-fold cross-validation MAE: {cv_mae:.3f} days")
    print(f"Baseline MAE: {baseline_mae:.3f} days")

    sample = pd.DataFrame([{
        "Distance_km": 400,
        "Weight_kg": 25.0,
        "Shipment_Volume_units": 20,
        "Region": "North",
        "Traffic_Level": "Medium",
        "Weather": "Clear",
    }])
    estimated_days = model.predict(sample)[0]
    print(f"Example shipment predicted delivery time: {estimated_days:.2f} days")

    print("\nOptimization recommendations:")
    print("1. Prioritize shipments predicted to have long delivery times.")
    print("2. Consider traffic and weather when planning routes and dispatch times.")
    print("3. Use distance, weight, and shipment volume for capacity planning.")
    print("4. Monitor prediction errors and retrain the model with real operational data.")


if __name__ == "__main__":
    main()
