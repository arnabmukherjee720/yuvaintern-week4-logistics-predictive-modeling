# YuvaIntern Week 4 – Predictive Modeling and Optimization in Logistics

This project predicts logistics shipment delivery time using a simulated dataset and a Random Forest regression model.

## Files
- `week4_logistics_prediction_simulated.csv` – 500 simulated shipment records.
- `week4_predictive_model.py` – preprocessing, model training, testing, evaluation, cross-validation, and an example prediction.
- `requirements.txt` – required Python packages.

## Prediction target
`Delivery_Days` is the target. Input features include distance, weight, shipment volume, region, traffic level, and weather.

## Run the project
1. Install Python 3.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python week4_predictive_model.py
   ```

The script uses an 80/20 train-test split, a Random Forest Regressor, a mean-prediction baseline, and 5-fold cross-validation. It prints MAE, RMSE, and R-squared metrics.

## Dataset note
The dataset is simulated for internship practice and is not real company shipment data. Results should be validated on real operational data before business use.
