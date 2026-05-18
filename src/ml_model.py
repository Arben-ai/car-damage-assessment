import pickle
import numpy as np
import pandas as pd
from pathlib import Path

DAMAGE_CLASSES = ['dent', 'scratch', 'crack', 'glass_breakage', 'lamp_breakage', 'tire_flat']
DAMAGE_COST_MULTIPLIERS = {
    'dent': 1.0, 'scratch': 0.6, 'crack': 1.4,
    'glass_breakage': 1.2, 'lamp_breakage': 0.9, 'tire_flat': 0.7
}


def load_artifacts(models_dir: str) -> tuple:
    models_dir = Path(models_dir)
    with open(models_dir / 'xgb_repair_cost.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(models_dir / 'ml_imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    with open(models_dir / 'ml_feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, imputer, feature_names


def predict_cost(
    cv_result: dict,
    vehicle_age: int,
    vehicle_value: int,
    model,
    imputer,
    feature_names: list
) -> dict:
    damage_class = cv_result['damage_class']
    confidence = cv_result['confidence']
    multiplier = DAMAGE_COST_MULTIPLIERS.get(damage_class, 1.0)

    row = {name: 0 for name in feature_names}
    row['VEHICLE_AGE'] = vehicle_age
    row['BLUEBOOK'] = vehicle_value
    row['cv_confidence'] = confidence
    row['cv_damage_multiplier'] = multiplier
    row['VALUE_PER_AGE'] = vehicle_value / (vehicle_age + 1)

    cv_col = f'cv_damage_class_{damage_class}'
    if cv_col in row:
        row[cv_col] = 1

    X = pd.DataFrame([row])[feature_names]
    X = pd.DataFrame(imputer.transform(X), columns=feature_names)

    log_pred = model.predict(X)[0]
    cost = float(np.expm1(log_pred))
    cost = max(200, min(cost, 50000))

    return {
        'estimated_cost_usd': round(cost, 2),
        'cost_range_low': round(cost * 0.8, 2),
        'cost_range_high': round(cost * 1.2, 2),
    }
