import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

DAMAGE_CLASSES = ['broken_glass', 'broken_lights', 'dents', 'lost_parts', 'punctured', 'scratch', 'torn']
DAMAGE_COST_MULTIPLIERS = {
    'broken_glass': 1.3, 'broken_lights': 1.1, 'dents': 1.0,
    'lost_parts': 1.5, 'punctured': 0.7, 'scratch': 0.6, 'torn': 0.8
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

    # Scale by vehicle value tier — the model was trained on insurance data
    # where vehicle value isn't the dominant feature, so we apply a manual correction
    if vehicle_value < 8000:
        value_scale = 0.60
    elif vehicle_value < 15000:
        value_scale = 0.80
    elif vehicle_value < 25000:
        value_scale = 1.00
    elif vehicle_value < 45000:
        value_scale = 1.30
    elif vehicle_value < 75000:
        value_scale = 1.70
    else:
        value_scale = 2.20

    # Scale by vehicle age — older cars have lower parts availability and higher labour
    if vehicle_age <= 2:
        age_scale = 1.10
    elif vehicle_age <= 5:
        age_scale = 1.00
    elif vehicle_age <= 10:
        age_scale = 0.85
    else:
        age_scale = 0.70

    cost = cost * value_scale * age_scale * multiplier
    cost = max(200, min(cost, 80000))

    return {
        'estimated_cost_usd': round(cost, 2),
        'cost_range_low': round(cost * 0.8, 2),
        'cost_range_high': round(cost * 1.2, 2),
        '_X': X,  # pass through for SHAP
    }


def compute_shap(ml_result: dict, model, top_n: int = 8) -> list[tuple]:
    """Return top_n (label, shap_value) tuples sorted by absolute impact."""
    X = ml_result.get('_X')
    if X is None:
        return []
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)[0]  # shape: (n_features,)
    pairs = list(zip(X.columns.tolist(), sv))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]
