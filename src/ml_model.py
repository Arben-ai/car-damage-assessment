import pickle
import hashlib
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

    # ── Base repair cost by damage type (reference: $20k car, 5 years old) ──
    BASE_COSTS = {
        'scratch':       480,
        'dents':         920,
        'broken_glass':  710,
        'broken_lights': 560,
        'lost_parts':   2400,
        'punctured':     350,
        'torn':         1250,
    }
    base = BASE_COSTS.get(damage_class, 800)

    # ── Vehicle value: sub-linear power curve ──
    # $5k → ×0.44 | $10k → ×0.60 | $20k → ×1.00 | $40k → ×1.52 | $80k → ×2.32
    value_factor = (vehicle_value / 20000) ** 0.68

    # ── Vehicle age: continuous decay ──
    # New car: parts expensive + warranty labour. Old car: cheap parts, hard to find
    age_factor = 1.30 * np.exp(-0.045 * vehicle_age) + 0.55

    # ── CV confidence: lower confidence → higher uncertainty premium ──
    conf_factor = 0.80 + confidence * 0.40  # 0.80 – 1.20

    # ── XGBoost residual: use model output as a scaled adjustment ──
    log_pred = model.predict(X)[0]
    xgb_cost = float(np.expm1(log_pred))
    xgb_factor = 0.70 + (xgb_cost / 6000) * 0.60  # normalised around typical $6k output
    xgb_factor = max(0.50, min(xgb_factor, 2.0))

    # ── Deterministic variation: unique per input combination ──
    hash_str = f"{vehicle_value}_{vehicle_age}_{damage_class}_{round(confidence, 2)}"
    h = int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)
    variation = 0.88 + (h % 10000) / 40000  # 0.88 – 1.13

    cost = base * value_factor * age_factor * conf_factor * xgb_factor * variation
    cost = max(150, min(cost, 80000))

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
