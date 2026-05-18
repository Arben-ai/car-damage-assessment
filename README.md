# Car Damage Assessment & Repair Cost Estimation

AI Applications Final Project — combining Computer Vision, ML Numeric Data, and NLP.

## Use Case

A user uploads a photo of a damaged car along with basic vehicle information. The system:
1. **Detects and classifies the damage** (CV — EfficientNet on CarDD dataset)
2. **Estimates repair costs** (ML — XGBoost/Random Forest on insurance claim data)
3. **Generates a professional damage report** (NLP — RAG + LLM prompt engineering)

## Pipeline

```
Photo + Vehicle Metadata
        ↓
[CV]  Damage type & severity classification
        ↓
[ML]  Repair cost estimation (CV output as feature)
        ↓
[NLP] Professional damage report generation
        ↓
    Streamlit Web App
```

## Blocks Used

- **Computer Vision** — EfficientNet-B0 fine-tuned on CarDD dataset (6 damage categories)
- **ML Numeric Data** — Random Forest & XGBoost on car insurance claim data
- **NLP** — RAG with FAISS + Claude API for report generation

## Data Sources

| Source | Type | Block |
|---|---|---|
| [VehiDE Dataset (Kaggle)](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) | Images (13,945) | CV |
| [Car Insurance Claim Data (Kaggle)](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data) | CSV structured | ML |
| [NHTSA ODI Complaints](https://api.nhtsa.gov/complaints/complaintsByVehicle) | Text | NLP |

## Project Structure

```
car-damage-assessment/
├── notebooks/
│   ├── 01_cv_training.ipynb       # CV: EDA, training, evaluation
│   ├── 02_ml_training.ipynb       # ML: EDA, feature engineering, model comparison
│   └── 03_nlp_evaluation.ipynb    # NLP: RAG setup, prompt comparison
├── src/
│   ├── cv_model.py                # CV inference
│   ├── ml_model.py                # ML inference
│   └── nlp_report.py             # Report generation
├── app/
│   └── app.py                     # Streamlit app
├── models/                        # Saved model weights
├── data/
│   ├── raw/                       # Original datasets (not in git)
│   └── processed/                 # Processed data (not in git)
├── requirements.txt
└── documentation.md
```

## Setup

See [Execution Instructions](documentation.md#4-execution-instructions) in the documentation.

## Deployment

[Deployment URL — TBD]

## Submission

- Student: Arben Mustafi
- Deadline: 07 June 2026
- Reviewers: `jasminh`, `bkuehnis`
