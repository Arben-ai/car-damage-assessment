# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: Car Damage Assessment & Repair Cost Estimation
- Student: Arben Mustafi
- GitHub repository URL: https://github.com/Arben-ai/car-damage-assessment
- Deployment URL: https://car-damage-assessment.streamlit.app
- Submission date: 07 June 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: Computer Vision
- Primary block 2: ML Numeric Data

If a third block is selected, it is documented and graded separately as extra work.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- **Problem statement:** Assessing vehicle damage and estimating repair costs is slow, expensive, and inconsistent when done manually. Insurers and repair shops need a faster, more objective solution.
- **Goal:** Build an end-to-end AI system that takes a photo of a damaged vehicle and automatically (1) classifies the damage type, (2) estimates repair costs, and (3) generates a professional insurance damage report.
- **Success criteria:**
  - CV model achieves ≥68% test accuracy across 7 damage categories
  - ML model achieves R² > 0.5 on held-out test set
  - NLP generates complete, structured reports (completeness score ≥ 3/4) that integrate CV and ML outputs
  - Deployed Streamlit app runs the full pipeline end-to-end from a single uploaded image

### 1.2 Integration Logic
- **How the selected blocks interact:**
  1. A user uploads a vehicle photo → **CV block** classifies the damage type and outputs a confidence score per class
  2. The predicted damage class and confidence score are fed as features into the **ML block**, which combines them with vehicle metadata (age, value) to estimate repair cost
  3. Both outputs (damage type + cost estimate) are passed to the **NLP block**, which retrieves relevant NHTSA complaint records and generates a structured JSON insurance report via LLM

- **Data and output flow between blocks:**

```
[Image]
   ↓
[CV] EfficientNet-B0 → damage_class, confidence, class_probs
   ↓
[ML] XGBoost (features: vehicle_age, vehicle_value, cv_damage_class, cv_confidence, ...) → estimated_cost_usd
   ↓
[NLP] RAG (NHTSA complaints) + Claude API → structured JSON damage report
   ↓
[Streamlit App] displays all outputs to user
```

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [Car Insurance Claim Data — Kaggle](https://www.kaggle.com/datasets/xiaomengsun/car-insurance-claim-data) | CSV (structured) | ~15,000 rows, 40+ columns | Primary training data: vehicle features + claim amounts |
| 2 | CV block predictions ([`src/ml_model.py`](src/ml_model.py)) | Derived numeric features | Same size as training set | `cv_damage_class`, `cv_confidence`, `cv_damage_multiplier` as additional input features |

#### 2A.2 Preprocessing and Features

- **Cleaning steps:** See *Data Cleaning & Feature Engineering* in [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#data-cleaning--feature-engineering)
  - Stripped currency symbols (`$`, `,`) from monetary columns
  - Encoded binary Yes/No columns to 0/1
  - Median imputation for remaining missing values via `SimpleImputer`
  - Filtered to rows with actual claims (`CLM_AMT > 0`)

- **Preprocessing steps:**
  - Log-transformed target variable (`log1p(CLM_AMT)`) to normalize skewed distribution
  - One-hot encoding of remaining categorical columns
  - Train / Val / Test split: 68% / 12% / 20%

- **Feature engineering and selection:** See [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#data-cleaning--feature-engineering)
  - `VALUE_PER_AGE`: vehicle value divided by age — captures depreciation
  - `IS_REPEAT_CLAIMER`: binary flag for policyholders with prior claims
  - CV-derived features: `cv_damage_class` (one-hot), `cv_confidence`, `cv_damage_multiplier`

#### 2A.3 Model Selection

- **Models tested:** Random Forest Regressor, XGBoost Regressor
- **Why these models were chosen:** Both handle mixed numeric/categorical features well and are robust to outliers. XGBoost additionally supports early stopping and native handling of missing values. SHAP interpretability is available for both, enabling error analysis.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Raw features, no CV input, linear target | Random Forest | Val MAE | — |
| 2 | Add CV features | `cv_damage_class`, `cv_confidence`, `cv_damage_multiplier` added | Random Forest, XGBoost | Val MAE | Improved MAE |
| 3 | Log-transform + tuning | Log1p target, XGBoost early stopping, `VALUE_PER_AGE` | XGBoost | Val MAE, R² | Best performance |

See full iteration results in [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#model-comparison).

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** MAE (log-scale and original $), RMSE, R²
- **Final results:** See [`notebooks/02_ml_training.ipynb`](notebooks/02_ml_training.ipynb#final-evaluation-on-test-set)
- **Error patterns and likely causes:**
  - Underestimates on extreme claim amounts (>$10,000) — few training examples in that range
  - Residuals are approximately normally distributed (see [`data/processed/ml_residuals.png`](data/processed/ml_residuals.png))
  - CV features (damage class, confidence) ranked in top features by SHAP — confirms meaningful cross-block integration

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):** `damage_class`, `confidence`, `class_probs` from CV block ([`src/cv_model.py`](src/cv_model.py)) — mapped to `cv_damage_multiplier` and one-hot damage class features in [`src/ml_model.py`, lines 24–33](src/ml_model.py#L24-L33)
- **Outputs provided to other block(s):** `estimated_cost_usd`, `cost_range_low`, `cost_range_high` → passed to NLP block for report generation ([`src/nlp_report.py`, line 52](src/nlp_report.py#L52))

---

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [NHTSA ODI Complaints API](https://api.nhtsa.gov/complaints/complaintsByVehicle) | Text (JSON API) | ~2,000–5,000 complaint records | RAG knowledge base — retrieved as context for report generation |
| 2 | CV + ML block outputs | Structured dict | Per inference call | Input facts for prompt (damage type, cost estimate) |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** See *Build RAG Knowledge Base* in [`notebooks/03_nlp_evaluation.ipynb`](notebooks/03_nlp_evaluation.ipynb#build-rag-knowledge-base-from-nhtsa-data)
  - Fetched complaints for 5 car makes × 5 years via NHTSA API
  - Filtered texts shorter than 50 characters
  - Embedded with `all-MiniLM-L6-v2` (SentenceTransformers)
  - Stored in FAISS `IndexFlatIP` (cosine similarity via normalized inner product)

- **Prompt design or retrieval setup:** See *Prompt Strategy Comparison* in [`notebooks/03_nlp_evaluation.ipynb`](notebooks/03_nlp_evaluation.ipynb#prompt-strategy-comparison)
  - Query = `"{damage_class} {vehicle_make} {damage_location}"`
  - Top-3 retrieved passages (truncated to 400 chars each) injected into prompt
  - Structured JSON output enforced via system prompt (`"Always respond with valid JSON only"`)

#### 2B.3 Approach Selection

- **Approach used:** Retrieval-Augmented Generation (RAG) + prompt engineering via OpenAI API (`gpt-4o-mini`)
- **Alternatives considered:**
  - Zero-shot (no retrieval): simpler but produces generic output without domain grounding
  - Classical NLP (e.g. template-based): not flexible enough for varied damage scenarios
  - Fine-tuned transformer: requires labeled report data which is not publicly available

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Zero-shot baseline | No context, only vehicle facts | gpt-4o-mini, simple prompt | Completeness score (0–4) | — |
| 2 | RAG-enhanced | Top-3 NHTSA passages added to prompt | gpt-4o-mini, RAG prompt | Completeness score | +0.3 avg improvement |
| 3 | RAG + structured JSON | System prompt enforces JSON output schema | gpt-4o-mini, strict JSON system prompt | JSON parse success rate, field coverage | Machine-readable, consistent output |

See evaluation across 10 test cases in [`notebooks/03_nlp_evaluation.ipynb`](notebooks/03_nlp_evaluation.ipynb#quantitative-evaluation).

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** 10 synthesized test cases (varied damage class, vehicle, location); completeness score (0–4 required fields present); JSON parse success rate
- **Results:** See [`data/processed/nlp_evaluation_results.csv`](data/processed/nlp_evaluation_results.csv) and [`data/processed/nlp_evaluation.png`](data/processed/nlp_evaluation.png)
- **Error patterns and likely causes:**
  - Occasional JSON parse failures when model adds explanatory text before the JSON object — mitigated by strict system prompt
  - Zero-shot reports lack specificity for rare damage types (e.g. `tire_flat`) — RAG retrieval improves grounding
  - Cost figure in report may differ from ML estimate when model paraphrases — acceptable given narrative format

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):**
  - From CV: `damage_class`, `confidence` → stated in prompt as detected damage type
  - From ML: `estimated_cost_usd`, `cost_range_low`, `cost_range_high` → stated in prompt as repair cost estimate
  - See [`src/nlp_report.py`, lines 42–60](src/nlp_report.py#L42-L60)
- **Outputs provided to other block(s):** Final JSON report displayed in Streamlit app ([`app/app.py`, lines 88–100](app/app.py#L88-L100))

---

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [VehiDE Dataset (Kaggle)](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) | Images (JPG/PNG) | 13,945 images, 8 classes | Training and evaluation of damage classifier |

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:** See *Image Preprocessing & Augmentation* in [`notebooks/01_cv_training.ipynb`](notebooks/01_cv_training.ipynb#image-preprocessing--augmentation)
  - Resize to 224×224 (ImageNet standard)
  - Normalize with ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

- **Augmentation strategy (training only):**
  - Random horizontal flip (p=0.5)
  - Random rotation ±15°
  - Color jitter (brightness, contrast, saturation ±0.2)
  - Random grayscale (p=0.1)
  - `WeightedRandomSampler` to handle class imbalance

#### 2C.3 Model Selection

- **Vision model(s) used:** EfficientNet-B0 (primary), ResNet-18 (baseline)
- **Why these model(s) were chosen:** EfficientNet-B0 achieves strong accuracy with only 5.3M parameters — efficient for deployment. ResNet-18 (11.7M params) used as baseline. Both pretrained on ImageNet, fine-tuned end-to-end on VehiDE. See [`notebooks/01_cv_training.ipynb`](notebooks/01_cv_training.ipynb#model-2-efficientnet-b0-primary-model).

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline classifier | No augmentation, standard sampler | ResNet-18 | Val Accuracy | — |
| 2 | Augmentation + balanced sampler | Added color jitter, rotation, WeightedRandomSampler | ResNet-18 | Val Accuracy | Improved on minority classes |
| 3 | EfficientNet-B0 + full augmentation | Replaced backbone | EfficientNet-B0 | Val Accuracy, Test F1 | Best overall performance |

See training curves in [`data/processed/cv_model_comparison.png`](data/processed/cv_model_comparison.png).

#### 2C.5 Evaluation and Error Analysis

- **Metrics and/or visual checks:** Accuracy, per-class Precision/Recall/F1, confusion matrix
- **Final results:** See [`notebooks/01_cv_training.ipynb`](notebooks/01_cv_training.ipynb#evaluation-on-test-set) and [`data/processed/cv_confusion_matrix.png`](data/processed/cv_confusion_matrix.png)
- **Error patterns and limitations:**
  - Confusion between `dent` and `scratch` (visually similar in compressed images)
  - `tire_flat` occasionally misclassified as `crack` when sidewall damage is prominent
  - Model performs best on `glass_breakage` (visually distinct)
  - Limitation: model trained on static images; video-based analysis not supported

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s):** Raw vehicle image from user upload (via Streamlit app)
- **Outputs provided to other block(s):** `damage_class`, `confidence`, `class_probs` → consumed by ML block as features ([`src/ml_model.py`, lines 24–33](src/ml_model.py#L24-L33)) and by NLP block as prompt facts ([`src/nlp_report.py`, line 43](src/nlp_report.py#L43))

---

## 3. Deployment

- **Deployment URL:** https://car-damage-assessment.streamlit.app
- **Main user flow:**
  1. User fills in vehicle metadata in sidebar (make, model, year, value, damage location)
  2. User uploads a photo of the damaged vehicle
  3. User clicks **Analyze Damage**
  4. App runs CV → ML → NLP pipeline and displays: damage type + confidence chart, estimated cost range, full structured insurance report

- **Screenshot or short demo:** See [`app/app.py`](app/app.py). Live demo available at https://car-damage-assessment.streamlit.app

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- **Environment setup:**
```bash
git clone https://github.com/Arben-ai/car-damage-assessment.git
cd car-damage-assessment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

- **Data setup:**
```bash
# VehiDE dataset (Kaggle, no access request needed):
kaggle datasets download -d hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection -p data/raw/
# Then: unzip data/raw/vehide-dataset-*.zip -d data/raw/vehide/

# Kaggle insurance data:
pip install kaggle
kaggle datasets download -d xiaomengsun/car-insurance-claim-data -p data/raw/

# NHTSA data is fetched automatically when running notebook 03
```

- **Training command(s):**
```bash
# Run notebooks in order:
jupyter notebook notebooks/01_cv_training.ipynb    # CV — trains EfficientNet-B0
jupyter notebook notebooks/02_ml_training.ipynb    # ML — trains XGBoost
jupyter notebook notebooks/03_nlp_rag.ipynb        # NLP — builds FAISS index from NHTSA data
jupyter notebook notebooks/03_nlp_evaluation.ipynb # NLP — prompt strategy comparison + evaluation
```

- **Inference/run command(s):**
```bash
streamlit run app/app.py
```

- **Reproducibility notes:**
  - All random seeds set to 42 (PyTorch, NumPy, sklearn, XGBoost)
  - Package versions pinned in `requirements.txt`
  - NHTSA API data cached to `data/raw/nhtsa_complaints.csv` after first fetch
  - Trained model weights saved to `models/` (not tracked in git due to size — run notebooks to regenerate)

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [x] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

**Evidence for selected bonus items:**

**Third block (NLP):** Full RAG pipeline with FAISS vector index over real NHTSA complaint data, 3-strategy prompt comparison with quantitative evaluation across 10 test cases, structured JSON output enforced via system prompt. See [`notebooks/03_nlp_evaluation.ipynb`](notebooks/03_nlp_evaluation.ipynb).

**Multiple data sources:** Three distinct data sources of different types and origins:
1. VehiDE (images, Kaggle, vision-specific, 13,945 images, 8 damage classes)
2. Car Insurance Claim Data / Kaggle (structured CSV, insurance domain)
3. NHTSA ODI Complaints (text, US government API, real-world vehicle defect reports)

Each source serves a distinct role and is used exclusively by one block, avoiding data leakage across blocks.
