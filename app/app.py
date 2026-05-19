import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from src.cv_model import load_model, predict as cv_predict
from src.ml_model import load_artifacts, predict_cost
from src.nlp_report import load_rag_index, generate_report

MODELS_DIR    = Path(__file__).parent.parent / 'models'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

DAMAGE_LABELS = {
    'broken_glass':  'Glasbruch',
    'broken_lights': 'Lampenbruch',
    'dents':         'Delle',
    'lost_parts':    'Fehlende Teile',
    'punctured':     'Loch / Perforation',
    'scratch':       'Kratzer',
    'torn':          'Riss / Einriss'
}
DAMAGE_ICONS = {
    'broken_glass': '🪟', 'broken_lights': '💡', 'dents': '🔨',
    'lost_parts': '🔩', 'punctured': '🕳️', 'scratch': '✏️', 'torn': '💥'
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px; padding: 3rem 2.5rem; margin-bottom: 2rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
}
.hero h1 { color: #fff; font-size: 2.6rem; font-weight: 700; margin: 0 0 0.6rem 0; }
.hero p  { color: #adb5bd; font-size: 1.1rem; margin: 0; }
.hero-badge {
    display: inline-block; background: rgba(255,75,75,0.15); color: #ff4b4b;
    border: 1px solid rgba(255,75,75,0.3); border-radius: 20px;
    padding: 0.3rem 1rem; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem;
}
.step-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.2rem 1rem; text-align: center; height: 100%;
}
.step-num {
    background: #ff4b4b; color: white; font-weight: 700; font-size: 0.85rem;
    width: 28px; height: 28px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;
}
.step-title { color: #fff; font-weight: 600; font-size: 0.9rem; }
.step-sub   { color: #868e96; font-size: 0.78rem; margin-top: 0.2rem; }
.section-title {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #ff4b4b; margin: 1.2rem 0 0.6rem 0;
}
.result-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
}
.damage-badge {
    display: inline-block; background: rgba(255,75,75,0.15); color: #ff6b6b;
    border: 1px solid rgba(255,75,75,0.25); border-radius: 8px;
    padding: 0.4rem 1rem; font-size: 1.1rem; font-weight: 600;
}
.conf-bar-bg  { background: rgba(255,255,255,0.08); border-radius: 6px; height: 8px; width: 100%; margin-top: 0.4rem; }
.conf-bar-fill { background: linear-gradient(90deg, #ff4b4b, #ff9f43); border-radius: 6px; height: 8px; }
.cost-big   { font-size: 2.4rem; font-weight: 700; color: #51cf66; line-height: 1; }
.cost-range { color: #868e96; font-size: 0.9rem; margin-top: 0.3rem; }
.report-field { margin-bottom: 0.8rem; }
.report-label { color: #868e96; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.report-value { color: #fff; font-size: 0.95rem; margin-top: 0.1rem; }
.severity-moderate { color: #ffd43b; } .severity-severe { color: #ff4b4b; } .severity-minor { color: #51cf66; }
.success-bar {
    background: linear-gradient(90deg, rgba(81,207,102,0.15), rgba(81,207,102,0.05));
    border: 1px solid rgba(81,207,102,0.3); border-radius: 10px;
    padding: 0.8rem 1.2rem; color: #51cf66; font-weight: 600;
    text-align: center; margin-top: 1rem;
}
</style>
"""

st.set_page_config(page_title='Car Damage Assessment', page_icon='🚗', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    cv_model = load_model(str(MODELS_DIR / 'efficientnet_b0_best.pth'), 'cpu')
    ml_model, imputer, feature_names = load_artifacts(str(MODELS_DIR))
    index, texts, embedder = load_rag_index(str(MODELS_DIR), str(PROCESSED_DIR))
    return cv_model, ml_model, imputer, feature_names, index, texts, embedder

with st.spinner('Loading AI models...'):
    cv_model, ml_model, imputer, feature_names, index, texts, embedder = load_all_models()

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · CV + ML + NLP</div>
    <h1>🚗 Car Damage Assessment</h1>
    <p>Upload a photo of a damaged vehicle for instant damage detection,<br>
    repair cost estimation and a structured insurance report.</p>
</div>""", unsafe_allow_html=True)

# Pipeline steps
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="step-card"><div class="step-num">1</div>
        <div class="step-title">📸 Damage Detection</div>
        <div class="step-sub">EfficientNet-B0 classifies the damage type from the photo</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="step-card"><div class="step-num">2</div>
        <div class="step-title">💰 Cost Estimation</div>
        <div class="step-sub">XGBoost predicts repair cost using vehicle & damage data</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="step-card"><div class="step-num">3</div>
        <div class="step-title">📄 Insurance Report</div>
        <div class="step-sub">GPT-4o-mini generates a structured claim report via RAG</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🚘 Vehicle Information")
    st.markdown("---")
    vehicle_make  = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Volkswagen', 'Other'])
    vehicle_model = st.text_input('Model', value='Corolla')
    vehicle_year  = st.number_input('Year', min_value=2000, max_value=2026, value=2020)
    vehicle_value = st.number_input('Estimated value (USD)', min_value=1000, max_value=200000, value=18000, step=500)
    vehicle_age   = 2026 - vehicle_year
    damage_loc    = st.selectbox('Damage location', [
        'front bumper', 'rear bumper', 'hood', 'front left door', 'front right door',
        'rear left door', 'rear right door', 'windshield', 'side panel', 'roof', 'other'
    ])
    st.markdown("---")
    api_key = st.text_input('🔑 OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
    if api_key:
        st.success('API key ready ✓')

# Upload + results
col_img, col_res = st.columns([1, 1], gap="large")
with col_img:
    uploaded = st.file_uploader('Upload vehicle damage photo', type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

if uploaded and st.button('🔍 Analyze Damage', type='primary', use_container_width=True):
    if not api_key:
        st.error('Please enter your OpenAI API Key in the sidebar.')
        st.stop()

    with col_res:
        with st.status('Running AI pipeline...', expanded=True) as status:
            st.write('📸 Step 1: Detecting damage type...')
            cv_result = cv_predict(image, cv_model)
            st.write('💰 Step 2: Estimating repair cost...')
            vehicle_info = {'make': vehicle_make, 'model': vehicle_model,
                            'year': vehicle_year, 'value_usd': vehicle_value, 'location': damage_loc}
            ml_result = predict_cost(cv_result, vehicle_age, vehicle_value, ml_model, imputer, feature_names)
            st.write('📄 Step 3: Generating insurance report...')
            report = generate_report(cv_result, ml_result, vehicle_info, index, texts, embedder, api_key)
            status.update(label='Analysis complete!', state='complete')

        # CV
        st.markdown('<div class="section-title">Computer Vision — Damage Detection</div>', unsafe_allow_html=True)
        dk   = cv_result['damage_class']
        conf = cv_result['confidence']
        st.markdown(f"""<div class="result-card">
            <div style="margin-bottom:0.8rem"><span class="damage-badge">{DAMAGE_ICONS.get(dk,'🔍')} {DAMAGE_LABELS.get(dk,dk)}</span></div>
            <div style="color:#adb5bd;font-size:0.85rem;margin-bottom:0.3rem">Confidence: <strong style="color:white">{conf:.1%}</strong></div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div></div>
        </div>""", unsafe_allow_html=True)
        st.bar_chart({DAMAGE_LABELS.get(k,k): v for k,v in cv_result['class_probs'].items()}, height=160)

        # ML
        st.markdown('<div class="section-title">ML — Repair Cost Estimation</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="result-card">
            <div class="cost-big">${ml_result['estimated_cost_usd']:,.0f}</div>
            <div class="cost-range">Range: ${ml_result['cost_range_low']:,.0f} – ${ml_result['cost_range_high']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

        # NLP
        st.markdown('<div class="section-title">NLP — Insurance Report</div>', unsafe_allow_html=True)
        if 'error' not in report:
            dmg   = report.get('damage', {})
            asmnt = report.get('assessment', {})
            sev   = dmg.get('severity', '').lower()
            sev_cls = f"severity-{sev}" if sev in ('moderate','severe','minor') else ''
            st.markdown(f"""<div class="result-card">
                <div class="report-field">
                    <div class="report-label">Damage Type & Severity</div>
                    <div class="report-value">{dmg.get('type','—')} — <span class="{sev_cls}">{dmg.get('severity','—')}</span></div>
                </div>
                <div class="report-field">
                    <div class="report-label">Description</div>
                    <div class="report-value">{dmg.get('description','—')}</div>
                </div>
                <div class="report-field">
                    <div class="report-label">Recommendation</div>
                    <div class="report-value">{asmnt.get('repair_recommendation','—')}</div>
                </div>
                <div class="report-field">
                    <div class="report-label">Confidence Level</div>
                    <div class="report-value">{asmnt.get('confidence_level','—')}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            with st.expander('📋 Full JSON Report'):
                st.json(report)
        else:
            st.warning('Report generation failed.')
            st.text(report.get('raw_output', ''))

        st.markdown('<div class="success-bar">✅ Analysis complete — all 3 AI blocks executed successfully</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Applications Final Project — CV + ML + NLP | Arben Mustafi | 2026")
