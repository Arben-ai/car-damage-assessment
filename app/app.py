import sys, os, json, io, textwrap
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import numpy as np
from fpdf import FPDF

from src.cv_model import load_model, predict as cv_predict, compute_gradcam
from src.ml_model import load_artifacts, predict_cost, compute_shap
from src.nlp_report import load_rag_index, generate_report, retrieve

MODELS_DIR    = Path(__file__).parent.parent / 'models'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

DAMAGE_LABELS = {
    'broken_glass':  'Broken Glass',  'broken_lights': 'Broken Lights',
    'dents':         'Dents',         'lost_parts':    'Missing Parts',
    'punctured':     'Puncture',      'scratch':       'Scratch',
    'torn':          'Tear / Rip'
}
DAMAGE_ICONS = {
    'broken_glass': '🪟', 'broken_lights': '💡', 'dents': '🔨',
    'lost_parts': '🔩', 'punctured': '🕳️', 'scratch': '✏️', 'torn': '💥'
}
DAMAGE_COLORS = {
    'broken_glass': '#74c0fc', 'broken_lights': '#ffd43b', 'dents': '#ff9f43',
    'lost_parts': '#ff6b6b', 'punctured': '#da77f2', 'scratch': '#63e6be', 'torn': '#ff4b4b'
}
SEVERITY_COLOR = {'minor': '#51cf66', 'moderate': '#ffd43b', 'severe': '#ff4b4b'}
SHAP_LABELS = {
    'VEHICLE_AGE':              'Vehicle Age',
    'BLUEBOOK':                 'Vehicle Value',
    'cv_confidence':            'CV Confidence',
    'cv_damage_multiplier':     'Damage Type Weight',
    'VALUE_PER_AGE':            'Value per Year',
    'cv_damage_class_broken_glass':  'Damage: Broken Glass',
    'cv_damage_class_broken_lights': 'Damage: Broken Lights',
    'cv_damage_class_dents':         'Damage: Dents',
    'cv_damage_class_lost_parts':    'Damage: Missing Parts',
    'cv_damage_class_punctured':     'Damage: Puncture',
    'cv_damage_class_scratch':       'Damage: Scratch',
    'cv_damage_class_torn':          'Damage: Tear / Rip',
}
SEVERITY_MULT  = {'Minor': 0.6, 'Moderate': 1.0, 'Severe': 1.5, 'Total Loss': 2.4}

# Market average prices per damage type and vehicle tier
MARKET_PRICES = {
    'scratch':       {'Budget': 320,  'Mid-range': 580,  'Premium': 1100},
    'dents':         {'Budget': 420,  'Mid-range': 750,  'Premium': 1400},
    'broken_glass':  {'Budget': 280,  'Mid-range': 540,  'Premium': 1300},
    'broken_lights': {'Budget': 230,  'Mid-range': 470,  'Premium': 1050},
    'lost_parts':    {'Budget': 580,  'Mid-range': 1050, 'Premium': 2400},
    'punctured':     {'Budget': 190,  'Mid-range': 380,  'Premium': 860},
    'torn':          {'Budget': 480,  'Mid-range': 860,  'Premium': 1700},
}
PREMIUM_MAKES = {'BMW', 'Mercedes', 'Audi'}
BUDGET_MAKES  = {'Hyundai'}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08) !important; }

/* ── Sidebar vehicle card ── */
.vehicle-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(15,52,96,0.18);
}
.vehicle-card-title {
    color: #adb5bd; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem;
}
.vehicle-card-name {
    color: #ffffff; font-size: 1.15rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.6rem;
}
.vehicle-card-row {
    display: flex; justify-content: space-between;
    border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.6rem; margin-top: 0.2rem;
}
.vehicle-card-label { color: #6c757d; font-size: 0.75rem; }
.vehicle-card-value { color: #e9ecef; font-size: 0.78rem; font-weight: 600; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px; padding: 3.5rem 3rem; margin-bottom: 2rem;
    text-align: center; border: none;
    box-shadow: 0 8px 32px rgba(15,52,96,0.18);
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,75,75,0.15); color: #ff6b6b;
    border: 1px solid rgba(255,75,75,0.35); border-radius: 20px;
    padding: 0.35rem 1.1rem; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.hero h1 { color: #fff; font-size: 3rem; font-weight: 800; margin: 0 0 0.8rem 0; line-height: 1.1; }
.hero p  { color: #adb5bd; font-size: 1.05rem; margin: 0 auto; max-width: 600px; line-height: 1.6; }

/* ── Step cards with hover ── */
.step-card {
    background: #1e2130; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center; height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.step-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.step-num {
    background: linear-gradient(135deg, #e03131, #ff6b6b); color: white;
    font-weight: 800; font-size: 0.8rem; width: 30px; height: 30px;
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; margin-bottom: 0.6rem;
}
.step-title { color: #f1f3f5; font-weight: 700; font-size: 0.92rem; margin-bottom: 0.3rem; }
.step-sub   { color: #868e96; font-size: 0.78rem; line-height: 1.5; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #ff6b6b; margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem; border-bottom: 2px solid rgba(224,49,49,0.3);
}

/* ── Tabs as pills ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: #1e2130;
    border-radius: 12px;
    padding: 0.35rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #868e96 !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background: #e03131 !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── Cards with hover ── */
.card {
    background: #1e2130; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.4rem; margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
}

/* ── Damage result hero ── */
.damage-result {
    background: #1e2130;
    border-radius: 16px; padding: 1.6rem; margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    text-align: center;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.damage-result:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.10); }
.damage-result-icon  { font-size: 3rem; line-height: 1; margin-bottom: 0.5rem; }
.damage-result-label { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.3rem; }
.damage-result-conf  { color: #868e96; font-size: 0.88rem; }

.damage-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    border-radius: 10px; padding: 0.5rem 1.2rem; font-size: 1.1rem; font-weight: 700;
    margin-bottom: 0.8rem;
}
.field-label { color: #868e96; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.2rem; }
.field-value { color: #e9ecef; font-size: 0.95rem; line-height: 1.5; }
.cost-big    { font-size: 3rem; font-weight: 800; color: #51cf66; line-height: 1; }
.cost-sub    { color: #868e96; font-size: 0.88rem; margin-top: 0.4rem; }

/* ── Similar case cards with hover ── */
.case-card {
    background: #1a2744; border: 1px solid #1971c2;
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    font-size: 0.83rem; color: #adb5bd; line-height: 1.6;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.case-card:hover { transform: translateY(-2px); box-shadow: 0 4px 14px rgba(25,113,194,0.12); }
.case-card strong { color: #1971c2; }

/* ── Pipeline steps ── */
.pipeline-step {
    display: flex; align-items: center; gap: 0.9rem;
    background: #1e2130; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.pipeline-step.done   { border-left: 3px solid #2f9e44; }
.pipeline-step.active { border-left: 3px solid #e03131; background: #2a1a1e; }
.pipeline-icon  { font-size: 1.2rem; }
.pipeline-label { font-size: 0.88rem; font-weight: 600; color: #e9ecef; }
.pipeline-sub   { font-size: 0.75rem; color: #868e96; }

/* ── Success banner ── */
.success-banner {
    background: rgba(47,158,68,0.1);
    border: 1px solid rgba(47,158,68,0.3); border-radius: 12px;
    padding: 1rem 1.5rem; text-align: center; margin-top: 1rem;
    color: #69db7c; font-weight: 600; font-size: 0.95rem;
}
.market-better { color: #69db7c; font-weight: 700; }
.market-worse  { color: #ff6b6b; font-weight: 700; }
</style>
"""

# ── PDF generator ─────────────────────────────────────────────────────────────
def _s(text):
    """Strip characters outside Latin-1 so Helvetica doesn't crash."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def build_pdf(vehicle_info, cv_result, ml_result, report, severity, adjusted_cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, 0, 210, 42, 'F')
    # Red accent stripe
    pdf.set_fill_color(224, 49, 49)
    pdf.rect(0, 42, 210, 3, 'F')
    # Title
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_xy(14, 9)
    pdf.cell(130, 11, 'DAMAGE ASSESSMENT', ln=False)
    # Badge top-right
    pdf.set_fill_color(224, 49, 49)
    pdf.set_xy(158, 10)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.cell(38, 8, '  AI GENERATED', fill=True, border=0)
    # Subtitle
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(173, 181, 189)
    pdf.set_xy(14, 22)
    pdf.cell(180, 6, f"Insurance Report  ·  {datetime.now().strftime('%B %d, %Y  ·  %H:%M')}", ln=True)
    pdf.set_xy(14, 29)
    pdf.set_font('Helvetica', '', 8)
    vi = vehicle_info
    pdf.cell(180, 6, _s(f"{vi.get('year','')} {vi.get('make','')} {vi.get('model','')}  ·  {vi.get('location','—')}  ·  ${vi.get('value_usd',0):,} USD"), ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(52)

    def section(title):
        pdf.ln(2)
        pdf.set_fill_color(26, 26, 46)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(255, 107, 107)
        pdf.cell(0, 7, f'  {title}', fill=True, ln=True, new_x='LMARGIN', new_y='NEXT')
        # Accent line under section title
        pdf.set_fill_color(224, 49, 49)
        pdf.rect(pdf.get_x(), pdf.get_y(), 210, 0.8, 'F')
        pdf.ln(3)
        pdf.set_text_color(30, 30, 30)

    def row(label, value, bold_val=False, highlight=False):
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(55, 7, _s(label))
        if highlight:
            pdf.set_text_color(47, 158, 68)
            pdf.set_font('Helvetica', 'B', 11)
        else:
            pdf.set_text_color(30, 30, 30)
            pdf.set_font('Helvetica', 'B' if bold_val else '', 9)
        pdf.multi_cell(0, 7, _s(value), new_x='LMARGIN', new_y='NEXT')

    # ── Two-column summary box ───────────────────────────────────────────────
    pdf.set_fill_color(240, 240, 248)
    pdf.rect(14, pdf.get_y(), 182, 22, 'F')
    y_box = pdf.get_y() + 4
    dk = cv_result['damage_class']
    damage_label = DAMAGE_LABELS.get(dk, dk)

    pdf.set_xy(18, y_box)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(224, 49, 49)
    pdf.cell(85, 7, _s(damage_label), ln=False)
    pdf.set_xy(105, y_box)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(47, 158, 68)
    pdf.cell(85, 7, f'${adjusted_cost:,.0f} USD', ln=False)

    pdf.set_xy(18, y_box + 8)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(85, 5, f"Confidence: {cv_result['confidence']:.1%}  ·  Severity: {severity}", ln=False)
    pdf.set_xy(105, y_box + 8)
    pdf.cell(85, 5, f"Range: ${adjusted_cost*0.8:,.0f} – ${adjusted_cost*1.2:,.0f}", ln=False)
    pdf.ln(28)

    # ── Damage Assessment ────────────────────────────────────────────────────
    section('DAMAGE ASSESSMENT  (Computer Vision · EfficientNet-B0)')
    row('Damage Type', damage_label, bold_val=True)
    row('Severity', severity, bold_val=True)
    row('AI Confidence', f"{cv_result['confidence']:.1%}")
    dmg = report.get('damage', {})
    if dmg.get('description'):
        row('Description', dmg['description'])

    # ── Cost Estimate ────────────────────────────────────────────────────────
    section('REPAIR COST ESTIMATE  (XGBoost ML Model)')
    row('Estimated Cost', f"${adjusted_cost:,.0f} USD", highlight=True)
    row('Cost Range', f"${adjusted_cost*0.8:,.0f} – ${adjusted_cost*1.2:,.0f} USD")
    asmnt = report.get('assessment', {})
    if asmnt.get('repair_recommendation'):
        row('Recommendation', asmnt['repair_recommendation'])
    if asmnt.get('confidence_level'):
        row('Confidence Level', asmnt['confidence_level'])

    # ── Class Probabilities ──────────────────────────────────────────────────
    section('CLASS PROBABILITIES  (All Damage Types)')
    probs = sorted(cv_result['class_probs'].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in probs:
        label = DAMAGE_LABELS.get(cls, cls)
        is_top = cls == dk
        filled = max(1, int(prob * 78))
        empty  = max(1, 79 - filled)
        pdf.set_font('Helvetica', 'B' if is_top else '', 8)
        pdf.set_text_color(30, 30, 30) if is_top else pdf.set_text_color(80, 80, 80)
        pdf.cell(52, 6, _s(label))
        pdf.set_fill_color(224, 49, 49) if is_top else pdf.set_fill_color(100, 100, 140)
        pdf.cell(filled, 4, '', fill=True)
        pdf.set_fill_color(230, 230, 240)
        pdf.cell(empty, 4, '', fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(20, 6, f'{prob:.1%}', ln=True)
    pdf.ln(2)

    # ── Notes ────────────────────────────────────────────────────────────────
    notes = report.get('notes', '')
    if notes:
        section('NOTES')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, _s(notes), new_x='LMARGIN', new_y='NEXT')

    # ── Footer ───────────────────────────────────────────────────────────────
    pdf.set_auto_page_break(auto=False)
    pdf.set_y(-20)
    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, pdf.get_y(), 210, 20, 'F')
    pdf.set_fill_color(224, 49, 49)
    pdf.rect(0, pdf.get_y(), 210, 1.5, 'F')
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(173, 181, 189)
    pdf.cell(0, 5, 'AI-generated report for informational purposes only  ·  Car Damage Assessment  ·  Arben Mustafi  ·  2026  ·  car-damage-assessment.streamlit.app', align='C', new_x='LMARGIN', new_y='NEXT')

    return bytes(pdf.output())


# ── App setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Car Damage Assessment', page_icon='🚗', layout='wide')

# Load API key from Streamlit Secrets (deployed) or .env (local)
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv('OPENAI_API_KEY', ''))
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all_models():
    cv_model = load_model(str(MODELS_DIR / 'efficientnet_b0_best.pth'), 'cpu')
    ml_model, imputer, feature_names = load_artifacts(str(MODELS_DIR))
    index, texts, embedder = load_rag_index(str(MODELS_DIR), str(PROCESSED_DIR))
    return cv_model, ml_model, imputer, feature_names, index, texts, embedder

with st.spinner(''):
    cv_model, ml_model, imputer, feature_names, index, texts, embedder = load_all_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚘 Vehicle Information")
    vehicle_make  = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Volkswagen', 'Mercedes', 'Audi', 'Hyundai', 'Other'])
    vehicle_model = st.text_input('Model', value='Corolla')
    vehicle_year  = st.number_input('Year', min_value=2000, max_value=2026, value=2020)
    vehicle_value = st.number_input('Value (USD)', min_value=1000, max_value=200000, value=18000, step=500)
    vehicle_age   = 2026 - vehicle_year
    damage_loc    = st.selectbox('Damage Location', [
        'front bumper', 'rear bumper', 'hood', 'front left door', 'front right door',
        'rear left door', 'rear right door', 'windshield', 'side panel', 'roof', 'other'
    ])

    st.markdown(f"""
    <div class="vehicle-card">
        <div class="vehicle-card-title">Selected Vehicle</div>
        <div class="vehicle-card-name">{vehicle_year} {vehicle_make} {vehicle_model}</div>
        <div class="vehicle-card-row">
            <span class="vehicle-card-label">Value</span>
            <span class="vehicle-card-value">${vehicle_value:,}</span>
        </div>
        <div class="vehicle-card-row">
            <span class="vehicle-card-label">Age</span>
            <span class="vehicle-card-value">{vehicle_age} years</span>
        </div>
        <div class="vehicle-card-row">
            <span class="vehicle-card-label">Location</span>
            <span class="vehicle-card-value">{damage_loc}</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AI-Powered · CV + ML + NLP</div>
    <h1>Car Damage Assessment</h1>
    <p>Upload a photo of a damaged vehicle for instant damage classification,
    repair cost estimation and a professional insurance report.</p>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, num, icon, title, sub in [
    (c1, '1', '📸', 'Damage Detection',  'EfficientNet-B0 classifies the damage type with confidence'),
    (c2, '2', '💰', 'Cost Estimation',   'XGBoost predicts repair cost + market comparison'),
    (c3, '3', '📄', 'Insurance Report',  'RAG + GPT-4o-mini generates a downloadable PDF report'),
]:
    with col:
        st.markdown(f"""<div class="step-card">
            <div class="step-num">{num}</div>
            <div class="step-title">{icon} {title}</div>
            <div class="step-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")
with col_left:
    st.markdown('<div class="section-header">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader('', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True, caption=f'📎 {uploaded.name}')

analyze = st.button('🔍  Analyze Damage', type='primary', use_container_width=True, disabled=not bool(uploaded))

if uploaded and analyze:
    if not api_key:
        st.error('Please enter your OpenAI API Key in the sidebar.')
        st.stop()

    STEPS = [
        ('📸', 'Damage Classification',  'EfficientNet-B0 analyzing image...'),
        ('💰', 'Cost Estimation',        'XGBoost predicting repair cost...'),
        ('🔍', 'Similar Case Retrieval', 'FAISS searching NHTSA database...'),
        ('🔥', 'Damage Region Mapping',  'GradCAM generating heatmap...'),
        ('📄', 'Report Generation',      'GPT-4o-mini writing insurance report...'),
    ]
    progress_placeholder = st.empty()

    def render_steps(done_up_to):
        html = '<div style="margin:0.5rem 0">'
        for i, (icon, label, sub) in enumerate(STEPS):
            if i < done_up_to:
                cls = 'pipeline-step done'
                status_icon = '✅'
            elif i == done_up_to:
                cls = 'pipeline-step active'
                status_icon = '⏳'
            else:
                cls = 'pipeline-step'
                status_icon = '○'
            html += f'''<div class="{cls}">
                <span class="pipeline-icon">{icon}</span>
                <div><div class="pipeline-label">{status_icon} Step {i+1} — {label}</div>
                <div class="pipeline-sub">{sub}</div></div></div>'''
        html += '</div>'
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    render_steps(0)
    cv_result = cv_predict(image, cv_model)

    render_steps(1)
    vehicle_info = {'make': vehicle_make, 'model': vehicle_model,
                    'year': vehicle_year, 'value_usd': vehicle_value, 'location': damage_loc}
    ml_result = predict_cost(cv_result, vehicle_age, vehicle_value, ml_model, imputer, feature_names)

    render_steps(2)
    query = f"{cv_result['damage_class']} {vehicle_make} {damage_loc}"
    similar_cases = retrieve(query, index, texts, embedder, k=3)

    render_steps(3)
    from src.cv_model import DAMAGE_CLASSES as _DC
    pred_idx = _DC.index(cv_result['damage_class'])
    try:
        gradcam_image = compute_gradcam(image, cv_model, pred_idx)
    except Exception:
        gradcam_image = None

    render_steps(4)
    report = generate_report(cv_result, ml_result, vehicle_info, index, texts, embedder, api_key)

    progress_placeholder.markdown("""<div style="background:#ebfbee;border:1px solid #8ce99a;border-radius:10px;
        padding:0.8rem 1.2rem;color:#2f9e44;font-weight:600;font-size:0.9rem;text-align:center">
        ✅ All 5 steps complete — results ready below</div>""", unsafe_allow_html=True)

    shap_values = compute_shap(ml_result, ml_model)
    st.session_state['analysis'] = {
        'cv_result': cv_result, 'ml_result': ml_result, 'report': report,
        'similar_cases': similar_cases, 'vehicle_info': vehicle_info,
        'query': query, 'shap_values': shap_values, 'gradcam_image': gradcam_image,
    }

    # Append to session history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    thumb_buf = io.BytesIO()
    image.copy().thumbnail((120, 90))
    image.copy().resize((120, 90)).save(thumb_buf, format='JPEG')
    st.session_state['history'].append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'thumb': thumb_buf.getvalue(),
        'cv_result': cv_result,
        'ml_result': ml_result,
        'vehicle_info': {'make': vehicle_make, 'model': vehicle_model,
                         'year': vehicle_year, 'value_usd': vehicle_value},
        'filename': uploaded.name,
    })

if 'analysis' in st.session_state:
    r             = st.session_state['analysis']
    cv_result     = r['cv_result']
    ml_result     = r['ml_result']
    report        = r['report']
    similar_cases = r['similar_cases']
    vehicle_info  = r['vehicle_info']
    query         = r['query']
    shap_values   = r['shap_values']
    gradcam_image = r.get('gradcam_image')

    dk    = cv_result['damage_class']
    conf  = cv_result['confidence']
    cost  = ml_result['estimated_cost_usd']
    color = DAMAGE_COLORS.get(dk, '#ff4b4b')

    with col_right:
        tab1, tab2, tab3, tab4 = st.tabs(['🔍 Damage', '💰 Cost & Market', '📄 Report & PDF', '🗂️ Similar Cases'])

        # ── TAB 1: Damage ──────────────────────────────────────────────────
        with tab1:
            st.markdown('<div class="section-header">Computer Vision Result</div>', unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number', value=conf * 100,
                number={'suffix': '%', 'font': {'size': 32, 'color': '#212529'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#adb5bd', 'tickfont': {'color': '#6c757d'}},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': '#f1f3f5', 'borderwidth': 0,
                    'steps': [{'range': [0, 100], 'color': '#e9ecef'}],
                    'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': conf*100}
                },
                title={'text': 'Model Confidence', 'font': {'color': '#6c757d', 'size': 13}}
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    height=220, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""<div class="damage-result" style="border-top: 4px solid {color}">
                <div class="damage-result-icon">{DAMAGE_ICONS.get(dk,'🔍')}</div>
                <div class="damage-result-label" style="color:{color}">{DAMAGE_LABELS.get(dk, dk)}</div>
                <div class="damage-result-conf">
                    Detected with <strong style="color:#212529;font-size:1rem">{conf:.1%}</strong> confidence
                    &nbsp;·&nbsp; {vehicle_year} {vehicle_make} {vehicle_model}
                </div>
            </div>""", unsafe_allow_html=True)

            probs = cv_result['class_probs']
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            fig_bar = go.Figure(go.Bar(
                x=list(sorted_probs.values()),
                y=[f"{DAMAGE_ICONS.get(k,'•')} {DAMAGE_LABELS.get(k,k)}" for k in sorted_probs],
                orientation='h',
                marker=dict(color=[DAMAGE_COLORS.get(k,'#adb5bd') for k in sorted_probs], opacity=0.9),
                text=[f'{v:.0%}' for v in sorted_probs.values()], textposition='outside',
                textfont=dict(color='#495057', size=11)
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=260, margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
                yaxis=dict(tickfont=dict(color='#495057', size=11)), bargap=0.35
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # GradCAM damage region
            if gradcam_image is not None:
                st.markdown('<div class="section-header">🔥 AI Damage Region</div>',
                            unsafe_allow_html=True)
                st.image(gradcam_image, use_container_width=True,
                         caption='GradCAM heatmap — red box marks the region that drove the classification')
                st.markdown(
                    '<p style="font-size:0.78rem;color:#6c757d">'
                    'Heatmap shows which pixels most influenced the damage prediction. '
                    'Warm colours (red/yellow) = high activation · Cool colours (blue) = low activation.</p>',
                    unsafe_allow_html=True)

        # ── TAB 2: Cost + Market + Severity ───────────────────────────────
        with tab2:
            st.markdown('<div class="section-header">Severity Adjustment</div>', unsafe_allow_html=True)
            severity = st.select_slider(
                'Adjust damage severity (overrides AI estimate)',
                options=['Minor', 'Moderate', 'Severe', 'Total Loss'],
                value='Moderate',
                help='Drag to manually adjust severity. Cost estimate updates automatically.'
            )
            adj_cost = cost * SEVERITY_MULT[severity]

            st.markdown('<div class="section-header">Repair Cost Estimate</div>', unsafe_allow_html=True)
            sev_color = {'Minor': '#2f9e44', 'Moderate': '#e67700', 'Severe': '#e03131', 'Total Loss': '#c92a2a'}[severity]
            st.markdown(f"""<div class="card" style="text-align:center">
                <div class="cost-big">${adj_cost:,.0f}</div>
                <div class="cost-sub">
                    Severity: <strong style="color:{sev_color}">{severity}</strong> &nbsp;·&nbsp;
                    Range: <strong style="color:#495057">${adj_cost*0.8:,.0f} – ${adj_cost*1.2:,.0f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

            # Cost breakdown donut
            labor_pct = 0.45; parts_pct = 0.47; other_pct = 0.08
            fig_donut = go.Figure(go.Pie(
                labels=['Labor', 'Parts & Materials', 'Other'],
                values=[labor_pct * adj_cost, parts_pct * adj_cost, other_pct * adj_cost],
                hole=0.65,
                marker=dict(colors=['#e03131', '#2f9e44', '#f08c00'],
                            line=dict(color='#ffffff', width=2)),
                textinfo='label+percent', textfont=dict(color='#212529', size=11),
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color='#495057'), bgcolor='rgba(0,0,0,0)'),
                annotations=[dict(text=f'<b>${adj_cost:,.0f}</b>', x=0.5, y=0.5,
                                  font=dict(size=15, color='#212529'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Market comparison
            st.markdown('<div class="section-header">Market Price Comparison</div>', unsafe_allow_html=True)
            if vehicle_make in PREMIUM_MAKES:
                tier = 'Premium'
            elif vehicle_make in BUDGET_MAKES:
                tier = 'Budget'
            else:
                tier = 'Mid-range'

            market_data = MARKET_PRICES.get(dk, {})
            categories  = list(market_data.keys())
            market_vals = list(market_data.values())
            bar_colors  = ['#e03131' if c == tier else '#adb5bd' for c in categories]

            fig_market = go.Figure()
            fig_market.add_trace(go.Bar(
                name='Market Average', x=categories, y=market_vals,
                marker_color=bar_colors,
                text=[f'${v:,}' for v in market_vals], textposition='outside',
                textfont=dict(color='#495057')
            ))
            fig_market.add_hline(
                y=adj_cost, line_dash='dot', line_color='#e67700', line_width=2,
                annotation_text=f'Your estimate: ${adj_cost:,.0f}',
                annotation_font_color='#e67700', annotation_position='top left'
            )
            fig_market.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=250, margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(tickfont=dict(color='#495057')),
                yaxis=dict(tickfont=dict(color='#6c757d'), gridcolor='#e9ecef'),
                showlegend=False
            )
            st.plotly_chart(fig_market, use_container_width=True)

            market_avg = market_data.get(tier, adj_cost)
            diff = adj_cost - market_avg
            diff_pct = abs(diff / market_avg * 100) if market_avg else 0
            if diff < 0:
                st.markdown(f'<p style="color:#2f9e44;font-size:0.88rem">✅ Your estimate is <strong>${abs(diff):,.0f} ({diff_pct:.0f}%) below</strong> the {tier} market average for {DAMAGE_LABELS.get(dk,dk)}.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:#e67700;font-size:0.88rem">⚠️ Your estimate is <strong>${diff:,.0f} ({diff_pct:.0f}%) above</strong> the {tier} market average for {DAMAGE_LABELS.get(dk,dk)}.</p>', unsafe_allow_html=True)

            # ── SHAP feature importance ────────────────────────────────────
            if shap_values:
                st.markdown('<div class="section-header">🔍 What Drives This Cost Estimate?</div>', unsafe_allow_html=True)
                st.caption('SHAP values show how each feature pushes the cost above or below the baseline (log scale).')
                labels = [SHAP_LABELS.get(f, f) for f, _ in shap_values]
                values = [v for _, v in shap_values]
                bar_cols = ['#e03131' if v > 0 else '#1971c2' for v in values]
                fig_shap = go.Figure(go.Bar(
                    x=values[::-1], y=labels[::-1], orientation='h',
                    marker_color=bar_cols[::-1],
                    text=[f'+{v:.3f}' if v > 0 else f'{v:.3f}' for v in values[::-1]],
                    textposition='outside', textfont=dict(color='#495057', size=10),
                ))
                fig_shap.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=300, margin=dict(t=10, b=10, l=10, r=70),
                    xaxis=dict(showgrid=True, gridcolor='#e9ecef', zeroline=True,
                               zerolinecolor='#adb5bd', zerolinewidth=1.5,
                               tickfont=dict(color='#6c757d', size=10)),
                    yaxis=dict(tickfont=dict(color='#495057', size=11)),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.markdown(
                    '<p style="font-size:0.78rem;color:#6c757d">'
                    '<span style="color:#e03131;font-weight:700">■ Red</span> = increases estimated cost &nbsp;·&nbsp; '
                    '<span style="color:#1971c2;font-weight:700">■ Blue</span> = decreases estimated cost</p>',
                    unsafe_allow_html=True
                )

        # ── TAB 3: Report + PDF ────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="section-header">Generated Insurance Report</div>', unsafe_allow_html=True)
            if 'error' not in report:
                dmg   = report.get('damage', {})
                asmnt = report.get('assessment', {})
                sev_r = dmg.get('severity', '').lower()
                sev_c = {'minor': '#2f9e44', 'moderate': '#e67700', 'severe': '#e03131'}.get(sev_r, '#495057')

                st.markdown(f"""<div class="card">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
                        <div><div class="field-label">Vehicle</div>
                            <div class="field-value">{vehicle_year} {vehicle_make} {vehicle_model}</div></div>
                        <div><div class="field-label">Value</div>
                            <div class="field-value">${vehicle_value:,}</div></div>
                        <div><div class="field-label">Damage Type</div>
                            <div class="field-value">{dmg.get('type','—')}</div></div>
                        <div><div class="field-label">Severity</div>
                            <div class="field-value" style="color:{sev_c};font-weight:700">{dmg.get('severity','—').upper()}</div></div>
                        <div><div class="field-label">Location</div>
                            <div class="field-value">{dmg.get('location','—')}</div></div>
                        <div><div class="field-label">AI Confidence</div>
                            <div class="field-value">{asmnt.get('confidence_level','—')}</div></div>
                    </div>
                    <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #e9ecef">
                        <div class="field-label">Description</div>
                        <div class="field-value" style="margin-top:0.3rem">{dmg.get('description','—')}</div>
                    </div>
                    <div style="margin-top:0.8rem">
                        <div class="field-label">Recommendation</div>
                        <div class="field-value" style="margin-top:0.3rem">{asmnt.get('repair_recommendation','—')}</div>
                    </div>
                    <div style="margin-top:0.8rem">
                        <div class="field-label">Adjusted Cost ({severity})</div>
                        <div class="field-value" style="color:#2f9e44;font-weight:700;font-size:1.1rem;margin-top:0.3rem">
                            ${adj_cost:,.0f}
                            <span style="color:#6c757d;font-size:0.85rem;font-weight:400">
                            (${adj_cost*0.8:,.0f} – ${adj_cost*1.2:,.0f})</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Downloads
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dl1, dl2 = st.columns(2)
                with dl1:
                    try:
                        pdf_bytes = build_pdf(vehicle_info, cv_result, ml_result, report, severity, adj_cost)
                        st.download_button(
                            '⬇️ Download PDF Report', pdf_bytes,
                            file_name=f'damage_report_{timestamp}.pdf',
                            mime='application/pdf', use_container_width=True
                        )
                    except Exception as e:
                        st.error(f'PDF error: {e}')
                with dl2:
                    st.download_button(
                        '⬇️ Download JSON', json.dumps(report, indent=2),
                        file_name=f'damage_report_{timestamp}.json',
                        mime='application/json', use_container_width=True
                    )

                with st.expander('🔎 Raw JSON'):
                    st.json(report)
            else:
                st.warning('Report generation failed.')
                st.text(report.get('raw_output', ''))

        # ── TAB 4: Similar Cases ───────────────────────────────────────────
        with tab4:
            st.markdown('<div class="section-header">Similar Cases from NHTSA Database</div>', unsafe_allow_html=True)
            st.caption(f'Query: *"{query}"* — top 3 most similar complaints retrieved via FAISS cosine similarity')
            for i, case_text in enumerate(similar_cases, 1):
                parts = case_text.split(' | ')
                vehicle_part = parts[0].replace('Vehicle: ', '') if parts else ''
                complaint_part = ' | '.join(parts[2:]).replace('Complaint: ', '') if len(parts) > 2 else case_text
                st.markdown(f"""<div class="case-card">
                    <strong>Case #{i} — {vehicle_part}</strong><br>
                    {complaint_part[:400]}{'...' if len(complaint_part) > 400 else ''}
                </div>""", unsafe_allow_html=True)

    # ── Success banner ─────────────────────────────────────────────────────
    st.markdown(f"""<div class="success-banner">
        ✅ &nbsp; Analysis complete &nbsp;·&nbsp;
        Damage: <strong>{DAMAGE_LABELS.get(dk,dk)}</strong> ({conf:.0%}) &nbsp;·&nbsp;
        Adjusted Cost: <strong>${adj_cost:,.0f}</strong> ({severity}) &nbsp;·&nbsp;
        {datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

elif uploaded and 'analysis' not in st.session_state:
    with col_right:
        st.markdown("""<div style="height:300px;display:flex;flex-direction:column;justify-content:center;
                    align-items:center;text-align:center;padding:3rem;
                    background:#ffffff;border:1px dashed #dee2e6;
                    border-radius:16px;color:#6c757d">
            <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
            <div style="font-weight:600;color:#495057;margin-bottom:0.4rem">Ready to Analyze</div>
            <div style="font-size:0.85rem">Click "Analyze Damage" to run the full AI pipeline</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Analysis History ──────────────────────────────────────────────────────────
if st.session_state.get('history'):
    history = st.session_state['history']
    with st.expander(f'🗃️ Analysis History ({len(history)} {"entry" if len(history)==1 else "entries"})', expanded=False):
        col_clear, _ = st.columns([1, 5])
        with col_clear:
            if st.button('🗑️ Clear History', use_container_width=True):
                st.session_state['history'] = []
                st.rerun()

        for i, entry in enumerate(reversed(history)):
            idx = len(history) - i
            cv  = entry['cv_result']
            ml  = entry['ml_result']
            vi  = entry['vehicle_info']
            dk  = cv['damage_class']
            col_img, col_info = st.columns([1, 4])
            with col_img:
                st.image(entry['thumb'], width=110)
            with col_info:
                st.markdown(f"""<div class="card" style="margin-bottom:0.4rem;padding:1rem">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                        <span style="font-weight:700;color:#212529">#{idx} &nbsp; {DAMAGE_ICONS.get(dk,'')} {DAMAGE_LABELS.get(dk,dk)}</span>
                        <span style="font-size:0.78rem;color:#868e96">{entry['timestamp']} &nbsp;·&nbsp; {entry['filename']}</span>
                    </div>
                    <div style="display:flex;gap:2rem;font-size:0.85rem;color:#495057">
                        <span>🚘 {vi['year']} {vi['make']} {vi['model']}</span>
                        <span>💰 <strong style="color:#2f9e44">${ml['estimated_cost_usd']:,.0f}</strong></span>
                        <span>🎯 Confidence: <strong>{cv['confidence']:.0%}</strong></span>
                        <span>💵 Vehicle value: ${vi['value_usd']:,}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            if i < len(history) - 1:
                st.markdown('<hr style="border:none;border-top:1px solid #e9ecef;margin:0.3rem 0">', unsafe_allow_html=True)

st.caption("AI Applications Final Project — CV + ML + NLP | Arben Mustafi | 2026")
