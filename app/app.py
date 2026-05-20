import sys, os, json, io, textwrap
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import numpy as np
from fpdf import FPDF

from src.cv_model import load_model, predict as cv_predict
from src.ml_model import load_artifacts, predict_cost
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
[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid rgba(255,255,255,0.06); }
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px; padding: 3.5rem 3rem; margin-bottom: 2rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,75,75,0.12); color: #ff6b6b;
    border: 1px solid rgba(255,75,75,0.25); border-radius: 20px;
    padding: 0.35rem 1.1rem; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1.2rem;
}
.hero h1 { color: #fff; font-size: 3rem; font-weight: 800; margin: 0 0 0.8rem 0; line-height: 1.1; }
.hero p  { color: #868e96; font-size: 1.05rem; margin: 0 auto; max-width: 600px; line-height: 1.6; }
.step-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center; height: 100%;
}
.step-num {
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b); color: white;
    font-weight: 800; font-size: 0.8rem; width: 30px; height: 30px;
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; margin-bottom: 0.6rem;
}
.step-title { color: #fff; font-weight: 700; font-size: 0.92rem; margin-bottom: 0.3rem; }
.step-sub   { color: #868e96; font-size: 0.78rem; line-height: 1.5; }
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #ff4b4b; margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,75,75,0.15);
}
.card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.4rem; margin-bottom: 0.8rem;
}
.damage-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    border-radius: 10px; padding: 0.5rem 1.2rem; font-size: 1.1rem; font-weight: 700;
    margin-bottom: 0.8rem;
}
.field-label { color: #868e96; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.2rem; }
.field-value { color: #fff; font-size: 0.95rem; line-height: 1.5; }
.cost-big    { font-size: 3rem; font-weight: 800; color: #51cf66; line-height: 1; }
.cost-sub    { color: #868e96; font-size: 0.88rem; margin-top: 0.4rem; }
.case-card {
    background: rgba(116,192,252,0.04); border: 1px solid rgba(116,192,252,0.15);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    font-size: 0.83rem; color: #ced4da; line-height: 1.6;
}
.case-card strong { color: #74c0fc; }
.success-banner {
    background: linear-gradient(90deg, rgba(81,207,102,0.12), rgba(81,207,102,0.04));
    border: 1px solid rgba(81,207,102,0.25); border-radius: 12px;
    padding: 1rem 1.5rem; text-align: center; margin-top: 1rem;
    color: #51cf66; font-weight: 600; font-size: 0.95rem;
}
.market-better { color: #51cf66; font-weight: 700; }
.market-worse  { color: #ff4b4b; font-weight: 700; }
</style>
"""

# ── PDF generator ─────────────────────────────────────────────────────────────
def _s(text):
    """Strip characters outside Latin-1 so Helvetica doesn't crash."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def build_pdf(vehicle_info, cv_result, ml_result, report, severity, adjusted_cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header bar
    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_xy(0, 8)
    pdf.cell(210, 10, 'VEHICLE DAMAGE ASSESSMENT REPORT', align='C', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(210, 7, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}   |   AI Applications Final Project", align='C', ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(46)

    def section(title):
        pdf.set_fill_color(240, 240, 248)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(48, 43, 99)
        pdf.cell(0, 8, f'  {title}', fill=True, ln=True)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(2)

    def row(label, value, bold_val=False):
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(55, 7, _s(label))
        pdf.set_text_color(30, 30, 30)
        pdf.set_font('Helvetica', 'B' if bold_val else '', 9)
        pdf.multi_cell(0, 7, _s(value), new_x='LMARGIN', new_y='NEXT')

    # Vehicle info
    section('VEHICLE INFORMATION')
    vi = vehicle_info
    row('Make / Model', f"{vi.get('year','')} {vi.get('make','')} {vi.get('model','')}")
    row('Estimated Value', f"${vi.get('value_usd',0):,} USD")
    row('Damage Location', vi.get('location', '—'))
    pdf.ln(3)

    # Damage assessment
    section('DAMAGE ASSESSMENT  (Computer Vision)')
    dk = cv_result['damage_class']
    row('Damage Type', DAMAGE_LABELS.get(dk, dk), bold_val=True)
    row('Severity', severity, bold_val=True)
    row('AI Confidence', f"{cv_result['confidence']:.1%}")
    dmg = report.get('damage', {})
    if dmg.get('description'):
        row('Description', dmg['description'])
    pdf.ln(3)

    # Cost estimate
    section('REPAIR COST ESTIMATE  (ML Model + Severity Adjustment)')
    row('Estimated Cost', f"${adjusted_cost:,.0f} USD", bold_val=True)
    row('Cost Range', f"${adjusted_cost*0.8:,.0f} – ${adjusted_cost*1.2:,.0f} USD")
    asmnt = report.get('assessment', {})
    if asmnt.get('repair_recommendation'):
        row('Recommendation', asmnt['repair_recommendation'])
    if asmnt.get('confidence_level'):
        row('Confidence Level', asmnt['confidence_level'])
    pdf.ln(3)

    # Class probabilities
    section('CLASS PROBABILITIES  (All Damage Types)')
    probs = sorted(cv_result['class_probs'].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in probs:
        label = DAMAGE_LABELS.get(cls, cls)
        filled = max(1, int(prob * 78))
        empty  = max(1, 79 - filled)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(50, 6, _s(label))
        pdf.set_fill_color(48, 43, 99)
        pdf.cell(filled, 4, '', fill=True)
        pdf.set_fill_color(220, 220, 235)
        pdf.cell(empty, 4, '', fill=True)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(20, 6, f'{prob:.1%}', ln=True)
    pdf.ln(3)

    # Notes
    notes = report.get('notes', '')
    if notes:
        section('NOTES')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 6, _s(notes), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)

    # Footer
    pdf.set_y(-20)
    pdf.set_fill_color(240, 240, 248)
    pdf.rect(0, pdf.get_y(), 210, 20, 'F')
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, 'This report was generated automatically by an AI system and is intended for informational purposes only.', align='C', ln=True)
    pdf.cell(0, 6, 'Car Damage Assessment  |  Arben Mustafi  |  2026  |  car-damage-assessment.streamlit.app', align='C')

    return bytes(pdf.output())


# ── App setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Car Damage Assessment', page_icon='🚗', layout='wide')
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
    st.markdown("## 🚘 Vehicle Info")
    vehicle_make  = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Volkswagen', 'Mercedes', 'Audi', 'Hyundai', 'Other'])
    vehicle_model = st.text_input('Model', value='Corolla')
    vehicle_year  = st.number_input('Year', min_value=2000, max_value=2026, value=2020)
    vehicle_value = st.number_input('Value (USD)', min_value=1000, max_value=200000, value=18000, step=500)
    vehicle_age   = 2026 - vehicle_year
    damage_loc    = st.selectbox('Damage Location', [
        'front bumper', 'rear bumper', 'hood', 'front left door', 'front right door',
        'rear left door', 'rear right door', 'windshield', 'side panel', 'roof', 'other'
    ])
    st.markdown("---")
    api_key = st.text_input('🔑 OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
    if api_key:
        st.success('✓ API key ready')

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

    with st.status('Running AI pipeline...', expanded=True) as status:
        st.write('📸 **Step 1** — Classifying damage type...')
        cv_result = cv_predict(image, cv_model)
        st.write('💰 **Step 2** — Estimating repair cost...')
        vehicle_info = {'make': vehicle_make, 'model': vehicle_model,
                        'year': vehicle_year, 'value_usd': vehicle_value, 'location': damage_loc}
        ml_result = predict_cost(cv_result, vehicle_age, vehicle_value, ml_model, imputer, feature_names)
        st.write('🔍 **Step 3** — Retrieving similar cases...')
        query = f"{cv_result['damage_class']} {vehicle_make} {damage_loc}"
        similar_cases = retrieve(query, index, texts, embedder, k=3)
        st.write('📄 **Step 4** — Generating insurance report...')
        report = generate_report(cv_result, ml_result, vehicle_info, index, texts, embedder, api_key)
        status.update(label='✅ Analysis complete!', state='complete')

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
                number={'suffix': '%', 'font': {'size': 32, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#444', 'tickfont': {'color': '#666'}},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                    'steps': [{'range': [0, 100], 'color': 'rgba(255,255,255,0.04)'}],
                    'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': conf*100}
                },
                title={'text': 'Model Confidence', 'font': {'color': '#868e96', 'size': 13}}
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    height=220, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""<div class="card">
                <div class="damage-badge" style="background:rgba(0,0,0,0.3);border:1px solid {color}40;color:{color}">
                    {DAMAGE_ICONS.get(dk,'🔍')} {DAMAGE_LABELS.get(dk, dk)}
                </div>
                <div style="color:#868e96;font-size:0.82rem">Detected with <strong style="color:white">{conf:.1%}</strong> confidence</div>
            </div>""", unsafe_allow_html=True)

            probs = cv_result['class_probs']
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            fig_bar = go.Figure(go.Bar(
                x=list(sorted_probs.values()),
                y=[f"{DAMAGE_ICONS.get(k,'•')} {DAMAGE_LABELS.get(k,k)}" for k in sorted_probs],
                orientation='h',
                marker=dict(color=[DAMAGE_COLORS.get(k,'#444') for k in sorted_probs], opacity=0.85),
                text=[f'{v:.0%}' for v in sorted_probs.values()], textposition='outside',
                textfont=dict(color='#adb5bd', size=11)
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=260, margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
                yaxis=dict(tickfont=dict(color='#ced4da', size=11)), bargap=0.35
            )
            st.plotly_chart(fig_bar, use_container_width=True)

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
            sev_color = {'Minor': '#51cf66', 'Moderate': '#ffd43b', 'Severe': '#ff4b4b', 'Total Loss': '#e03131'}[severity]
            st.markdown(f"""<div class="card" style="text-align:center">
                <div class="cost-big">${adj_cost:,.0f}</div>
                <div class="cost-sub">
                    Severity: <strong style="color:{sev_color}">{severity}</strong> &nbsp;·&nbsp;
                    Range: <strong style="color:#ffd43b">${adj_cost*0.8:,.0f} – ${adj_cost*1.2:,.0f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

            # Cost breakdown donut
            labor_pct = 0.45; parts_pct = 0.47; other_pct = 0.08
            fig_donut = go.Figure(go.Pie(
                labels=['Labor', 'Parts & Materials', 'Other'],
                values=[labor_pct * adj_cost, parts_pct * adj_cost, other_pct * adj_cost],
                hole=0.65,
                marker=dict(colors=['#ff4b4b', '#51cf66', '#ffd43b'],
                            line=dict(color='rgba(0,0,0,0)', width=0)),
                textinfo='label+percent', textfont=dict(color='white', size=11),
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color='#adb5bd'), bgcolor='rgba(0,0,0,0)'),
                annotations=[dict(text=f'<b>${adj_cost:,.0f}</b>', x=0.5, y=0.5,
                                  font=dict(size=15, color='white'), showarrow=False)]
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
            bar_colors  = ['#ff4b4b' if c == tier else '#444' for c in categories]

            fig_market = go.Figure()
            fig_market.add_trace(go.Bar(
                name='Market Average', x=categories, y=market_vals,
                marker_color=bar_colors,
                text=[f'${v:,}' for v in market_vals], textposition='outside',
                textfont=dict(color='#adb5bd')
            ))
            fig_market.add_hline(
                y=adj_cost, line_dash='dot', line_color='#ffd43b', line_width=2,
                annotation_text=f'Your estimate: ${adj_cost:,.0f}',
                annotation_font_color='#ffd43b', annotation_position='top left'
            )
            fig_market.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=250, margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(tickfont=dict(color='#ced4da')),
                yaxis=dict(tickfont=dict(color='#666'), gridcolor='rgba(255,255,255,0.05)'),
                showlegend=False
            )
            st.plotly_chart(fig_market, use_container_width=True)

            market_avg = market_data.get(tier, adj_cost)
            diff = adj_cost - market_avg
            diff_pct = abs(diff / market_avg * 100) if market_avg else 0
            if diff < 0:
                st.markdown(f'<p style="color:#51cf66;font-size:0.88rem">✅ Your estimate is <strong>${abs(diff):,.0f} ({diff_pct:.0f}%) below</strong> the {tier} market average for {DAMAGE_LABELS.get(dk,dk)}.</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:#ffd43b;font-size:0.88rem">⚠️ Your estimate is <strong>${diff:,.0f} ({diff_pct:.0f}%) above</strong> the {tier} market average for {DAMAGE_LABELS.get(dk,dk)}.</p>', unsafe_allow_html=True)

        # ── TAB 3: Report + PDF ────────────────────────────────────────────
        with tab3:
            st.markdown('<div class="section-header">Generated Insurance Report</div>', unsafe_allow_html=True)
            if 'error' not in report:
                dmg   = report.get('damage', {})
                asmnt = report.get('assessment', {})
                sev_r = dmg.get('severity', '').lower()
                sev_c = SEVERITY_COLOR.get(sev_r, '#fff')

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
                    <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.07)">
                        <div class="field-label">Description</div>
                        <div class="field-value" style="margin-top:0.3rem">{dmg.get('description','—')}</div>
                    </div>
                    <div style="margin-top:0.8rem">
                        <div class="field-label">Recommendation</div>
                        <div class="field-value" style="margin-top:0.3rem">{asmnt.get('repair_recommendation','—')}</div>
                    </div>
                    <div style="margin-top:0.8rem">
                        <div class="field-label">Adjusted Cost ({severity})</div>
                        <div class="field-value" style="color:#51cf66;font-weight:700;font-size:1.1rem;margin-top:0.3rem">
                            ${adj_cost:,.0f}
                            <span style="color:#868e96;font-size:0.85rem;font-weight:400">
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

elif uploaded and not analyze:
    with col_right:
        st.markdown("""<div style="height:300px;display:flex;flex-direction:column;justify-content:center;
                    align-items:center;text-align:center;padding:3rem;
                    background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                    border-radius:16px;color:#868e96">
            <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
            <div style="font-weight:600;color:#adb5bd;margin-bottom:0.4rem">Ready to Analyze</div>
            <div style="font-size:0.85rem">Click "Analyze Damage" to run the full AI pipeline</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Applications Final Project — CV + ML + NLP | Arben Mustafi | 2026")
