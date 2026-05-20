import sys, os, json, io, textwrap
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import numpy as np

from src.cv_model import load_model, predict as cv_predict
from src.ml_model import load_artifacts, predict_cost
from src.nlp_report import load_rag_index, generate_report, retrieve

MODELS_DIR    = Path(__file__).parent.parent / 'models'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

DAMAGE_LABELS = {
    'broken_glass':  'Broken Glass',   'broken_lights': 'Broken Lights',
    'dents':         'Dents',          'lost_parts':    'Missing Parts',
    'punctured':     'Puncture',       'scratch':       'Scratch',
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
    transition: border-color 0.2s;
}
.step-card:hover { border-color: rgba(255,75,75,0.3); }
.step-num {
    background: linear-gradient(135deg, #ff4b4b, #ff6b6b); color: white;
    font-weight: 800; font-size: 0.8rem; width: 30px; height: 30px;
    border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; margin-bottom: 0.6rem; box-shadow: 0 4px 12px rgba(255,75,75,0.3);
}
.step-title { color: #fff; font-weight: 700; font-size: 0.92rem; margin-bottom: 0.3rem; }
.step-sub   { color: #868e96; font-size: 0.78rem; line-height: 1.5; }
.upload-zone {
    background: rgba(255,255,255,0.02); border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 1rem;
}
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
.dl-row { display: flex; gap: 0.8rem; margin-top: 1rem; }
.sidebar-section { color: #868e96; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin: 1.2rem 0 0.5rem 0; }
</style>
"""

st.set_page_config(page_title='Car Damage Assessment', page_icon='🚗', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
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
    st.markdown('<div class="sidebar-section">Details</div>', unsafe_allow_html=True)
    vehicle_make  = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Volkswagen', 'Mercedes', 'Audi', 'Hyundai', 'Other'])
    vehicle_model = st.text_input('Model', value='Corolla')
    vehicle_year  = st.number_input('Year', min_value=2000, max_value=2026, value=2020)
    vehicle_value = st.number_input('Value (USD)', min_value=1000, max_value=200000, value=18000, step=500)
    vehicle_age   = 2026 - vehicle_year
    damage_loc    = st.selectbox('Damage Location', [
        'front bumper', 'rear bumper', 'hood', 'front left door', 'front right door',
        'rear left door', 'rear right door', 'windshield', 'side panel', 'roof', 'other'
    ])
    st.markdown('<div class="sidebar-section">API</div>', unsafe_allow_html=True)
    api_key = st.text_input('OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))
    if api_key:
        st.success('✓ API key ready')
    else:
        st.warning('Enter API key to generate report')

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AI-Powered · CV + ML + NLP</div>
    <h1>Car Damage Assessment</h1>
    <p>Upload a photo of a damaged vehicle to instantly receive damage classification,
    repair cost estimation and a fully structured insurance claim report — powered by three AI models.</p>
</div>""", unsafe_allow_html=True)

# ── Pipeline overview ─────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
for col, num, icon, title, sub in [
    (c1, '1', '📸', 'Damage Detection', 'EfficientNet-B0 identifies the damage type with confidence score'),
    (c2, '2', '💰', 'Cost Estimation',  'XGBoost predicts repair costs using vehicle & damage features'),
    (c3, '3', '📄', 'Insurance Report', 'RAG + GPT-4o-mini generates a structured claim document'),
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

# ── Analyze button ────────────────────────────────────────────────────────────
analyze = st.button('🔍  Analyze Damage', type='primary', use_container_width=True,
                    disabled=not bool(uploaded))

if uploaded and analyze:
    if not api_key:
        st.error('Please enter your OpenAI API Key in the sidebar.')
        st.stop()

    # Run pipeline
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
        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(['🔍 Damage', '💰 Cost', '📄 Report', '🗂️ Similar Cases'])

        # TAB 1 — Damage
        with tab1:
            st.markdown('<div class="section-header">Computer Vision Result</div>', unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=conf * 100,
                number={'suffix': '%', 'font': {'size': 32, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#444', 'tickfont': {'color': '#666'}},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40],  'color': 'rgba(255,255,255,0.04)'},
                        {'range': [40, 70], 'color': 'rgba(255,255,255,0.04)'},
                        {'range': [70, 100],'color': 'rgba(255,255,255,0.04)'},
                    ],
                    'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': conf*100}
                },
                title={'text': 'Model Confidence', 'font': {'color': '#868e96', 'size': 13}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=220, margin=dict(t=30, b=0, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Damage badge
            st.markdown(f"""<div class="card">
                <div class="damage-badge" style="background:rgba(0,0,0,0.3);border:1px solid {color}40;color:{color}">
                    {DAMAGE_ICONS.get(dk,'🔍')} {DAMAGE_LABELS.get(dk, dk)}
                </div>
                <div style="color:#868e96;font-size:0.82rem">Detected with <strong style="color:white">{conf:.1%}</strong> confidence</div>
            </div>""", unsafe_allow_html=True)

            # Probability chart
            probs = cv_result['class_probs']
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            labels = [f"{DAMAGE_ICONS.get(k,'•')} {DAMAGE_LABELS.get(k,k)}" for k in sorted_probs]
            values = list(sorted_probs.values())
            bar_colors = [DAMAGE_COLORS.get(k, '#444') for k in sorted_probs]

            fig_bar = go.Figure(go.Bar(
                x=values, y=labels, orientation='h',
                marker=dict(color=bar_colors, opacity=0.85),
                text=[f'{v:.0%}' for v in values], textposition='outside',
                textfont=dict(color='#adb5bd', size=11)
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=260, margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
                yaxis=dict(tickfont=dict(color='#ced4da', size=11)),
                bargap=0.35
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 2 — Cost
        with tab2:
            st.markdown('<div class="section-header">Repair Cost Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="card" style="text-align:center">
                <div class="cost-big">${cost:,.0f}</div>
                <div class="cost-sub">Estimated repair cost</div>
                <div style="margin-top:0.8rem;color:#868e96;font-size:0.85rem">
                    Range: <strong style="color:#ffd43b">${ml_result['cost_range_low']:,.0f}</strong>
                    –
                    <strong style="color:#ffd43b">${ml_result['cost_range_high']:,.0f}</strong>
                </div>
            </div>""", unsafe_allow_html=True)

            # Cost breakdown donut
            labor_pct = 0.45 + (hash(dk) % 10) * 0.01
            parts_pct = 1 - labor_pct - 0.08
            other_pct = 0.08
            fig_donut = go.Figure(go.Pie(
                labels=['Labor', 'Parts & Materials', 'Other'],
                values=[labor_pct * cost, parts_pct * cost, other_pct * cost],
                hole=0.65,
                marker=dict(colors=['#ff4b4b', '#51cf66', '#ffd43b'],
                            line=dict(color='rgba(0,0,0,0)', width=0)),
                textinfo='label+percent',
                textfont=dict(color='white', size=11),
                insidetextorientation='radial'
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(color='#adb5bd'), bgcolor='rgba(0,0,0,0)'),
                annotations=[dict(text=f'<b>${cost:,.0f}</b>', x=0.5, y=0.5,
                                  font=dict(size=16, color='white'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric('Labor', f'${labor_pct*cost:,.0f}', f'{labor_pct:.0%}')
            with col_b:
                st.metric('Parts', f'${parts_pct*cost:,.0f}', f'{parts_pct:.0%}')
            with col_c:
                st.metric('Other', f'${other_pct*cost:,.0f}', f'{other_pct:.0%}')

        # TAB 3 — Report
        with tab3:
            st.markdown('<div class="section-header">Generated Insurance Report</div>', unsafe_allow_html=True)
            if 'error' not in report:
                dmg   = report.get('damage', {})
                asmnt = report.get('assessment', {})
                sev   = dmg.get('severity', '').lower()
                sev_color = SEVERITY_COLOR.get(sev, '#fff')

                st.markdown(f"""<div class="card">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
                        <div>
                            <div class="field-label">Vehicle</div>
                            <div class="field-value">{vehicle_year} {vehicle_make} {vehicle_model}</div>
                        </div>
                        <div>
                            <div class="field-label">Value</div>
                            <div class="field-value">${vehicle_value:,}</div>
                        </div>
                        <div>
                            <div class="field-label">Damage Type</div>
                            <div class="field-value">{dmg.get('type','—')}</div>
                        </div>
                        <div>
                            <div class="field-label">Severity</div>
                            <div class="field-value" style="color:{sev_color};font-weight:600">{dmg.get('severity','—').upper()}</div>
                        </div>
                        <div>
                            <div class="field-label">Location</div>
                            <div class="field-value">{dmg.get('location','—')}</div>
                        </div>
                        <div>
                            <div class="field-label">Confidence</div>
                            <div class="field-value">{asmnt.get('confidence_level','—')}</div>
                        </div>
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
                        <div class="field-label">Estimated Cost</div>
                        <div class="field-value" style="color:#51cf66;font-weight:700;font-size:1.1rem;margin-top:0.3rem">
                            ${asmnt.get('estimated_cost_usd',0):,} &nbsp;
                            <span style="color:#868e96;font-size:0.85rem;font-weight:400">
                            ({asmnt.get('cost_range','—')})</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Download buttons
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                json_str = json.dumps(report, indent=2)

                txt_report = textwrap.dedent(f"""
                VEHICLE DAMAGE INSURANCE REPORT
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                {'='*50}

                VEHICLE
                  Make/Model : {vehicle_year} {vehicle_make} {vehicle_model}
                  Value      : ${vehicle_value:,}
                  Location   : {damage_loc}

                DAMAGE ASSESSMENT
                  Type       : {dmg.get('type','—')}
                  Severity   : {dmg.get('severity','—').upper()}
                  Description: {dmg.get('description','—')}

                COST ESTIMATE
                  Estimated  : ${asmnt.get('estimated_cost_usd',0):,}
                  Range      : {asmnt.get('cost_range','—')}
                  Confidence : {asmnt.get('confidence_level','—')}

                RECOMMENDATION
                  {asmnt.get('repair_recommendation','—')}

                NOTES
                  {report.get('notes','—')}
                {'='*50}
                AI Applications Final Project | Arben Mustafi | 2026
                """).strip()

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button('⬇️ Download JSON', json_str,
                        file_name=f'damage_report_{timestamp}.json', mime='application/json', use_container_width=True)
                with dl2:
                    st.download_button('⬇️ Download Report (.txt)', txt_report,
                        file_name=f'damage_report_{timestamp}.txt', mime='text/plain', use_container_width=True)

                with st.expander('🔎 Raw JSON'):
                    st.json(report)
            else:
                st.warning('Report generation failed.')
                st.text(report.get('raw_output', ''))

        # TAB 4 — Similar Cases
        with tab4:
            st.markdown('<div class="section-header">Similar Cases from NHTSA Database</div>', unsafe_allow_html=True)
            st.caption(f'Query: *"{query}"* — top 3 most similar complaints retrieved via FAISS')
            for i, case_text in enumerate(similar_cases, 1):
                # Parse the structured text
                parts = case_text.split(' | ')
                vehicle_part = parts[0].replace('Vehicle: ', '') if parts else ''
                complaint_part = ' | '.join(parts[2:]).replace('Complaint: ', '') if len(parts) > 2 else case_text
                st.markdown(f"""<div class="case-card">
                    <strong>Case #{i} — {vehicle_part}</strong><br>
                    {complaint_part[:400]}{'...' if len(complaint_part) > 400 else ''}
                </div>""", unsafe_allow_html=True)

    # Success banner (full width below columns)
    st.markdown(f"""<div class="success-banner">
        ✅ &nbsp; Analysis complete &nbsp;·&nbsp;
        Damage: <strong>{DAMAGE_LABELS.get(dk,dk)}</strong> ({conf:.0%}) &nbsp;·&nbsp;
        Cost: <strong>${cost:,.0f}</strong> &nbsp;·&nbsp;
        Report generated at {datetime.now().strftime('%H:%M:%S')}
    </div>""", unsafe_allow_html=True)

elif uploaded and not analyze:
    with col_right:
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
                    align-items:center;text-align:center;padding:3rem;
                    background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                    border-radius:16px;color:#868e96">
            <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
            <div style="font-weight:600;color:#adb5bd;margin-bottom:0.4rem">Ready to Analyze</div>
            <div style="font-size:0.85rem">Click "Analyze Damage" to run the AI pipeline</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("AI Applications Final Project — CV + ML + NLP | Arben Mustafi | 2026")
