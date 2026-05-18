import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import json

from src.cv_model import load_model, predict as cv_predict
from src.ml_model import load_artifacts, predict_cost
from src.nlp_report import load_rag_index, generate_report

MODELS_DIR = Path(__file__).parent.parent / 'models'
PROCESSED_DIR = Path(__file__).parent.parent / 'data' / 'processed'

DAMAGE_LABELS = {
    'broken_glass': 'Glasbruch',
    'broken_lights': 'Lampenbruch',
    'dents': 'Delle',
    'lost_parts': 'Fehlende Teile',
    'punctured': 'Loch/Perforation',
    'scratch': 'Kratzer',
    'torn': 'Riss/Einriss'
}

st.set_page_config(
    page_title='Car Damage Assessment',
    page_icon='🚗',
    layout='wide'
)

st.title('Car Damage Assessment & Repair Cost Estimation')
st.markdown('Upload a photo of a damaged vehicle to receive an AI-powered damage analysis and insurance report.')

@st.cache_resource
def load_all_models():
    device = 'cpu'
    cv_model = load_model(str(MODELS_DIR / 'efficientnet_b0_best.pth'), device)
    ml_model, imputer, feature_names = load_artifacts(str(MODELS_DIR))
    index, texts, embedder = load_rag_index(str(MODELS_DIR), str(PROCESSED_DIR))
    return cv_model, ml_model, imputer, feature_names, index, texts, embedder

with st.spinner('Loading models...'):
    cv_model, ml_model, imputer, feature_names, index, texts, embedder = load_all_models()

# Sidebar — vehicle info
with st.sidebar:
    st.header('Vehicle Information')
    vehicle_make  = st.selectbox('Make', ['Toyota', 'Honda', 'Ford', 'BMW', 'Volkswagen', 'Other'])
    vehicle_model = st.text_input('Model', value='Corolla')
    vehicle_year  = st.number_input('Year', min_value=2000, max_value=2026, value=2020)
    vehicle_value = st.number_input('Estimated vehicle value (USD)', min_value=1000, max_value=200000, value=18000, step=500)
    vehicle_age   = 2026 - vehicle_year
    damage_loc    = st.selectbox('Damage location', ['front bumper', 'rear bumper', 'hood', 'front left door',
                                                      'front right door', 'rear left door', 'rear right door',
                                                      'windshield', 'side panel', 'roof', 'other'])
    api_key = st.text_input('OpenAI API Key', type='password', value=os.getenv('OPENAI_API_KEY', ''))

# Main — image upload
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader('Upload vehicle damage photo', type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption='Uploaded image', use_container_width=True)

if uploaded and st.button('Analyze Damage', type='primary', use_container_width=True):
    if not api_key:
        st.error('Please enter your OpenAI API Key in the sidebar.')
        st.stop()

    with st.spinner('Running damage analysis...'):

        # Step 1: CV
        cv_result = cv_predict(image, cv_model)

        # Step 2: ML
        vehicle_info = {
            'make': vehicle_make, 'model': vehicle_model,
            'year': vehicle_year, 'value_usd': vehicle_value,
            'location': damage_loc
        }
        ml_result = predict_cost(cv_result, vehicle_age, vehicle_value, ml_model, imputer, feature_names)

        # Step 3: NLP
        report = generate_report(cv_result, ml_result, vehicle_info, index, texts, embedder, api_key)

    # Display results
    with col2:
        st.subheader('Analysis Results')

        # CV result
        st.markdown('#### Computer Vision — Damage Detection')
        damage_label = DAMAGE_LABELS.get(cv_result['damage_class'], cv_result['damage_class'])
        st.metric('Detected Damage Type', damage_label)
        st.metric('Confidence', f"{cv_result['confidence']:.1%}")

        st.markdown('**Class probabilities:**')
        probs_df = {DAMAGE_LABELS.get(k, k): v for k, v in cv_result['class_probs'].items()}
        st.bar_chart(probs_df)

    st.divider()
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown('#### ML — Repair Cost Estimation')
        st.metric('Estimated Cost', f"${ml_result['estimated_cost_usd']:,.0f}")
        st.metric('Cost Range', f"${ml_result['cost_range_low']:,.0f} – ${ml_result['cost_range_high']:,.0f}")

    with col4:
        st.markdown('#### NLP — Generated Insurance Report')
        if 'error' not in report:
            if 'damage' in report:
                st.markdown(f"**Damage:** {report['damage'].get('type', '')} — {report['damage'].get('severity', '')}")
                st.markdown(f"**Description:** {report['damage'].get('description', '')}")
            if 'assessment' in report:
                st.markdown(f"**Recommendation:** {report['assessment'].get('repair_recommendation', '')}")
            if 'notes' in report:
                st.markdown(f"**Notes:** {report.get('notes', '')}")

            with st.expander('Full JSON Report'):
                st.json(report)
        else:
            st.warning('Report generation returned unstructured output.')
            st.text(report.get('raw_output', ''))

    st.success('Analysis complete.')

st.divider()
st.caption('AI Applications Final Project — CV + ML + NLP | Arben Mustafi | 2026')
