import streamlit as st

st.set_page_config(page_title='About', page_icon='ℹ️', layout='wide')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.page-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px; padding: 2.5rem; margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}
.page-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.page-header p  { color: #adb5bd; margin: 0; }
.info-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.8rem; margin-bottom: 1.2rem;
}
.info-card h3 { color: #ff4b4b; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 1rem 0; }
.tech-pill {
    display: inline-block; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
    padding: 0.3rem 0.8rem; font-size: 0.82rem; color: #ced4da;
    margin: 0.2rem;
}
.stat-box {
    background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.2);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.stat-num  { font-size: 2rem; font-weight: 700; color: #ff4b4b; line-height: 1; }
.stat-label { color: #868e96; font-size: 0.82rem; margin-top: 0.3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>ℹ️ About This Project</h1>
    <p>AI Applications Final Project — combining all three learning blocks into one end-to-end system.</p>
</div>""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>📌 Project Overview</h3>
        <p style="color:#ced4da; font-size:0.92rem; line-height:1.7">
        This application demonstrates an end-to-end AI pipeline for automated vehicle damage assessment.
        A user uploads a photo of a damaged car, and the system automatically identifies the damage type,
        estimates repair costs, and generates a structured insurance claim report — all within seconds.
        </p>
        <p style="color:#ced4da; font-size:0.92rem; line-height:1.7">
        The project integrates all three AI blocks covered in the module:
        <strong style="color:white">Computer Vision</strong>, <strong style="color:white">Machine Learning</strong>
        and <strong style="color:white">Natural Language Processing</strong>.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🗃️ Datasets</h3>
        <table style="width:100%; color:#ced4da; font-size:0.88rem; border-collapse:collapse">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <td style="padding:0.6rem 0; color:#fff; font-weight:600">VehiDE</td>
                <td style="padding:0.6rem 0">13,945 car damage images · 7 classes · Kaggle</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <td style="padding:0.6rem 0; color:#fff; font-weight:600">Car Insurance Claims</td>
                <td style="padding:0.6rem 0">10,302 policies · claim amounts · Kaggle</td>
            </tr>
            <tr>
                <td style="padding:0.6rem 0; color:#fff; font-weight:600">NHTSA ODI Complaints</td>
                <td style="padding:0.6rem 0">Vehicle complaint texts · public API · RAG knowledge base</td>
            </tr>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🛠️ Tech Stack</h3>
        <span class="tech-pill">Python 3.14</span>
        <span class="tech-pill">PyTorch</span>
        <span class="tech-pill">EfficientNet-B0</span>
        <span class="tech-pill">timm</span>
        <span class="tech-pill">XGBoost</span>
        <span class="tech-pill">scikit-learn</span>
        <span class="tech-pill">SHAP</span>
        <span class="tech-pill">OpenAI GPT-4o-mini</span>
        <span class="tech-pill">sentence-transformers</span>
        <span class="tech-pill">FAISS</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">pandas</span>
        <span class="tech-pill">NumPy</span>
    </div>""", unsafe_allow_html=True)

with col2:
    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="stat-box"><div class="stat-num">7</div><div class="stat-label">Damage Classes</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat-box"><div class="stat-num">3</div><div class="stat-label">AI Blocks</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    s3, s4 = st.columns(2)
    with s3:
        st.markdown('<div class="stat-box"><div class="stat-num">13k</div><div class="stat-label">Training Images</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="stat-box"><div class="stat-num">~50%</div><div class="stat-label">CV Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h3>📊 Model Performance</h3>
        <table style="width:100%; color:#ced4da; font-size:0.88rem; border-collapse:collapse">
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <td style="padding:0.5rem 0; color:#868e96">CV Val Accuracy</td>
                <td style="padding:0.5rem 0; color:#fff; text-align:right">~50%</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <td style="padding:0.5rem 0; color:#868e96">ML Test MAE</td>
                <td style="padding:0.5rem 0; color:#fff; text-align:right">~$2,957</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.08)">
                <td style="padding:0.5rem 0; color:#868e96">ML Model</td>
                <td style="padding:0.5rem 0; color:#fff; text-align:right">XGBoost</td>
            </tr>
            <tr>
                <td style="padding:0.5rem 0; color:#868e96">NLP Model</td>
                <td style="padding:0.5rem 0; color:#fff; text-align:right">GPT-4o-mini</td>
            </tr>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>👤 Author</h3>
        <p style="color:#ced4da; font-size:0.92rem; margin:0">
        <strong style="color:white">Arben Mustafi</strong><br>
        AI Applications Module · 2026<br>
        <a href="https://github.com/Arben-ai/car-damage-assessment" style="color:#74c0fc">
        github.com/Arben-ai/car-damage-assessment</a>
        </p>
    </div>""", unsafe_allow_html=True)
