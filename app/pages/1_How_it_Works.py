import streamlit as st

st.set_page_config(page_title='How it Works', page_icon='⚙️', layout='wide')

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
.block-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.8rem; height: 100%;
}
.block-tag {
    display: inline-block; border-radius: 6px; padding: 0.25rem 0.7rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.8rem;
}
.tag-cv  { background: rgba(255,75,75,0.15);  color: #ff4b4b; }
.tag-ml  { background: rgba(81,207,102,0.15); color: #51cf66; }
.tag-nlp { background: rgba(116,192,252,0.15);color: #74c0fc; }
.block-title { color: #fff; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; }
.block-sub   { color: #adb5bd; font-size: 0.88rem; margin-bottom: 1.2rem; }
.detail-row  { display: flex; gap: 0.5rem; align-items: flex-start; margin-bottom: 0.5rem; }
.detail-icon { font-size: 1rem; margin-top: 0.05rem; }
.detail-text { color: #ced4da; font-size: 0.88rem; }
.arrow { text-align: center; font-size: 2rem; color: #ff4b4b; padding-top: 3rem; }
.metric-pill {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.6rem 1rem; margin-top: 1rem;
    display: flex; justify-content: space-between; align-items: center;
}
.metric-label { color: #868e96; font-size: 0.8rem; }
.metric-value { color: #fff; font-weight: 600; font-size: 0.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>⚙️ How it Works</h1>
    <p>A three-stage AI pipeline combining Computer Vision, Machine Learning and Natural Language Processing.</p>
</div>""", unsafe_allow_html=True)

c1, arrow1, c2, arrow2, c3 = st.columns([4, 0.6, 4, 0.6, 4])

with c1:
    st.markdown("""
    <div class="block-card">
        <div class="block-tag tag-cv">Block 1 · Computer Vision</div>
        <div class="block-title">📸 Damage Detection</div>
        <div class="block-sub">EfficientNet-B0 fine-tuned on the VehiDE dataset</div>
        <div class="detail-row"><div class="detail-icon">📦</div>
            <div class="detail-text"><strong>Dataset:</strong> VehiDE — 13,945 images, 7 damage classes</div></div>
        <div class="detail-row"><div class="detail-icon">🧠</div>
            <div class="detail-text"><strong>Model:</strong> EfficientNet-B0 (ImageNet pretrained, fine-tuned)</div></div>
        <div class="detail-row"><div class="detail-icon">🎯</div>
            <div class="detail-text"><strong>Output:</strong> Damage class + per-class confidence scores</div></div>
        <div class="detail-row"><div class="detail-icon">🏷️</div>
            <div class="detail-text"><strong>Classes:</strong> Kratzer, Delle, Glasbruch, Lampenbruch, Riss, Loch, Fehlende Teile</div></div>
        <div class="metric-pill">
            <span class="metric-label">Val Accuracy</span>
            <span class="metric-value">~50 %</span>
        </div>
        <div class="metric-pill">
            <span class="metric-label">Parameters</span>
            <span class="metric-value">5.3 M</span>
        </div>
    </div>""", unsafe_allow_html=True)

with arrow1:
    st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="block-card">
        <div class="block-tag tag-ml">Block 2 · Machine Learning</div>
        <div class="block-title">💰 Cost Estimation</div>
        <div class="block-sub">XGBoost trained on car insurance claim data</div>
        <div class="detail-row"><div class="detail-icon">📦</div>
            <div class="detail-text"><strong>Dataset:</strong> Car Insurance Claims — 10,302 policies</div></div>
        <div class="detail-row"><div class="detail-icon">🧠</div>
            <div class="detail-text"><strong>Model:</strong> XGBoost Regressor (log-transformed target)</div></div>
        <div class="detail-row"><div class="detail-icon">🔗</div>
            <div class="detail-text"><strong>CV Integration:</strong> Damage class & confidence as features</div></div>
        <div class="detail-row"><div class="detail-icon">🎯</div>
            <div class="detail-text"><strong>Output:</strong> Estimated cost + ±20% confidence range</div></div>
        <div class="metric-pill">
            <span class="metric-label">Test MAE</span>
            <span class="metric-value">~$2,957</span>
        </div>
        <div class="metric-pill">
            <span class="metric-label">vs. Random Forest</span>
            <span class="metric-value">XGBoost wins</span>
        </div>
    </div>""", unsafe_allow_html=True)

with arrow2:
    st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="block-card">
        <div class="block-tag tag-nlp">Block 3 · NLP / RAG</div>
        <div class="block-title">📄 Insurance Report</div>
        <div class="block-sub">RAG pipeline with FAISS + GPT-4o-mini</div>
        <div class="detail-row"><div class="detail-icon">📦</div>
            <div class="detail-text"><strong>Knowledge base:</strong> NHTSA ODI vehicle complaints</div></div>
        <div class="detail-row"><div class="detail-icon">🔍</div>
            <div class="detail-text"><strong>Retrieval:</strong> FAISS cosine similarity, top-3 passages</div></div>
        <div class="detail-row"><div class="detail-icon">🧠</div>
            <div class="detail-text"><strong>Embeddings:</strong> all-MiniLM-L6-v2 (384-dim)</div></div>
        <div class="detail-row"><div class="detail-icon">✍️</div>
            <div class="detail-text"><strong>Generation:</strong> GPT-4o-mini with structured JSON output</div></div>
        <div class="metric-pill">
            <span class="metric-label">LLM</span>
            <span class="metric-value">GPT-4o-mini</span>
        </div>
        <div class="metric-pill">
            <span class="metric-label">Response format</span>
            <span class="metric-value">JSON (structured)</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
### 🔄 Data Flow

1. **User** uploads a damage photo and enters vehicle information
2. **CV Model** analyzes the image → outputs damage class + confidence
3. **ML Model** combines CV output + vehicle data → predicts repair cost
4. **RAG System** retrieves relevant complaint cases from NHTSA database
5. **GPT-4o-mini** uses all context to generate a structured insurance claim report
""")
