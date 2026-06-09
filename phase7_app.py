# ============================================================
# PHASE 7: STREAMLIT WEB APP
# AI Smart Waste Segregation Assistant (SDG 12)
# Run: streamlit run phase7_app.py
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Waste Segregation Assistant",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; color: white; }
    h1, h2, h3 { color: white !important; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        border: 1px solid #2d2d5e;
    }
    .advice-item {
        background: #0d3349;
        border-left: 4px solid #27ae60;
        padding: 8px 14px;
        border-radius: 4px;
        margin: 6px 0;
        color: #ecf0f1;
        font-size: 15px;
    }
    .impact-box {
        background: #1a2e1a;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 8px;
        color: #a8e6a3;
        font-size: 15px;
        margin: 8px 0;
    }
    .sdg-box {
        background: #1a1a3e;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 8px;
        color: #aac4e8;
        font-size: 15px;
        margin: 8px 0;
    }
    .metric-box {
        background: #1e1e3e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #3d3d7e;
    }
</style>
""", unsafe_allow_html=True)

# ─── Config ──────────────────────────────────────────────────
MODEL_PATH = "waste_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_INFO = {
    "cardboard": {
        "emoji": "📦", "color": "#3498db",
        "bin": "Blue Recycling Bin",
        "advice": [
            "Flatten boxes before recycling",
            "Remove plastic tape or staples",
            "Keep dry — wet cardboard cannot be recycled",
            "Break down large boxes to save space"
        ],
        "impact": "Recycling 1 tonne of cardboard saves 17 trees 🌳",
        "sdg": "SDG 12: Reduces deforestation & production waste"
    },
    "glass": {
        "emoji": "🍶", "color": "#27ae60",
        "bin": "Green Glass Bin",
        "advice": [
            "Rinse bottles and jars before recycling",
            "Remove metal lids — recycle separately",
            "Wrap broken glass safely before disposal",
            "Glass can be recycled infinitely"
        ],
        "impact": "Recycling glass uses 40% less energy ⚡",
        "sdg": "SDG 12: Conserves raw materials and saves energy"
    },
    "metal": {
        "emoji": "🥫", "color": "#f39c12",
        "bin": "Yellow Metal Bin",
        "advice": [
            "Rinse cans to remove food residue",
            "Crush cans to save space",
            "Both aluminium and steel are recyclable",
            "Remove paper labels if possible"
        ],
        "impact": "Recycling aluminium uses 95% less energy ⚡",
        "sdg": "SDG 12: Reduces mining impact and industrial waste"
    },
    "paper": {
        "emoji": "📄", "color": "#9b59b6",
        "bin": "Blue Recycling Bin",
        "advice": [
            "Keep paper dry and clean",
            "Remove plastic windows from envelopes",
            "Shredded paper can be composted",
            "Greasy pizza boxes → compost, not recycle"
        ],
        "impact": "Recycling 1 tonne saves 380 gallons of oil 🛢️",
        "sdg": "SDG 12: Saves forests and reduces chemical pollution"
    },
    "plastic": {
        "emoji": "🧴", "color": "#e74c3c",
        "bin": "Yellow Recycling Bin",
        "advice": [
            "Check recycling number (♻ 1–7) on the bottom",
            "Rinse containers before recycling",
            "Plastic bags → store drop-off, not curbside bin",
            "Avoid single-use plastics (SDG 12)"
        ],
        "impact": "Recycling prevents 1000s of years in landfill 🌍",
        "sdg": "SDG 12 + SDG 13: Reduces plastic pollution and CO₂"
    },
    "trash": {
        "emoji": "🗑️", "color": "#7f8c8d",
        "bin": "Black General Waste Bin",
        "advice": [
            "This item likely cannot be recycled",
            "Place in general waste bin",
            "Try to reduce use of non-recyclable items",
            "Consider reusable product alternatives"
        ],
        "impact": "Reducing general waste decreases landfill overload 🏭",
        "sdg": "SDG 12: Encourages mindful consumption habits"
    }
}

# ─── Load Model (cached) ─────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return None

# ─── Prediction ──────────────────────────────────────────────
def predict(img: Image.Image, model):
    img_resized = img.resize(IMG_SIZE)
    img_array   = np.array(img_resized.convert("RGB")) / 255.0
    img_array   = np.expand_dims(img_array, axis=0)
    preds       = model.predict(img_array, verbose=0)[0]
    idx         = np.argmax(preds)
    return CLASSES[idx], preds[idx] * 100, preds

# ─── UI ───────────────────────────────────────────────────────
st.markdown("# ♻️ AI Smart Waste Segregation Assistant")
st.markdown("**Powered by Deep Learning | Supporting SDG 12 — Responsible Consumption**")
st.markdown("---")

# Load model
model = load_model()
if model is None:
    st.error("⚠️ Model file 'waste_model.h5' not found. Please train the model first (run phase4_train_model.py).")
    st.stop()
else:
    st.success("✅ AI Model loaded and ready!")

st.markdown("### 📸 Upload a Waste Image")
st.caption("Supported: JPG, JPEG, PNG — Works best with clear photos of single items")

uploaded_file = st.file_uploader(
    "Drop your image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Your uploaded image", use_container_width=True)

    with col2:
        with st.spinner("🤖 Analysing waste type..."):
            pred_class, confidence, all_probs = predict(img, model)
            info = WASTE_INFO[pred_class]

        st.markdown(f"""
        <div class="result-card">
            <h2 style="color:{info['color']}; margin:0">{info['emoji']} {pred_class.upper()}</h2>
            <p style="color:#aaa; margin:4px 0">Confidence: <strong style="color:white">{confidence:.1f}%</strong></p>
            <p style="color:#aaa; margin:4px 0">Dispose in: <strong style="color:{info['color']}">{info['bin']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Disposal Advice ───────────────────────────────────────
    st.markdown("### 📋 Disposal Instructions")
    for tip in info["advice"]:
        st.markdown(f'<div class="advice-item">• {tip}</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f'<div class="impact-box">🌱 <strong>Environmental Impact</strong><br>{info["impact"]}</div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="sdg-box">🌐 <strong>SDG Connection</strong><br>{info["sdg"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Probability chart ─────────────────────────────────────
    st.markdown("### 📊 All Category Probabilities")

    import plotly.graph_objects as go

    colors = [info["color"] if c == pred_class else "#3d3d5e" for c in CLASSES]

    fig = go.Figure(go.Bar(
        x=CLASSES,
        y=all_probs * 100,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in all_probs],
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        yaxis=dict(title="Confidence (%)", range=[0, 110], gridcolor="#333"),
        xaxis=dict(title="Waste Category"),
        showlegend=False,
        height=350,
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Responsible AI Notice ─────────────────────────────────
    st.markdown("---")
    st.info("""
    ⚖️ **Responsible AI Notice**
    This AI is for awareness and guidance only — not enforcement. Predictions may occasionally be incorrect for
    unclear or unusual images. Always use your judgment and local recycling guidelines.
    No personal data is collected or stored.
    """)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding:40px; color:#888;">
        <p style="font-size:60px">♻️</p>
        <p style="font-size:18px">Upload a photo of your waste item above</p>
        <p>The AI will identify: Cardboard · Glass · Metal · Paper · Plastic · General Trash</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555;'>Built for 1M1B Internship | SDG 12 — Responsible Consumption & Production</p>",
    unsafe_allow_html=True
)
