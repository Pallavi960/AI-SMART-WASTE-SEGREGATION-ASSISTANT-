# ============================================================
# PHASE 7 (v2): ADVANCED STREAMLIT DASHBOARD
# AI Smart Waste Segregation Assistant (SDG 12)
# Run: streamlit run phase7_app.py
#
# NEW FEATURES vs v1:
#   - Multi-tab dashboard (Classify / Batch / Analytics / About)
#   - Session history with running stats
#   - Batch image upload + bulk results table
#   - Environmental impact calculator (cumulative)
#   - Test-Time Augmentation toggle
#   - Model confidence gauge
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Waste Segregation Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: white; }
    h1, h2, h3, h4 { color: white !important; }
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px; padding: 24px; margin: 10px 0;
        border: 1px solid #2d2d5e;
    }
    .advice-item {
        background: #0d3349; border-left: 4px solid #27ae60;
        padding: 8px 14px; border-radius: 4px; margin: 6px 0;
        color: #ecf0f1; font-size: 15px;
    }
    .impact-box {
        background: #1a2e1a; border-left: 4px solid #2ecc71;
        padding: 12px 16px; border-radius: 8px; color: #a8e6a3;
        font-size: 15px; margin: 8px 0;
    }
    .sdg-box {
        background: #1a1a3e; border-left: 4px solid #3498db;
        padding: 12px 16px; border-radius: 8px; color: #aac4e8;
        font-size: 15px; margin: 8px 0;
    }
    .metric-box {
        background: #1e1e3e; border-radius: 12px; padding: 18px;
        text-align: center; border: 1px solid #3d3d7e;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 8px 8px 0 0;
        padding: 10px 20px; color: #aaa;
    }
    .stTabs [aria-selected="true"] { background-color: #27ae60; color: white; }
</style>
""", unsafe_allow_html=True)

# ─── Config ──────────────────────────────────────────────────
MODEL_PATH = "waste_model.h5"
IMG_SIZE   = (224, 224)
CLASSES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_INFO = {
    "cardboard": {"emoji": "📦", "color": "#3498db", "bin": "Blue Recycling Bin",
        "advice": ["Flatten boxes before recycling", "Remove plastic tape or staples",
                   "Keep dry — wet cardboard cannot be recycled", "Break down large boxes to save space"],
        "impact": "Recycling 1 tonne of cardboard saves 17 trees 🌳",
        "sdg": "SDG 12: Reduces deforestation & production waste",
        "co2_saved_kg": 3.5},
    "glass": {"emoji": "🍶", "color": "#27ae60", "bin": "Green Glass Bin",
        "advice": ["Rinse bottles and jars before recycling", "Remove metal lids — recycle separately",
                   "Wrap broken glass safely before disposal", "Glass can be recycled infinitely"],
        "impact": "Recycling glass uses 40% less energy ⚡",
        "sdg": "SDG 12: Conserves raw materials and saves energy",
        "co2_saved_kg": 0.3},
    "metal": {"emoji": "🥫", "color": "#f39c12", "bin": "Yellow Metal Bin",
        "advice": ["Rinse cans to remove food residue", "Crush cans to save space",
                   "Both aluminium and steel are recyclable", "Remove paper labels if possible"],
        "impact": "Recycling aluminium uses 95% less energy ⚡",
        "sdg": "SDG 12: Reduces mining impact and industrial waste",
        "co2_saved_kg": 1.8},
    "paper": {"emoji": "📄", "color": "#9b59b6", "bin": "Blue Recycling Bin",
        "advice": ["Keep paper dry and clean", "Remove plastic windows from envelopes",
                   "Shredded paper can be composted", "Greasy pizza boxes → compost, not recycle"],
        "impact": "Recycling 1 tonne saves 380 gallons of oil 🛢️",
        "sdg": "SDG 12: Saves forests and reduces chemical pollution",
        "co2_saved_kg": 1.1},
    "plastic": {"emoji": "🧴", "color": "#e74c3c", "bin": "Yellow Recycling Bin",
        "advice": ["Check recycling number (♻ 1–7) on the bottom", "Rinse containers before recycling",
                   "Plastic bags → store drop-off, not curbside bin", "Avoid single-use plastics (SDG 12)"],
        "impact": "Recycling prevents 1000s of years in landfill 🌍",
        "sdg": "SDG 12 + SDG 13: Reduces plastic pollution and CO₂",
        "co2_saved_kg": 2.5},
    "trash": {"emoji": "🗑️", "color": "#7f8c8d", "bin": "Black General Waste Bin",
        "advice": ["This item likely cannot be recycled", "Place in general waste bin",
                   "Try to reduce use of non-recyclable items", "Consider reusable product alternatives"],
        "impact": "Reducing general waste decreases landfill overload 🏭",
        "sdg": "SDG 12: Encourages mindful consumption habits",
        "co2_saved_kg": 0.0},
}

# ─── Session State Init ──────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {time, class, confidence}

# ─── Load Model (cached) ─────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        return None

# ─── Prediction (with optional TTA) ──────────────────────────
def predict(img: Image.Image, model, use_tta=True):
    if not use_tta:
        img_resized = img.resize(IMG_SIZE)
        img_array   = np.array(img_resized.convert("RGB")) / 255.0
        img_array   = np.expand_dims(img_array, axis=0)
        preds       = model.predict(img_array, verbose=0)[0]
        idx         = np.argmax(preds)
        return CLASSES[idx], preds[idx] * 100, preds

    # TTA: average over multiple augmented views
    versions = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(img).enhance(1.15),
        ImageEnhance.Brightness(img).enhance(0.85),
    ]
    all_preds = []
    for v in versions:
        v_resized = v.resize(IMG_SIZE)
        v_array   = np.array(v_resized.convert("RGB")) / 255.0
        v_array   = np.expand_dims(v_array, axis=0)
        all_preds.append(model.predict(v_array, verbose=0)[0])
    preds = np.mean(all_preds, axis=0)
    idx   = np.argmax(preds)
    return CLASSES[idx], preds[idx] * 100, preds


def log_prediction(pred_class, confidence):
    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "category": pred_class,
        "confidence": round(confidence, 1),
        "co2_saved": WASTE_INFO[pred_class]["co2_saved_kg"]
    })


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ♻️ Dashboard Settings")
    use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                         help="Averages predictions over multiple image variants for more robust results")
    st.markdown("---")
    st.markdown("### 📊 Session Stats")

    model = load_model()
    if model is None:
        st.error("Model not found")
    else:
        st.success("Model: Loaded ✅")

    total_scans = len(st.session_state.history)
    total_co2 = sum(h["co2_saved"] for h in st.session_state.history)

    st.metric("Total Scans", total_scans)
    st.metric("Estimated CO₂ Saved", f"{total_co2:.1f} kg")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption("Built for 1M1B Internship | SDG 12")


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("# ♻️ AI Smart Waste Segregation Dashboard")
st.markdown("**Powered by Deep Learning (CNN + Transfer Learning) | Supporting SDG 12 — Responsible Consumption**")

if model is None:
    st.error("⚠️ Model file 'waste_model.h5' not found. Please train the model first (phase4_train_model.py).")
    st.stop()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📚 Batch Upload", "📈 Analytics", "ℹ️ About"])


# ═══════════════════════════════════════════════════════════
# TAB 1 — SINGLE IMAGE CLASSIFY
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📸 Upload a Waste Image")
    uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed", key="single")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Your uploaded image", use_container_width=True)

        with col2:
            with st.spinner("🤖 Analysing waste type..."):
                pred_class, confidence, all_probs = predict(img, model, use_tta=use_tta)
                info = WASTE_INFO[pred_class]
                log_prediction(pred_class, confidence)

            st.markdown(f"""
            <div class="result-card">
                <h2 style="color:{info['color']}; margin:0">{info['emoji']} {pred_class.upper()}</h2>
                <p style="color:#aaa; margin:4px 0">Confidence: <strong style="color:white">{confidence:.1f}%</strong></p>
                <p style="color:#aaa; margin:4px 0">Dispose in: <strong style="color:{info['color']}">{info['bin']}</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": "Model Confidence", "font": {"color": "white"}},
                number={"suffix": "%", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": info["color"]},
                    "steps": [
                        {"range": [0, 50], "color": "#3d1a1a"},
                        {"range": [50, 75], "color": "#3d3d1a"},
                        {"range": [75, 100], "color": "#1a3d1a"},
                    ],
                }
            ))
            gauge_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(t=40, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Disposal Instructions")
        for tip in info["advice"]:
            st.markdown(f'<div class="advice-item">• {tip}</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f'<div class="impact-box">🌱 <strong>Environmental Impact</strong><br>{info["impact"]}</div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="sdg-box">🌐 <strong>SDG Connection</strong><br>{info["sdg"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 All Category Probabilities")
        colors = [info["color"] if c == pred_class else "#3d3d5e" for c in CLASSES]
        fig = go.Figure(go.Bar(
            x=CLASSES, y=all_probs * 100, marker_color=colors,
            text=[f"{p*100:.1f}%" for p in all_probs], textposition="outside"
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
            yaxis=dict(title="Confidence (%)", range=[0, 110], gridcolor="#333"),
            xaxis=dict(title="Waste Category"), showlegend=False, height=350, margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        ⚖️ **Responsible AI Notice**
        This AI is for awareness and guidance only — not enforcement. Predictions may occasionally be incorrect for
        unclear or unusual images. Always use your judgment and local recycling guidelines. No personal data is stored.
        """)
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#888;">
            <p style="font-size:60px">♻️</p>
            <p style="font-size:18px">Upload a photo of your waste item above</p>
            <p>Identifies: Cardboard · Glass · Metal · Paper · Plastic · General Trash</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📚 Classify Multiple Images at Once")
    batch_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True, key="batch")

    if batch_files:
        results = []
        progress = st.progress(0, text="Processing images...")

        for i, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            pred_class, confidence, _ = predict(img, model, use_tta=use_tta)
            log_prediction(pred_class, confidence)
            results.append({
                "Image": f.name,
                "Category": pred_class.capitalize(),
                "Confidence": f"{confidence:.1f}%",
                "Bin": WASTE_INFO[pred_class]["bin"]
            })
            progress.progress((i + 1) / len(batch_files), text=f"Processed {i+1}/{len(batch_files)}")

        progress.empty()
        st.success(f"✅ Classified {len(batch_files)} images!")

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Summary chart
        category_counts = df["Category"].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                     title="Batch Classification Summary",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "waste_classification_results.csv", "text/csv")
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#888;">
            <p style="font-size:50px">📚</p>
            <p>Upload multiple images to classify them all at once</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Session Analytics")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet this session. Classify some images first!")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><h3>{len(hist_df)}</h3><p>Total Scans</p></div>', unsafe_allow_html=True)
        with col2:
            avg_conf = hist_df["confidence"].mean()
            st.markdown(f'<div class="metric-box"><h3>{avg_conf:.1f}%</h3><p>Avg Confidence</p></div>', unsafe_allow_html=True)
        with col3:
            total_co2 = hist_df["co2_saved"].sum()
            st.markdown(f'<div class="metric-box"><h3>{total_co2:.1f} kg</h3><p>Estimated CO₂ Saved</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        col4, col5 = st.columns(2)
        with col4:
            cat_counts = hist_df["category"].value_counts()
            fig1 = px.bar(x=cat_counts.index, y=cat_counts.values,
                          color=cat_counts.index,
                          color_discrete_map={c: WASTE_INFO[c]["color"] for c in CLASSES},
                          labels={"x": "Category", "y": "Count"},
                          title="Predictions by Category")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col5:
            fig2 = px.line(hist_df, y="confidence", title="Confidence Over Time",
                           markers=True)
            fig2.update_traces(line_color="#2ecc71")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", xaxis_title="Scan #", yaxis_title="Confidence (%)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🕒 Recent History")
        st.dataframe(hist_df[["time", "category", "confidence"]].iloc[::-1],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    **AI Smart Waste Segregation Assistant** uses a Convolutional Neural Network (CNN)
    built with transfer learning (MobileNetV2) to classify waste images into six categories:
    Cardboard, Glass, Metal, Paper, Plastic, and Trash.

    **Model details:**
    - Architecture: MobileNetV2 backbone + custom classification head
    - Regularization: L2 weight decay, dropout, label smoothing, class weighting
    - Dataset: TrashNet (2,527 images)
    - Inference: Optional Test-Time Augmentation for improved robustness

    **SDG Alignment:**
    - 🎯 SDG 12 — Responsible Consumption and Production
    - 🎯 SDG 13 — Climate Action (secondary)

    **Responsible AI:**
    - This tool is for awareness and guidance, not enforcement
    - Predictions may be wrong for unclear or unusual images
    - No personal data is collected or stored
    - Dataset may carry geographic/cultural bias

    ---
    Built for the **1M1B AI for Sustainability Virtual Internship**
    in collaboration with **IBM SkillsBuild & AICTE**.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#555;'>Built for 1M1B Internship | SDG 12 — Responsible Consumption & Production</p>", unsafe_allow_html=True)