import streamlit as st
import re
import numpy as np
import pandas as pd
import base64
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="AI-based RNA Modification Prediction", layout="centered")

# -----------------------------
# OPTIONAL BACKGROUND IMAGE
# -----------------------------
# Keep your background if file exists
bg_path = Path("rna.jpg")
if bg_path.exists():
    with open(bg_path, "rb") as f:
        img_bytes = f.read()
    b64_img = base64.b64encode(img_bytes).decode()

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🧬 AI-based RNA Modification Prediction Web App")
st.caption("Multi-modification predictor: m6A, m5C, m1A, Ψ + Novel detection")

# -----------------------------
# UTILITIES
# -----------------------------

def clean_sequence(seq: str) -> str:
    """Clean and normalize RNA sequence."""
    seq = seq.upper().replace("T", "U")
    seq = re.sub(r"[^AUGC]", "", seq)  # keep only AUGC
    return seq

def parse_fasta(text: str) -> str:
    """Extract sequence from FASTA content."""
    lines = text.strip().splitlines()
    seq_lines = [ln.strip() for ln in lines if not ln.startswith(">")]
    return clean_sequence("".join(seq_lines))

def simulate_probability(min_p=0.70, max_p=0.98):
    """Temporary probability generator (until real DL model added)."""
    return float(np.round(np.random.uniform(min_p, max_p), 3))

# -----------------------------
# MOTIF-BASED PREDICTORS (PLACEHOLDER)
# Later replace with trained CNN model predictions
# -----------------------------
def predict_m6A_sites(seq):
    # Common motif: DRACH -> simplified regex example: G[AG]AC
    pattern = r'(G[AG]AC)'
    return [(m.start(), m.group()) for m in re.finditer(pattern, seq)]

def predict_m5C_sites(seq):
    # Example motif pattern for demo
    pattern = r'(C[GA]GG)'
    return [(m.start(), m.group()) for m in re.finditer(pattern, seq)]

def predict_m1A_sites(seq):
    # Example motif for demo
    pattern = r'(A[UG]A)'
    return [(m.start(), m.group()) for m in re.finditer(pattern, seq)]

def predict_psi_sites(seq):
    # Example motif for demo
    pattern = r'(UGA)'
    return [(m.start(), m.group()) for m in re.finditer(pattern, seq)]

# -----------------------------
# NOVEL DETECTION (IsolationForest)
# -----------------------------
def ai_discover_novel_sites(seq):
    seq = clean_sequence(seq)
    if len(seq) < 5000:
        return 1, 0.0  # not novel

    features = np.array([[seq.count('A')/len(seq),
                          seq.count('U')/len(seq),
                          seq.count('G')/len(seq),
                          seq.count('C')/len(seq)]])

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "novel_model.joblib"

    if model_path.exists():
        model = joblib.load(model_path)
    else:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(features)
        joblib.dump(model, model_path)

    prediction = model.predict(features)[0]  # -1 anomaly, 1 normal
    score = float(model.decision_function(features)[0])
    return prediction, score

# -----------------------------
# INPUT AREA
# -----------------------------
st.subheader("1) Input RNA Sequence")

seq = st.text_area(
    "✏️ Enter or Paste RNA Sequence:",
    height=160,
    placeholder="Paste RNA sequence here (A, U, G, C only)"
)

uploaded_file = st.file_uploader(
    "📂 Or upload FASTA file",
    type=["fasta", "fa", "txt"]
)

if uploaded_file is not None:
    uploaded_seq = uploaded_file.read().decode("utf-8")
    st.session_state.seq = parse_fasta(uploaded_seq)
    st.success("✅ FASTA file loaded successfully!")

# Use session state if exists
if "seq" in st.session_state:
    seq = st.session_state.seq

seq_clean = clean_sequence(seq)

# -----------------------------
# OPTIONS
# -----------------------------
st.subheader("2) Select Modifications")
col1, col2 = st.columns(2)

with col1:
    use_m6a = st.checkbox("m6A", value=True)
    use_m5c = st.checkbox("m5C", value=True)

with col2:
    use_m1a = st.checkbox("m1A", value=True)
    use_psi = st.checkbox("Ψ (Pseudouridine)", value=True)

use_novel = st.checkbox("AI Novel Detection (Optional)", value=False)

threshold = st.slider(
    "3) Threshold (Probability cutoff)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
st.subheader("4) Run Prediction")

run_prediction = st.button("🔍 Predict") or uploaded_file is not None

if run_prediction:
    if not seq_clean:
        st.warning("⚠️ Please paste or upload an RNA sequence first.")
        st.stop()

    st.info(f"Sequence length: {len(seq_clean)} bases")

    results = []
    highlight_map = {}  # position -> (type, prob)

    # -----------------------------
    # m6A
    # -----------------------------
    if use_m6a:
        sites = predict_m6A_sites(seq_clean)
        for pos, motif in sites:
            # site base position (center of motif)
            center = pos + len(motif)//2
            prob = simulate_probability(0.75, 0.98)
            if prob >= threshold:
                results.append({
                    "Position": center + 1,
                    "Base": seq_clean[center],
                    "Modification": "m6A",
                    "Probability": prob,
                    "Motif": motif
                })
                highlight_map[center] = ("m6A", prob)

    # -----------------------------
    # m5C
    # -----------------------------
    if use_m5c:
        sites = predict_m5C_sites(seq_clean)
        for pos, motif in sites:
            center = pos + len(motif)//2
            prob = simulate_probability(0.70, 0.96)
            if prob >= threshold:
                results.append({
                    "Position": center + 1,
                    "Base": seq_clean[center],
                    "Modification": "m5C",
                    "Probability": prob,
                    "Motif": motif
                })
                highlight_map[center] = ("m5C", prob)

    # -----------------------------
    # m1A
    # -----------------------------
    if use_m1a:
        sites = predict_m1A_sites(seq_clean)
        for pos, motif in sites:
            center = pos + len(motif)//2
            prob = simulate_probability(0.68, 0.94)
            if prob >= threshold:
                results.append({
                    "Position": center + 1,
                    "Base": seq_clean[center],
                    "Modification": "m1A",
                    "Probability": prob,
                    "Motif": motif
                })
                highlight_map[center] = ("m1A", prob)

    # -----------------------------
    # Ψ (Pseudouridine)
    # -----------------------------
    if use_psi:
        sites = predict_psi_sites(seq_clean)
        for pos, motif in sites:
            center = pos + len(motif)//2
            prob = simulate_probability(0.65, 0.92)
            if prob >= threshold:
                results.append({
                    "Position": center + 1,
                    "Base": seq_clean[center],
                    "Modification": "Ψ",
                    "Probability": prob,
                    "Motif": motif
                })
                highlight_map[center] = ("Ψ", prob)

    # -----------------------------
    # NOVEL DETECTION
    # -----------------------------
    if use_novel:
        pred, score = ai_discover_novel_sites(seq_clean)
        st.subheader("🤖 AI Novel Detection Result")
        if pred == -1:
            st.success("🧠 Potential NOVEL modification-like pattern detected!")
            st.write(f"Novelty Score: **{abs(score):.3f}**")
        else:
            st.info("No novel patterns detected.")

    # -----------------------------
    # RESULTS TABLE
    # -----------------------------
    st.subheader("📌 Prediction Results")

    if len(results) > 0:
        df = pd.DataFrame(results)
        df = df.sort_values(by=["Position", "Modification"]).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

        # -----------------------------
        # HIGHLIGHTED SEQUENCE VIEW
        # -----------------------------
        st.subheader("🧬 Highlighted Sequence")

        # Color map
        color_map = {
            "m6A": "#FFD54F",  # yellow
            "m5C": "#81C784",  # green
            "m1A": "#FFB74D",  # orange
            "Ψ":   "#64B5F6"   # blue
        }

        # Build highlighted HTML
        html_seq = ""
        for i, base in enumerate(seq_clean):
            if i in highlight_map:
                mod, prob = highlight_map[i]
                color = color_map.get(mod, "#FFFFFF")
                html_seq += f"<span title='{mod} | P={prob}' style='background-color:{color}; padding:2px; border-radius:4px; font-weight:700;'>{base}</span>"
            else:
                html_seq += f"<span style='padding:2px;'>{base}</span>"

            # line breaks for readability
            if (i + 1) % 60 == 0:
                html_seq += "<br>"

        st.markdown(
            f"<div style='font-family:monospace; font-size:18px; line-height:1.8;'>{html_seq}</div>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # DOWNLOAD CSV
        # -----------------------------
        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="rna_modification_results.csv",
            mime="text/csv"
        )

    else:
        st.warning("No motifs found — try lowering threshold or different sequence")

    # STILL SHOW SEQUENCE (important UX fix)
    st.subheader("🧬 Sequence")
    st.text(seq_clean)
st.markdown("---")
st.caption("Developed by Laiba Riaz — Multi RNA Modification Prediction Tool 💡")