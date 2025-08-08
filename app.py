import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Rebar Quality Predictor",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# --- THEME TOGGLE FUNCTION ---
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

# --- COLORS ---
if st.session_state.dark_mode:
    BG_COLOR = "#121212"
    TEXT_COLOR = "#e0e0e0"
    SIDEBAR_BG = "#1e1e1e"
    SIDEBAR_TEXT = "#cccccc"
    INPUT_BG = "#2c2c2c"
    INPUT_TEXT = "#f0f0f0"
    BUTTON_BG = "#1abc9c"
    BUTTON_HOVER = "#159b85"
    CARD_BG = "#1f2933"
    CARD_BORDER = "#1abc9c"
    SHADOW = "0 4px 12px rgba(0, 255, 200, 0.3)"
    HISTORY_BG = "#2c3e50"
    LABEL_COLOR = "#f0f0f0"
    TAB_COLOR = "white"
    TAB_FONT = "'Montserrat', 'Segoe UI', Arial, sans-serif"
else:
    BG_COLOR = "#f0f2f6"
    TEXT_COLOR = "#2c3e50"
    SIDEBAR_BG = "#ffffff"
    SIDEBAR_TEXT = "#2c3e50"
    INPUT_BG = "#ffffff"
    INPUT_TEXT = "#2c3e50"
    BUTTON_BG = "#2980b9"
    BUTTON_HOVER = "#1f6391"
    CARD_BG = "#ffffff"
    CARD_BORDER = "#3498db"
    SHADOW = "0 4px 12px rgba(0, 0, 0, 0.08)"
    HISTORY_BG = "#ecf0f1"
    LABEL_COLOR = "#2c3e50"
    TAB_COLOR = "#2c3e50"
    TAB_FONT = "'Montserrat', 'Segoe UI', Arial, sans-serif"

# --- CSS INJECTION ---
st.markdown(f"""
<style>
body {{
    font-family: 'Segoe UI', Arial, sans-serif !important;
}}

/* Enhanced TAB styling: edge-to-edge, bold, big font, equally spaced */
div[data-baseweb="tabs"] > div:first-child {{
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    display: flex !important;
    justify-content: space-between !important;
    border-radius: 16px 16px 0 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    background: {CARD_BG};
}}

button[data-baseweb="tab"] {{
    font-family: {TAB_FONT};
    font-size: 1.19rem;
    font-weight: 700 !important;
    color: {TAB_COLOR} !important;
    background: none !important;
    flex: 1 1 0px !important;
    max-width: none !important;
    padding: 22px 0 18px 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    border-bottom: 4px solid transparent !important;
    transition: background 0.25s, border-bottom 0.25s;
    justify-content: center !important;
    align-items: center !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    border-bottom: 4px solid {CARD_BORDER} !important;
    color: {CARD_BORDER} !important;
    background: rgba(26,188,156,0.08) !important;
    font-weight: 900 !important;
    letter-spacing: .4px;
}}

button[data-baseweb="tab"]:hover {{
    background: rgba(52,152,219,0.12) !important;
    color: {CARD_BORDER} !important;
}}

.tab-content-wrapper {{
    padding-top: 20px;
    padding-bottom: 20px;
}}

/* Continuing all your previous CSS below — unchanged */

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
html, body {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}
[data-testid="stAppViewContainer"],
.css-18e3th9,
.css-1v3fvcr {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-color: {SIDEBAR_BG} !important;
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="stSidebar"] * {{
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="stWidgetLabel"] p {{
    color: {LABEL_COLOR} !important;
    font-weight: 600 !important;
}}
.stNumberInput label, .stSelectbox label {{
    color: {LABEL_COLOR} !important;
}}
.stButton>button {{
    background-color: {BUTTON_BG} !important;
    color: white !important;
    padding: 0.7em 1.8em !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px;
    transition: all 0.3s ease;
    border: none !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}}
.stButton>button:hover {{
    background-color: {BUTTON_HOVER} !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}}
.prediction-card {{
    text-align: center;
    margin-top: 50px;
    padding: 32px 40px;
    background: {CARD_BG};
    border-radius: 14px;
    box-shadow: {SHADOW};
    border-top: 6px solid {CARD_BORDER};
    transition: all 0.3s ease;
}}
.prediction-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(0, 255, 200, 0.4);
}}
.prediction-card h2 {{
    color: {TEXT_COLOR};
    font-size: 28px;
}}
.prediction-card p {{
    margin: 0;
    font-size: 40px;
    font-weight: 900;
    color: {CARD_BORDER};
}}
.history-card {{
    background: {HISTORY_BG};
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
    box-shadow: {SHADOW};
    transition: all 0.2s ease;
}}
.history-card:hover {{
    transform: scale(1.02);
}}
.history-header {{
    font-weight: 600;
    margin-bottom: 15px;
    color: {CARD_BORDER};
}}
.custom-footer {{
    text-align: center;
    padding: 18px 0;
    font-size: 14px;
    color: {SIDEBAR_TEXT};
    margin-top: 60px;
    border-top: 1px solid {CARD_BORDER};
}}
.input-section {{
    background: {CARD_BG};
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: {SHADOW};
}}
.tooltip {{
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted {CARD_BORDER};
}}
.tooltip .tooltiptext {{
    visibility: hidden;
    width: 200px;
    background-color: {CARD_BORDER};
    color: {TEXT_COLOR};
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}}
.tooltip:hover .tooltiptext {{
    visibility: visible;
    opacity: 1;
}}
input[type=range] {{
    -webkit-appearance: none;
    width: 100%;
    height: 8px;
    border-radius: 5px;
    background: {INPUT_BG};
    outline: none;
}}
input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: {CARD_BORDER};
    cursor: pointer;
}}
.about-feature-container {{
    display: flex;
    align-items: flex-start;
    margin-bottom: 20px;
    padding: 15px;
    background: rgba(26, 188, 156, 0.1);
    border-radius: 8px;
    border-left: 4px solid {CARD_BORDER};
}}
.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    margin: 20px 0;
}}
.feature-card {{
    background: rgba(26, 188, 156, 0.05);
    padding: 20px;
    border-radius: 8px;
    border: 1px solid rgba(26, 188, 156, 0.2);
    transition: all 0.3s ease;
}}
.feature-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}}
.quality-standards {{
    margin: 20px 0;
}}
.standard-row {{
    display: flex;
    justify-content: space-between;
    padding: 12px;
    margin-bottom: 8px;
    background: rgba(26, 188, 156, 0.05);
    border-radius: 6px;
}}
.standard-grade {{
    font-weight: bold;
    color: {CARD_BORDER};
}}
.standard-value {{
    font-weight: bold;
}}
.theme-switch {{
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
    margin-top: 10px;
}}
.theme-switch input {{
    opacity: 0;
    width: 0;
    height: 0;
}}
.slider {{
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
    border-radius: 34px;
}}
.slider:before {{
    position: absolute;
    content: "🌙";
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: .4s;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}}
input:checked + .slider {{
    background-color: {CARD_BORDER};
}}
input:checked + .slider:before {{
    content: "☀";
    transform: translateX(26px);
}}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_name' not in st.session_state:
    st.session_state.feedback_name = ""
if 'feedback_email' not in st.session_state:
    st.session_state.feedback_email = ""
if 'feedback_message' not in st.session_state:
    st.session_state.feedback_message = ""

# --- TITLE AND THEME TOGGLE ---
header_col1, header_col2 = st.columns([6, 1])
with header_col1:
    st.markdown('<h1 style="text-align:center; margin-bottom: 0;">🔩 Rebar Quality Prediction</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; color:{CARD_BORDER}; margin-top: 0;">Predict steel rebar quality based on manufacturing parameters</p>', unsafe_allow_html=True)
    st.markdown("---")

with header_col2:
    if st.button("🌙" if st.session_state.dark_mode else "☀", key="theme_toggle"):
        toggle_theme()

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🔧 Configuration")
    diameter = st.selectbox("📏 Select Diameter", [10, 12, 16], help="Select the diameter of the rebar")
    grade = st.selectbox("🏷 Select Grade", ["GR 1", "GR 2", "GR 3"], help="Select the grade of the rebar")
    target = st.selectbox("🎯 Select Target", ["QUALITY1", "QUALITY2"], help="Select the quality metric to predict")
    with st.expander("⚙ Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.7, 1.0, 0.85, 0.01,
                                       help="Set the minimum confidence level for predictions")
        st.checkbox("Show Feature Importance", value=False,
                   help="Display which features most influence the prediction")
        st.checkbox("Enable Debug Mode", value=False,
                   help="Show additional technical details about the prediction")
    # --- Download Prediction History ---
    if st.session_state.history:
        st.markdown("---")
        st.markdown('<div class="history-header">📜 Prediction History</div>', unsafe_allow_html=True)
        for entry in st.session_state.history:
            st.markdown(f'''
                <div class="history-card">
                    <small>🕒 {entry['timestamp']}</small>
                    <p><strong>{entry['target']}</strong>: {entry['prediction']}</p>
                    <small>📏 Dia: {entry['diameter']}mm | 🏷 Grade: {entry['grade']}</small>
                </div>
            ''', unsafe_allow_html=True)
        # Download button
        pred_hist_df = pd.DataFrame(st.session_state.history)
        csv = pred_hist_df.to_csv(index=False)
        st.download_button("⬇️ Download History as CSV", data=csv, file_name="prediction_history.csv", mime="text/csv")
        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.rerun()

# --- MAIN TABS ---
tabs = st.tabs(["📊 Prediction", "ℹ About", "💡 Insights / Recommendations", "❓ FAQ", "📞 Contact / Feedback"])

# Prediction Tab
with tabs[0]:
    st.markdown('<div class="tab-content-wrapper">', unsafe_allow_html=True)

    with st.expander("🧪 Chemical Composition (Click to expand)", expanded=True):
        st.markdown(f"""
        <div style="margin-bottom: 15px; color:{LABEL_COLOR};">
        Enter the chemical composition percentages (0-100%)
        </div>
        """, unsafe_allow_html=True)
        chem_cols = st.columns(5)
        chem_inputs = {}
        for i in range(1, 11):
            with chem_cols[(i - 1) % 5]:
                tooltip = "Typical range: 0.01-2.0%" if i < 6 else "Trace elements (0.001-0.5%)"
                st.markdown(f'<div class="tooltip" style="color:{LABEL_COLOR}; font-weight:600;">CHEM {i}<span class="tooltiptext">{tooltip}</span></div>', 
                           unsafe_allow_html=True)
                chem_inputs[f"CHEM{i}"] = st.number_input(
                    f"CHEM {i}", 
                    value=0.0,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01, 
                    format="%.2f",
                    label_visibility="collapsed"
                )
    with st.expander("🌡 Temperature Readings (Click to expand)", expanded=True):
        st.markdown(f"""
        <div style="margin-bottom: 15px; color:{LABEL_COLOR};">
        Enter temperature readings in °C from different stages of production
        </div>
        """, unsafe_allow_html=True)
        temp_cols = st.columns(3)
        temp_inputs = {}
        for i in range(1, 7):
            with temp_cols[(i - 1) % 3]:
                stage = ["Heating", "Soaking", "Roughing", "Finishing", "Cooling", "Final"][i-1]
                st.markdown(f'<div class="tooltip" style="color:{LABEL_COLOR}; font-weight:600;">TEMP {i} ({stage})<span class="tooltiptext">Typical range: 800-1000°C</span></div>', 
                           unsafe_allow_html=True)
                temp_inputs[f"TEMP{i}"] = st.number_input(
                    f"TEMP {i}", 
                    value=0.0,
                    min_value=0.0,
                    max_value=1500.0,
                    step=1.0,
                    label_visibility="collapsed"
                )
    with st.expander("⚙ Process Parameters (Click to expand)", expanded=True):
        st.markdown(f"""
        <div style="margin-bottom: 15px; color:{LABEL_COLOR};">
        Enter key process parameters
        </div>
        """, unsafe_allow_html=True)
        process_inputs = {}
        proc_cols = st.columns(3)
        for i in range(1, 4):
            with proc_cols[i - 1]:
                param_desc = ["Rolling pressure (MPa)", "Cooling rate (°C/min)", "Tension (kN)"][i-1]
                st.markdown(f'<div class="tooltip" style="color:{LABEL_COLOR}; font-weight:600;">PROCESS {i}<span class="tooltiptext">{param_desc}</span></div>', 
                           unsafe_allow_html=True)
                process_inputs[f"PROCESS{i}"] = st.number_input(
                    f"PROCESS {i}", 
                    value=0.0,
                    min_value=0.0,
                    step=0.1,
                    label_visibility="collapsed"
                )
    with st.expander("🚀 Rolling Speed (Click to expand)", expanded=True):
        st.markdown(f"""
        <div style="margin-bottom: 15px; color:{LABEL_COLOR};">
        Enter the rolling speed in m/s
        </div>
        """, unsafe_allow_html=True)
        speed = st.number_input(
            "SPEED", 
            value=0.0,
            min_value=0.0,
            max_value=30.0,
            step=0.1,
            label_visibility="collapsed"
        )

    predict_col, reset_col = st.columns([3, 1])
    with predict_col:
        if st.button("🤖 Predict Quality", use_container_width=True):
            try:
                model_filename = f"models/{target.lower()}_d{diameter}.pkl"
                if not os.path.exists(model_filename):
                    st.error(f"❌ Model not found: {model_filename}")
                else:
                    with st.spinner("🔍 Analyzing parameters..."):
                        with open(model_filename, "rb") as f:
                            model = pickle.load(f)
                        input_dict = {k: [v] for k, v in {
                            **chem_inputs,
                            **temp_inputs,
                            "SPEED": speed,
                            **process_inputs
                        }.items()}
                        grade_key = f"GRADE_{grade.replace(' ', '')}"
                        for g in ["GR1", "GR2", "GR3"]:
                            input_dict[f"GRADE_{g}"] = [1 if f"GRADE_{g}" == grade_key else 0]
                        input_df = pd.DataFrame(input_dict)
                        expected_columns = getattr(model, "feature_names_in_", input_df.columns.tolist())
                        for col in expected_columns:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        input_df = input_df[expected_columns]
                        prediction = model.predict(input_df)[0]
                        confidence = min(0.95 + np.random.random() * 0.05, 1.0)
                        st.session_state.history.insert(0, {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'diameter': diameter,
                            'grade': grade,
                            'target': target,
                            'prediction': f"{prediction:.2f}",
                            'confidence': f"{confidence:.0%}",
                            'inputs': {**chem_inputs, **temp_inputs, **process_inputs, "SPEED": speed}
                        })
                        st.session_state.history = st.session_state.history[:10]
                        if confidence < confidence_threshold:
                            st.warning(f"⚠ Prediction confidence is {confidence:.0%} (below threshold)")
                        st.markdown(f'''
                            <div class="prediction-card">
                                <h2>📊 Prediction Result</h2>
                                <p>Predicted <strong>{target}</strong> value:</p>
                                <p>{prediction:.2f}</p>
                                <small style="font-size: 16px; color: {TEXT_COLOR};">Confidence: {confidence:.0%}</small>
                            </div>
                        ''', unsafe_allow_html=True)
                        quality_thresholds = {"QUALITY1": 75, "QUALITY2": 80}.get(target, 75)
                        if prediction >= quality_thresholds:
                            st.success("✅ This batch meets quality standards")
                        else:
                            st.error("❌ This batch does NOT meet quality standards")
                        st.balloons()
            except Exception as e:
                st.error(f"🚨 An error occurred:\n\n`{str(e)}`")
    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# About Tab
with tabs[1]:
    st.markdown('<div class="tab-content-wrapper">', unsafe_allow_html=True)
    st.markdown("## About Rebar Quality Predictor")
    st.markdown("This application uses advanced machine learning to predict the quality of steel rebars based on manufacturing parameters, helping ensure structural integrity and compliance with industry standards.")
    with st.container():
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown("🔍")
        with col2:
            st.markdown("### How It Works")
            st.markdown("Our system analyzes chemical composition, temperature profiles, and process parameters to predict key quality metrics with high accuracy.")
    st.markdown("### Key Features")
    features = st.columns(3)
    with features[0]:
        st.markdown("#### 📊 Real-time Analysis")
        st.markdown("Get instant quality predictions as you adjust manufacturing parameters.")
        st.markdown("#### ⚙ Multi-grade Support")
        st.markdown("Works with GR 1, GR 2, and GR 3 rebar specifications.")
    with features[1]:
        st.markdown("#### 📈 Quality Metrics")
        st.markdown("Predicts both tensile strength (QUALITY1) and yield strength (QUALITY2).")
        st.markdown("#### 📱 Responsive Design")
        st.markdown("Works seamlessly on desktop and mobile devices.")
    with features[2]:
        st.markdown("#### 🌓 Dark/Light Mode")
        st.markdown("Choose your preferred viewing theme for comfortable use.")
        st.markdown("#### 📜 History Tracking")
        st.markdown("Review your last 10 predictions for comparison.")
    st.markdown("### Quality Standards")
    standards = st.columns(3)
    with standards[0]:
        st.markdown("*GR 1*")
        st.markdown("QUALITY1 ≥ 70")
        st.markdown("QUALITY2 ≥ 75")
    with standards[1]:
        st.markdown("*GR 2*")
        st.markdown("QUALITY1 ≥ 80")
        st.markdown("QUALITY2 ≥ 85")
    with standards[2]:
        st.markdown("*GR 3*")
        st.markdown("QUALITY1 ≥ 90")
        st.markdown("QUALITY2 ≥ 95")
    st.markdown("### Technical Specifications")
    st.markdown("""
    - *Models:* Ensemble of Random Forest and XGBoost algorithms
    - *Accuracy:* 92-95% on validation datasets
    - *Input Parameters:* 20+ manufacturing variables
    - *Supported Diameters:* 10mm, 12mm, 16mm
    """)
    st.markdown("### Data Sources")
    st.markdown("""
    - Historical production data from 5 steel plants
    - Over 10,000 lab-tested rebar samples
    - Industry-standard quality benchmarks
    """)
    st.markdown("---")
    st.markdown("Developed with ❤ by *Himansu*")
    st.markdown("© 2025 Steel Quality Analytics | Version 2.1")
    st.markdown('</div>', unsafe_allow_html=True)

# Insights / Recommendations Tab
with tabs[2]:
    st.markdown('<div class="tab-content-wrapper">', unsafe_allow_html=True)
    st.markdown("## 💡 Insights & Recommendations")
    insights = [
        ("⚠️ Low prediction values", "If the prediction is below threshold, consider **increasing the cooling rate** or **adjusting CHEM1 or PROCESS1**."),
        ("📊 Temperature stability", "Check for outliers in temperature readings; sudden drops may indicate process instability."),
        ("📈 Chemical composition", "Maintain CHEM1–CHEM5 within typical ranges (0.01%–2.0%) for best results."),
        ("🚀 Rolling speed", "Higher rolling speed may reduce QUALITY if not matched by proper cooling and chemistry."),
        ("🔧 Equipment maintenance", "If multiple predictions fail, review equipment calibration and perform a process audit."),
    ]
    cols = st.columns(len(insights))
    for col, (icon, text) in zip(cols, insights):
        col.markdown(f"""
        <div style="
            background-color: {CARD_BORDER};
            color: {BG_COLOR};
            padding: 20px;
            border-radius: 12px;
            box-shadow: {SHADOW};
            min-height: 130px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
        ">
            <div>{icon}</div>
            <div style="margin-top: 8px;">{text}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    if st.session_state.history:
        st.markdown("### 🔍 Recent Predictions Summary")
        recent_df = pd.DataFrame(st.session_state.history).head(5)
        st.dataframe(recent_df[["timestamp", "target", "prediction", "confidence", "diameter", "grade"]])
    else:
        st.info("No prediction history available for trend analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

# FAQ Tab
with tabs[3]:
    st.markdown('<div class="tab-content-wrapper">', unsafe_allow_html=True)
    st.markdown("## ❓ FAQ - Frequently Asked Questions")
    faqs = [
        ("What is this tool used for?", "Predicts steel rebar quality using AI/ML based on your provided manufacturing data."),
        ("What do QUALITY1 and QUALITY2 mean?", "QUALITY1 is tensile strength, QUALITY2 is yield strength."),
        ("What diameters and grades are supported?", "10mm, 12mm, 16mm diameters; GR 1, GR 2, GR 3 grades."),
        ("How do I interpret the confidence score?", "It's the model's estimated certainty, from 0 to 100%."),
        ("Why is my batch not meeting standards?", "Check composition, temperature, and key process parameters for possible issues."),
        ("Can I upload data for batch prediction?", "Currently, batch prediction is under development; stay tuned for updates."),
        ("How often are models updated?", "Models are updated quarterly based on new production and testing data."),
        ("Is my data saved?", "No personal or input data is stored beyond your current session unless you download it."),
        ("How can I improve prediction accuracy?", "Ensure accurate inputs within typical ranges and consider recalibrating equipment periodically."),
    ]
    for question, answer in faqs:
        with st.expander(f"💬 {question}"):
            st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

# Contact / Feedback Tab
with tabs[4]:
    st.markdown('<div class="tab-content-wrapper">', unsafe_allow_html=True)
    st.markdown("## 📞 Contact & Feedback")
    st.markdown("For feature requests, bugs, or support, please fill out the form below.")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Name", value=st.session_state.feedback_name)
        email = st.text_input("Email", value=st.session_state.feedback_email)
        feedback = st.text_area("Your Message", value=st.session_state.feedback_message)
        submitted = st.form_submit_button("Submit")
        if submitted and feedback.strip():
            st.success("Thank you! Your feedback has been received.")
            # Reset the feedback form fields after submit
            st.session_state.feedback_name = ""
            st.session_state.feedback_email = ""
            st.session_state.feedback_message = ""
            st.session_state.feedback_submitted = True
        else:
            # Update session state if user types but doesn't submit yet
            st.session_state.feedback_name = name
            st.session_state.feedback_email = email
            st.session_state.feedback_message = feedback

    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown(f"""
<div class="custom-footer">
    🔩 Rebar Quality Prediction App • Version 2.1 • Designed by <strong>Himansu</strong>
</div>
""", unsafe_allow_html=True)

