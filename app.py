

import streamlit as st
import plotly.graph_objects as go
import re
from groq import Groq
from dotenv import load_dotenv
import os
from utils.pdf_processor import extract_text_from_pdfs, split_text_into_chunks
from utils.rag_pipeline import create_vector_store, search_vector_store
from agents.report_agent import run_report_agent
from agents.severity_agent import run_severity_agent
from agents.referral_agent import run_referral_agent

load_dotenv()

st.set_page_config(
    page_title="MedReport AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
html, body, * { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }

/* === FONT HIERARCHY === */
.panel-hdr      { font-size: 0.95rem !important; font-family: 'DM Sans', sans-serif !important; letter-spacing: 0px !important; }
.sb-logo-name   { font-size: 1rem    !important; font-family: 'Syne', sans-serif !important; }
.stat-num       { font-size: 1.4rem  !important; font-weight: 700 !important; line-height: 1 !important; }
.stat-lbl       { font-size: 0.60rem !important; }
.bubble-q,
.bubble-a       { font-size: 0.81rem !important; line-height: 1.65 !important; }
.info-key       { font-size: 0.69rem !important; }
.info-val       { font-size: 0.69rem !important; }
.sb-section     { font-size: 0.62rem !important; }
.model-name     { font-size: 0.76rem !important; }
.mode-indicator { font-size: 0.69rem !important; }

/* === LAYOUT === */
.block-container { padding: 1.3rem 1.8rem 2.5rem 1.8rem !important; }

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid #131f35 !important;
    min-width: 282px !important;
    max-width: 282px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.1rem 1rem 2rem 1rem !important;
}
[data-testid="collapsedControl"] {
    background: #080f1e !important;
    border: 1px solid #131f35 !important;
    border-left: none !important;
    border-radius: 0 6px 6px 0 !important;
}

/* === SIDEBAR LOGO === */
.sb-logo {
    display: flex; align-items: center; gap: 11px;
    padding-bottom: 1rem; border-bottom: 1px solid #162035;
    margin-bottom: 0.2rem;
}
.sb-logo-icon {
    width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #1a4a8a 0%, #0e2540 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 14px rgba(26,74,138,0.45);
}
.sb-logo-name { color: #e8f0fa; font-weight: 700; }
.sb-logo-sub  { font-size: 0.62rem; color: #3a6a9a; margin-top: 2px; }

/* === SIDEBAR SECTION LABELS === */
.sb-section {
    font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
    color: #2a5a8a; margin: 1.1rem 0 0.5rem 0;
    padding-top: 1rem; border-top: 1px solid #131f35;
}
.sb-section:first-of-type { border-top: none; padding-top: 0; margin-top: 0.6rem; }

/* === SIDEBAR CARDS === */
.sb-card {
    background: #0c1e35; border: 1px solid #172840;
    border-radius: 10px; padding: 0.75rem 0.9rem;
    margin-bottom: 0.55rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
}

/* === MODEL BADGE === */
.model-badge {
    display: flex; align-items: center; gap: 9px;
    background: #091828; border: 1px solid #1a4070;
    border-radius: 8px; padding: 8px 11px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
}
.model-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: #38c97a;
    box-shadow: 0 0 7px rgba(56,201,122,0.65);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.model-name { font-weight: 700; color: #38c97a; }
.model-sub  { font-size: 0.60rem; color: #3a6a9a; margin-top: 2px; }

/* === INFO ROWS === */
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid rgba(20,40,65,0.7);
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #3a6a9a; }
.info-val { font-weight: 600; }
.green { color: #38c97a !important; }
.blue  { color: #4a9fd4 !important; }
.muted { color: #7a9ab8 !important; }

/* === STATUS BADGES === */
.status-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px; border-radius: 12px;
    font-size: 0.63rem; font-weight: 700;
}
.status-ready { background: rgba(56,201,122,0.1); color: #38c97a; border: 1px solid rgba(56,201,122,0.2); }
.status-wait  { background: rgba(74,159,212,0.1); color: #4a9fd4; border: 1px solid rgba(74,159,212,0.2); }

/* === TOKEN BOXES === */
.tok-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }
.tok-box {
    background: #091828; border: 1px solid #172840;
    border-radius: 7px; padding: 8px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.tok-num { font-size: 1.05rem; font-weight: 700; color: #e8f0fa; line-height: 1; }
.tok-lbl { font-size: 0.58rem; color: #3a6a9a; margin-top: 3px; display: block; }
.tok-total {
    background: #091828; border: 1px solid #172840; border-radius: 7px;
    padding: 6px 10px; margin-top: 6px;
    display: flex; justify-content: space-between; align-items: center;
}
.tok-total-lbl { font-size: 0.62rem; color: #3a6a9a; }
.tok-total-num { font-size: 0.92rem; font-weight: 700; color: #4a9fd4; }

/* === DISCLAIMER === */
.disclaimer {
    background: rgba(246,201,14,0.04); border: 1px solid rgba(246,201,14,0.12);
    border-left: 3px solid rgba(246,201,14,0.35); border-radius: 7px;
    padding: 8px 10px; margin-top: 1rem;
    font-size: 0.63rem; color: #7a6a30; line-height: 1.6;
}

/* === MAIN PANELS === */
.panel {
    background: #0b1928; border: 1px solid #162035;
    border-radius: 12px; padding: 1.1rem 1.15rem 1.4rem 1.15rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
}
.panel-hdr {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.2rem; font-weight: 800;
    font-family: 'DM Sans', sans-serif;
    text-transform: none; letter-spacing: 0px;
    color: #ffffff;
    background: linear-gradient(90deg, rgba(74,159,212,0.12), transparent);
    border-left: 4px solid #4a9fd4;
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    border-bottom: 1px solid #162035;
    margin-bottom: 0.9rem;
}

/* === MODE INDICATOR BANNER === */
.mode-indicator {
    padding: 6px 10px; border-radius: 7px; margin-bottom: 0.9rem;
    font-weight: 500;
}
.mode-report  { background: rgba(74,159,212,0.07); color: #4a9fd4; border: 1px solid rgba(74,159,212,0.15); }
.mode-general { background: rgba(56,201,122,0.07); color: #38c97a; border: 1px solid rgba(56,201,122,0.15); }

/* === CHAT AREA === */
.chat-wrap {
    max-height: 320px; overflow-y: auto;
    padding-right: 5px; margin-bottom: 1rem;
}
.chat-wrap::-webkit-scrollbar { width: 3px; }
.chat-wrap::-webkit-scrollbar-thumb { background: #162035; border-radius: 2px; }

.chat-empty {
    color: #1e3a5a; font-style: italic;
    padding: 1.5rem 0; text-align: center;
    font-size: 0.78rem;
}

.chat-lbl-q { font-size: 0.58rem; font-weight: 700; color: #4a9fd4; letter-spacing: 1px; text-align: right; margin-bottom: 3px; }
.chat-lbl-a { font-size: 0.58rem; font-weight: 700; color: #38c97a; letter-spacing: 1px; margin-bottom: 3px; }

.bubble-q {
    background: linear-gradient(135deg, #0f2d52, #112a4a);
    border-radius: 12px 12px 3px 12px;
    padding: 9px 13px; margin: 0 0 8px 28px;
    color: #c8daea;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.bubble-a {
    background: #091525; border: 1px solid #162035;
    border-radius: 12px 12px 12px 3px;
    padding: 9px 13px; margin: 0 28px 10px 0;
    color: #c8daea;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.bubble-a.general { border-left: 3px solid #38c97a; }

.mode-tag {
    display: inline-block; font-size: 0.54rem; font-weight: 700;
    letter-spacing: 0.8px; padding: 2px 6px;
    border-radius: 4px; margin-bottom: 3px;
}
.tag-report  { background: rgba(74,159,212,0.15); color: #4a9fd4; }
.tag-general { background: rgba(56,201,122,0.15); color: #38c97a; }

/* === INPUT BOX === */
.stTextInput > div > div > input {
    background: #091525 !important; border: 1px solid #162035 !important;
    border-radius: 9px !important; color: #e8f0fa !important;
    font-size: 0.84rem !important; padding: 0.55rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4a9fd4 !important;
    box-shadow: 0 0 0 3px rgba(74,159,212,0.10) !important;
}
.stTextInput > div > div > input::placeholder { color: #2a4a6a !important; }

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, #1a4a8a 0%, #155daa 100%) !important;
    color: #e8f0fa !important; border: none !important;
    border-radius: 9px !important; font-size: 0.83rem !important;
    font-weight: 600 !important; padding: 0.52rem 1rem !important;
    width: 100% !important; letter-spacing: 0.2px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(26,74,138,0.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e5aaa 0%, #1a6ecc 100%) !important;
    box-shadow: 0 6px 20px rgba(26,74,138,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(26,74,138,0.3) !important;
}

/* === FILE UPLOADER === */
.stFileUploader > div {
    background: #091525 !important;
    border: 1px dashed #162035 !important;
    border-radius: 10px !important;
}
.stFileUploader small { font-size: 0.58rem !important; color: #2a4a6a !important; }

[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileData"] {
    display: none !important;
}

/* === REPORT FILE BUTTONS === */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(74,159,212,0.06) !important;
    border: 1px solid rgba(74,159,212,0.12) !important;
    color: #4a9fd4 !important;
    font-size: 0.68rem !important;
    padding: 0.25rem 0.6rem !important;
    box-shadow: none !important;
    min-height: unset !important;
    width: auto !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    color: #fc8181 !important;
    background: rgba(252,129,129,0.07) !important;
    border-color: rgba(252,129,129,0.2) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* === STAT BOXES === */
.stat-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 7px; margin-bottom: 0.8rem; }
.stat-box {
    background: #091525; border: 1px solid #162035;
    border-radius: 9px; padding: 10px; text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.3);
}

/* === RAW DATA TABLE === */
.raw-tbl {
    width: 100%; border-collapse: collapse;
    table-layout: fixed;
}
.raw-tbl th {
    font-size: 0.60rem !important;
    background: #091525; color: #3a7aaa;
    padding: 7px 5px; font-weight: 700; letter-spacing: 0.5px;
    border-bottom: 1px solid #162035; text-align: left;
    position: sticky; top: 0;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.raw-tbl td {
    font-size: 0.68rem !important;
    padding: 6px 5px; color: #c8daea;
    border-bottom: 1px solid #0d1e30;
    overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
    transition: background 0.15s;
}
.raw-tbl tr:last-child td { border-bottom: none; }
.raw-tbl tr:nth-child(even) td { background: rgba(74,159,212,0.03); }
.raw-tbl tr:hover td { background: rgba(74,159,212,0.10); }
.raw-tbl th:nth-child(1), .raw-tbl td:nth-child(1) { width: 18px; }
.raw-tbl th:nth-child(2), .raw-tbl td:nth-child(2) { width: 90px; }
.raw-tbl th:nth-child(3), .raw-tbl td:nth-child(3) { width: 40px; }
.raw-tbl th:nth-child(4), .raw-tbl td:nth-child(4) { width: 70px; }
.raw-tbl th:nth-child(5), .raw-tbl td:nth-child(5) { width: 45px; }
.raw-tbl th:nth-child(6), .raw-tbl td:nth-child(6) { width: 52px; }

.high { color: #fc8181 !important; font-weight: 700; }
.low  { color: #f6c90e !important; font-weight: 700; }
.ok   { color: #38c97a !important; }

.tbl-wrap { max-height: 300px; overflow-y: auto; border-radius: 8px; border: 1px solid #162035; }
.tbl-wrap::-webkit-scrollbar { width: 3px; }
.tbl-wrap::-webkit-scrollbar-thumb { background: #162035; border-radius: 2px; }

/* === CHART DIVIDER === */
.chart-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, #4a9fd4, transparent);
    margin: 1.8rem 0; opacity: 0.45;
}

/* === CHART SECTION LABEL === */
.chart-section-lbl {
    font-size: 0.75rem !important; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.3px; text-transform: none;
    color: #7a9ab8; margin-bottom: 6px;
    display: flex; align-items: center; gap: 6px;
}

/* === RESULT CARDS === */
.res-card {
    border-radius: 10px; padding: 0.9rem 1rem;
    margin-bottom: 0.75rem; border: 1px solid transparent;
    box-shadow: 0 4px 18px rgba(0,0,0,0.28);
}
.rc-blue   { background: #080f1e; border-color: #162a50; border-left: 3px solid #4a9fd4; }
.rc-green  { background: #081510; border-color: #162a20; border-left: 3px solid #38c97a; }
.rc-red    { background: #180606; border-color: #3a1010; border-left: 3px solid #fc8181; }
.rc-yellow { background: #181205; border-color: #3a2a08; border-left: 3px solid #f6c90e; }

.rc-title { font-size: 0.72rem; font-weight: 600; font-family: 'DM Sans', sans-serif; letter-spacing: 0.3px; text-transform: none; margin-bottom: 7px; }
.rc-blue   .rc-title { color: #4a9fd4; }
.rc-green  .rc-title { color: #38c97a; }
.rc-red    .rc-title { color: #fc8181; }
.rc-yellow .rc-title { color: #f6c90e; }
.res-card p { color: #b8cee0; line-height: 1.7; margin: 0; font-size: 0.80rem; }
.res-card p strong, .res-card p b { font-weight: 600; color: #c8daea; }
.res-card p ol, .res-card p ul { font-size: 0.80rem; color: #b8cee0; padding-left: 1.2rem; margin: 0.4rem 0; }
.res-card li { font-size: 0.80rem; line-height: 1.7; }

.sev-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 14px; border-radius: 20px;
    font-size: 0.70rem; font-weight: 700; margin-bottom: 10px;
    letter-spacing: 0.4px;
}
.sp-high   { background: rgba(252,129,129,0.1); color: #fc8181; border: 1px solid rgba(252,129,129,0.25); }
.sp-medium { background: rgba(246,201,14,0.1);  color: #f6c90e; border: 1px solid rgba(246,201,14,0.25); }
.sp-low    { background: rgba(56,201,122,0.1);  color: #38c97a; border: 1px solid rgba(56,201,122,0.25); }

/* === HIDE STREAMLIT UI === */
footer { display: none !important; }
#MainMenu { display: none !important; }
div[data-testid="stTabs"] button { font-size: 0.76rem !important; color: #3a6a9a !important; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #4a9fd4 !important; border-bottom: 2px solid #4a9fd4 !important; }
</style>
""", unsafe_allow_html=True)

# SESSION STATE
defaults = {
    "reports":       {},
    "active_report": None,
    "chat_history":  [],
    "last_severity": "",
    "last_referral": "",
    "input_tok":     0,
    "output_tok":    0,
    "chat_mode":     "report",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# BLOOD TEST REFERENCE RANGES
PARAMS = {
    "Hemoglobin":          (13.5, 17.5, "g/dL"),
    "WBC":                 (4.5,  11.0, "×10³/µL"),
    "Platelet":            (150,  400,  "×10³/µL"),
    "RBC":                 (4.5,  5.9,  "×10⁶/µL"),
    "Hematocrit":          (41,   53,   "%"),
    "MCV":                 (80,   100,  "fL"),
    "MCH":                 (27,   33,   "pg"),
    "MCHC":                (32,   36,   "g/dL"),
    "Neutrophils":         (40,   70,   "%"),
    "Lymphocytes":         (20,   40,   "%"),
    "Fasting Blood Sugar": (70,   99,   "mg/dL"),
    "HbA1c":               (0,    5.7,  "%"),
    "SGPT":                (7,    40,   "U/L"),
    "SGOT":                (10,   40,   "U/L"),
    "Bilirubin":           (0.2,  1.2,  "mg/dL"),
    "Creatinine":          (0.7,  1.2,  "mg/dL"),
    "Urea":                (7,    20,   "mg/dL"),
    "Iron":                (60,   170,  "µg/dL"),
    "Ferritin":            (12,   300,  "ng/mL"),
    "Vitamin D":           (20,   50,   "ng/mL"),
    "Vitamin B12":         (200,  900,  "pg/mL"),
}

def extract_chart_data(text):
    found = {}
    for p, (lo, hi, u) in PARAMS.items():
        m = re.search(rf"{p}[\s\S]{{0,40}}?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            try:
                found[p] = (float(m.group(1)), lo, hi, u)
            except ValueError:
                pass
    return found

def general_medical_answer(question):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful medical information assistant. "
                    "Answer general medical questions clearly and simply. "
                    "Always remind the user to consult a doctor. "
                    "Keep answers concise (3-5 sentences max)."
                )},
                {"role": "user", "content": question},
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Could not get answer: {e}"

# CHART BUILDERS
def make_range_chart(data):
    params, norm_starts, norm_widths, val_pcts, colors, hover_vals = [], [], [], [], [], []

    for p, (v, lo, hi, u) in data.items():
        params.append(p)
        ref_min = min(v, lo)
        ref_max = max(v, hi)
        padding = (ref_max - ref_min) * 0.2 or 1
        ref_min -= padding
        ref_max += padding
        ref_range = ref_max - ref_min

        ns = (lo - ref_min) / ref_range * 100
        nw = (hi - lo) / ref_range * 100
        vp = (v  - ref_min) / ref_range * 100

        norm_starts.append(ns)
        norm_widths.append(nw)
        val_pcts.append(vp)
        colors.append("#fc8181" if (v < lo or v > hi) else "#38c97a")
        hover_vals.append(f"{v} {u}  |  Normal: {lo}–{hi} {u}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name="Your Value",
        y=params,
        x=val_pcts,
        mode='markers',
        marker=dict(
            color=colors, size=14,
            symbol='diamond',
            line=dict(color='white', width=1.5)
        ),
        customdata=hover_vals,
        hovertemplate="<b>%{y}</b>: %{customdata}<extra></extra>",
    ))

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8daea", family="DM Sans", size=11),
        legend=dict(orientation="h", y=1.08, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            range=[0, 100],
            showticklabels=False,
            gridcolor="rgba(74,159,212,0.05)",
            color="#3a6a9a",
            zeroline=False,
        ),
        yaxis=dict(color="#b8cee0", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
        height=max(320, len(params) * 44),
        margin=dict(l=5, r=15, t=40, b=20),
    )
    return fig

def make_bar_chart(data):
    ps = list(data.keys())
    vs = [v[0] for v in data.values()]
    cs = ["#fc8181" if (v < l or v > h) else "#4a9fd4" for _, (v, l, h, _) in data.items()]

    fig = go.Figure(go.Bar(
        x=ps, y=vs, marker_color=cs,
        text=[f"{v:.1f}" for v in vs],
        textposition='outside',
        textfont=dict(size=10, color="#7a9ab8"),
        hovertemplate="<b>%{x}</b>: %{y}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8daea", family="DM Sans", size=11),
        xaxis=dict(
            gridcolor="rgba(74,159,212,0.06)", color="#3a6a9a",
            tickangle=-40, tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(74,159,212,0.07)", color="#b8cee0",
            title=dict(text="Value", font=dict(size=10, color="#3a6a9a")),
        ),
        height=300, margin=dict(l=5, r=10, t=30, b=90),
    )
    return fig

def make_status_chart(data):
    normal   = sum(1 for _, (v, lo, hi, _) in data.items() if lo <= v <= hi)
    abnormal = len(data) - normal

    fig = go.Figure(go.Pie(
        labels=["Normal", "Abnormal"], values=[normal, abnormal], hole=0.62,
        marker=dict(colors=["#38c97a", "#a78bfa"], line=dict(color="#0b1928", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#e8f0fa"),
        hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
        pull=[0.04, 0.04],
    ))
    fig.add_annotation(
        text=f"<b>{len(data)}</b><br><span style='font-size:0.8em'>Total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e8f0fa"),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8daea", family="DM Sans", size=12),
        legend=dict(orientation="h", y=-0.08, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        height=320, margin=dict(l=20, r=20, t=30, b=30),
    )
    return fig

# SIDEBAR
with st.sidebar:

    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🏥</div>
        <div>
            <div class="sb-logo-name">MedReport AI</div>
            <div class="sb-logo-sub">Medical Analysis Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">AI Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <div class="model-badge">
            <div class="model-dot"></div>
            <div>
                <div class="model-name">llama-3.1-8b-instant</div>
                <div class="model-sub">Fast · Free · Production-ready</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ready = st.session_state.active_report is not None
    st.markdown('<div class="sb-section">System Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card">
        <div class="info-row">
            <span class="info-key">Status</span>
            <span class="status-badge {'status-ready' if ready else 'status-wait'}">
                {'✅ Ready' if ready else '⏳ Awaiting report'}
            </span>
        </div>
        <div class="info-row"><span class="info-key">Agents</span><span class="info-val blue">3 Active</span></div>
        <div class="info-row"><span class="info-key">RAG Pipeline</span><span class="info-val muted">FAISS + HuggingFace</span></div>
        <div class="info-row"><span class="info-key">Embeddings</span><span class="info-val muted">MiniLM-L6-v2</span></div>
        <div class="info-row"><span class="info-key">Reports Loaded</span><span class="info-val blue">{len(st.session_state.reports)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Upload Medical Reports</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "upload", type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state.reports]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)…"):
                for f in new_files:
                    try:
                        raw = extract_text_from_pdfs([f])
                        if not raw.strip():
                            st.warning(f"⚠️ No text in {f.name}")
                            continue
                        chunks = split_text_into_chunks(raw)
                        vs     = create_vector_store(chunks)
                        st.session_state.reports[f.name] = {
                            "vector_store": vs,
                            "chart_data":   extract_chart_data(raw),
                            "raw_text":     raw[:500],
                        }
                        if st.session_state.active_report is None:
                            st.session_state.active_report = f.name
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
            st.success(f"✅ {len(new_files)} report(s) loaded!")
            st.rerun()

    if st.session_state.reports:
        st.markdown('<div class="sb-section">Loaded Reports</div>', unsafe_allow_html=True)
        for name in list(st.session_state.reports.keys()):
            short_name = name[:20] + "…" if len(name) > 20 else name
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.markdown(f"""
                <div style="background:#0c1e35;border:1px solid #172840;border-radius:8px;
                    padding:7px 10px;display:flex;align-items:center;gap:8px;">
                    <div style="background:linear-gradient(135deg,#1a4a8a,#0e2540);
                        border-radius:6px;padding:5px;font-size:0.9rem;">📄</div>
                    <div style="min-width:0;">
                        <div style="font-size:0.71rem;color:#c8daea;font-weight:600;
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{short_name}</div>
                        <div style="font-size:0.58rem;color:#3a6a9a;margin-top:1px;">4.2KB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_{name}"):
                    del st.session_state.reports[name]
                    remaining = list(st.session_state.reports.keys())
                    st.session_state.active_report = remaining[0] if remaining else None
                    st.rerun()

    total_tok = st.session_state.input_tok + st.session_state.output_tok
    st.markdown('<div class="sb-section">Token Usage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card">
        <div class="tok-grid">
            <div class="tok-box"><div class="tok-num">{st.session_state.input_tok:,}</div><span class="tok-lbl">Input</span></div>
            <div class="tok-box"><div class="tok-num">{st.session_state.output_tok:,}</div><span class="tok-lbl">Output</span></div>
        </div>
        <div class="tok-total">
            <span class="tok-total-lbl">Total Tokens</span>
            <span class="tok-total-num">{total_tok:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Reset Tokens"):
        st.session_state.input_tok  = 0
        st.session_state.output_tok = 0
        st.rerun()

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Not a substitute for professional medical advice.
        Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)


# MAIN LAYOUT
active_data = st.session_state.reports.get(st.session_state.active_report, {})
active_vs   = active_data.get("vector_store")
active_cd   = active_data.get("chart_data", {})

left, right = st.columns([1, 1.35], gap="medium")

with left:

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-hdr">💬 Chat Analysis</div>', unsafe_allow_html=True)

    col_r, col_g = st.columns(2)
    with col_r:
        if st.button("📋 Report Mode", key="mode_report", use_container_width=True):
            st.session_state.chat_mode = "report"
            st.rerun()
    with col_g:
        if st.button("🩺 General Q&A", key="mode_general", use_container_width=True):
            st.session_state.chat_mode = "general"
            st.rerun()

    mode = st.session_state.chat_mode
    if mode == "report":
        st.markdown(
            '<div class="mode-indicator mode-report">'
            '📋 <b>Report Mode</b>   Questions answered from your uploaded report'
            '</div>', unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="mode-indicator mode-general">'
            '🩺 <b>General Mode</b> · Ask any medical question (no report needed)'
            '</div>', unsafe_allow_html=True,
        )

    if st.session_state.chat_history:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for entry in st.session_state.chat_history:
            q_safe  = entry["q"].replace("<", "&lt;").replace(">", "&gt;")
            a_safe  = entry["a"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            tag     = "tag-report"  if entry.get("mode") == "report" else "tag-general"
            tag_txt = "📋 REPORT"   if entry.get("mode") == "report" else "🩺 GENERAL"
            bubble  = "bubble-a"    if entry.get("mode") == "report" else "bubble-a general"
            st.markdown(
                f'<div class="chat-lbl-q">YOU</div>'
                f'<div class="bubble-q">{q_safe}</div>'
                f'<div class="chat-lbl-a">'
                f'<span class="mode-tag {tag}">{tag_txt}</span> 🤖 MEDREPORT AI'
                f'</div>'
                f'<div class="{bubble}">{a_safe}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        hint = "Upload a report and ask a question…" if mode == "report" else "Ask any medical question — e.g. 'What is hemoglobin?'"
        st.markdown(f'<div class="chat-empty">💬 {hint}</div>', unsafe_allow_html=True)

    ph = "e.g. What is my hemoglobin level?" if mode == "report" else "e.g. What causes iron deficiency?"
    question = st.text_input("q", placeholder=ph, label_visibility="collapsed")

    c1, c2 = st.columns([2, 1])
    with c1:
        analyze = st.button("🔍 Analyze", use_container_width=True)
    with c2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history  = []
            st.session_state.last_severity = ""
            st.session_state.last_referral = ""
            st.rerun()

    if analyze:
        if not question.strip():
            st.warning("⚠️ Please type a question.")
        elif mode == "report" and not active_vs:
            st.warning("⚠️ Upload a report first, or switch to General Q&A.")
        else:
            with st.spinner("🤖 Thinking…"):
                try:
                    if mode == "general":
                        answer = general_medical_answer(question)
                        st.session_state.input_tok  += len(question.split()) * 2
                        st.session_state.output_tok += len(answer.split())
                        st.session_state.chat_history.append({"q": question, "a": answer, "mode": "general"})
                    else:
                        ctx      = search_vector_store(active_vs, question)
                        answer   = run_report_agent(ctx, question)
                        severity = run_severity_agent(ctx)
                        referral = run_referral_agent(ctx, severity)
                        st.session_state.input_tok  += len(question.split()) * 2
                        st.session_state.output_tok += len(answer.split()) + len(severity.split())
                        st.session_state.chat_history.append({"q": question, "a": answer, "mode": "report"})
                        st.session_state.last_severity = severity
                        st.session_state.last_referral = referral
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    if active_cd:
        total_p = len(active_cd)
        abn_p   = sum(1 for _, (v, lo, hi, _) in active_cd.items() if v < lo or v > hi)
        norm_p  = total_p - abn_p

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-hdr">📄 Document Overview</div>', unsafe_allow_html=True)
        st.markdown('<div style="border-top:1px solid #162035;margin:0.6rem 0 0.8rem 0;"></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:0.72rem;color:#3a6a9a;margin-bottom:0.7rem;">'
            f'Analyzing: <span style="color:#4a9fd4;font-weight:600;">{st.session_state.active_report}</span>'
            f'</div>', unsafe_allow_html=True,
        )

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-num" style="color:#e8f0fa;">{total_p}</div>
                <span class="stat-lbl">Parameters</span>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#fc8181;">{abn_p}</div>
                <span class="stat-lbl">Abnormal</span>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#38c97a;">{norm_p}</div>
                <span class="stat-lbl">Normal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.60rem;font-weight:700;letter-spacing:1.2px;'
            'color:#3a7aaa;text-transform:uppercase;margin-bottom:7px;">🔬 Raw Data</div>',
            unsafe_allow_html=True,
        )

        rows = ""
        for i, (p, (v, lo, hi, u)) in enumerate(active_cd.items()):
            if v > hi:   css, tag = "high", "↑ HIGH"
            elif v < lo: css, tag = "low",  "↓ LOW"
            else:        css, tag = "ok",   "✓ OK"
            rows += (
                f"<tr>"
                f"<td style='color:#3a6a9a;'>{i+1}</td>"
                f"<td>{p}</td>"
                f"<td class='{css}'>{v}</td>"
                f"<td style='color:#4a7a9a;'>{lo}–{hi}</td>"
                f"<td style='color:#4a7a9a;'>{u}</td>"
                f"<td class='{css}'>{tag}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div class="tbl-wrap">
        <table class="raw-tbl">
            <thead><tr>
                <th>#</th><th>Parameter</th><th>Val</th>
                <th>Ref Range</th><th>Unit</th><th>Status</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


with right:

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-hdr">📊 Blood Parameters Visualization</div>', unsafe_allow_html=True)

    if active_cd:
        st.markdown('<div class="chart-section-lbl">📊 Range Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(make_range_chart(active_cd), use_container_width=True)

        lc1, lc2 = st.columns(2)
        lc1.markdown('<p style="font-size:0.74rem;color:#4a9fd4;margin:0;">🔷 Normal</p>',  unsafe_allow_html=True)
        lc2.markdown('<p style="font-size:0.74rem;color:#fc8181;margin:0;">🔴 Abnormal</p>', unsafe_allow_html=True)

        st.markdown('<hr class="chart-divider">', unsafe_allow_html=True)

        st.markdown('<div class="chart-section-lbl">📈 Bar Chart</div>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(active_cd), use_container_width=True)

        st.markdown('<hr class="chart-divider">', unsafe_allow_html=True)

        st.markdown('<div class="chart-section-lbl">🥧 Status Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(make_status_chart(active_cd), use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem;">
            <div style="font-size:3rem; opacity:0.2; margin-bottom:1rem;">📊</div>
            <div style="font-size:0.85rem; color:#1e3a5a;">
                Upload a medical report to view charts
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.last_severity:
        sev = st.session_state.last_severity.upper()

        if "HIGH" in sev:
            sc, si, sp, label = "rc-red",    "🔴", "sp-high",   "HIGH SEVERITY"
        elif "MEDIUM" in sev:
            sc, si, sp, label = "rc-yellow", "🟡", "sp-medium", "MEDIUM SEVERITY"
        else:
            sc, si, sp, label = "rc-green",  "🟢", "sp-low",    "LOW SEVERITY"

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-hdr">📋 Detailed Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="res-card {sc}">
            <div class="rc-title">{si} Severity Assessment</div>
            <div class="sev-pill {sp}">{si} {label}</div>
            <p>{st.session_state.last_severity}</p>
        </div>
        <div class="res-card rc-blue">
            <div class="rc-title">👨‍⚕️ Doctor Referral</div>
            <p>{st.session_state.last_referral}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif not active_cd:
        st.markdown("""
        <div class="panel">
            <div class="panel-hdr">📋 Analysis Results</div>
            <div style="text-align:center; padding:2.5rem 1rem;">
                <div style="font-size:2.2rem; opacity:0.2; margin-bottom:0.6rem;">🩺</div>
                <div style="font-size:0.80rem; color:#1e3a5a;">
                    Upload a report and click Analyze<br>to see results here
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)