import streamlit as st
import requests
import json
import time

# --- Configuration ---
# API_BASE_URL = "http://127.0.0.1:8000"  # For local testing
API_BASE_URL = "https://findoc-risk-engine.onrender.com" 

st.set_page_config(
    page_title="FinDoc AI",
    page_icon="🏦",
    layout="centered"
)

# --- Custom CSS for "Bank Dashboard" Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #0f2c59; /* Navy Blue */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    h3 {
        color: #444;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        margin-top: 5px;
    }
    .stButton>button {
        background-color: #0f2c59;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1a4a8d;
        color: white;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        border-left: 5px solid #0f2c59;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-med {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# --- Sidebar ---
with st.sidebar:
    st.title("FinDoc AI")
    st.info("Agentic Financial Risk & Compliance Analyzer")
    
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** a financial report (PDF).
    2. **Ask** a risk or compliance question.
    3. **Review** AI-generated analysis with citations.
    """)
    
    st.warning("Disclaimer: AI-assisted analysis. Verify all findings with original documents before making financial decisions.")
    
    api_status = st.empty()
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            api_status.success("System Online")
        else:
            api_status.error("System Offline")
    except:
        api_status.error("System Offline")

# --- Main Layout ---

# 🟦 Top Header
st.title("FinDoc AI")
st.markdown("### Agentic Financial Risk & Compliance Analyzer")
st.divider()

# 🟨 Step 1: Upload
st.header("1️⃣ Upload Financial Report")

uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'], label_visibility="collapsed")

if uploaded_file is not None:
    # Only upload if it's a new file or hasn't been processed
    if st.session_state.uploaded_file != uploaded_file.name:
        with st.status("Processing document...", expanded=True) as status:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                
                if response.status_code == 200:
                    st.session_state.uploaded_file = uploaded_file.name
                    status.update(label="Document processed successfully!", state="complete", expanded=False)
                    st.success(f"File '{uploaded_file.name}' is ready for analysis.")
                else:
                    status.update(label="Upload failed", state="error")
                    st.error(f"Error: {response.text}")
            except Exception as e:
                status.update(label="Connection error", state="error")
                st.error(f"Failed to connect to backend: {str(e)}")
    else:
        st.success(f"'{uploaded_file.name}' ready.")

else:
    st.session_state.uploaded_file = None
    st.info("Please upload a PDF to begin.")

# 🟩 Step 2: Ask Question
st.markdown("---")
st.header("2️⃣ Ask a Financial Question")

query_input = st.text_input(
    "Enter your query", 
    placeholder="e.g., Analyze debt risk and liabilities",
    disabled=(st.session_state.uploaded_file is None),
    label_visibility="collapsed"
)

if st.button("Analyze Risk", disabled=(st.session_state.uploaded_file is None or not query_input)):
    with st.spinner("Running AI Analysis Agents..."):
        try:
            payload = {"question": query_input}
            response = requests.post(f"{API_BASE_URL}/query", json=payload)
            
            if response.status_code == 200:
                st.session_state.analysis_result = response.json()
            else:
                st.error(f"Analysis failed: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to analysis engine: {str(e)}")

# 🟥 Step 3: Results
if st.session_state.analysis_result:
    st.markdown("---")
    st.header("3️⃣ Risk Analysis Result")
    
    result = st.session_state.analysis_result
    
    # Extract data
    answer = result.get("answer", "No answer provided.")
    confidence = result.get("confidence", 0.0)
    sources = result.get("sources", [])
    metrics = result.get("metrics", {})
    ratios = result.get("ratios", {})
    
    # 🎯 Confirm Visualization Color
    conf_color = "confidence-low"  # Default
    conf_label = "Low"
    if confidence > 0.75:
        conf_color = "confidence-high" 
        conf_label = "High"
    elif confidence > 0.5:
        conf_color = "confidence-med"
        conf_label = "Medium"
        
    conf_percentage = int(confidence * 100)

    # 📊 Result Block
    st.markdown(f"""
    <div class="result-card">
        <h4>Analysis Report</h4>
        <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
        <hr>
        <p>Confidence Score: <span class="{conf_color}">{conf_percentage}% ({conf_label})</span></p>
        <div style="background-color: #eee; border-radius: 5px; height: 10px; width: 100%;">
            <div style="background-color: {'#28a745' if confidence > 0.75 else '#ffc107' if confidence > 0.5 else '#dc3545'}; 
                        width: {conf_percentage}%; height: 100%; border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 📉 Financial Metrics Dashboard
    if metrics or ratios:
        st.markdown("<br><h4>Key Financial Data</h4>", unsafe_allow_html=True)
        
        # Merge dicts for display
        display_data = {**metrics, **ratios}
        
        # Create 3 columns grid
        cols = st.columns(3)
        for i, (key, value) in enumerate(display_data.items()):
            if value is not None:
                # Format label
                label = key.replace("_", " ").title()
                # Format value
                if isinstance(value, float):
                    val_str = f"{value:,.2f}"
                else:
                    val_str = str(value)
                    
                with cols[i % 3]:
                    st.metric(label=label, value=val_str)

    # 📚 Sources Block
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 Evidence from Report (Sources)"):
        if sources:
            for i, source in enumerate(sources):
                page_num = source.get("page_no", "N/A")
                content = source.get("snippet", "No text snippet available.")
                st.markdown(f"**Source {i+1} (Page {page_num})**")
                st.caption(f"\"{content.strip()}\"")
                st.divider()
        else:
            st.info("No specific text sources cited for this answer.")
