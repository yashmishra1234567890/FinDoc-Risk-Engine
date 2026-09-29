import streamlit as st
import requests
import json
import time
import os

# --- Configuration ---
# Use environment variable for backend URL to support both local and production seamlessly
API_BASE_URL = os.getenv("BACKEND_URL", "https://findoc-risk-engine.onrender.com")

st.set_page_config(
    page_title="FinDoc Risk Engine",
    page_icon="🏦",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    :root {
        --bg: #08111f;
        --panel: rgba(10, 18, 32, 0.82);
        --panel-2: rgba(18, 31, 54, 0.92);
        --line: rgba(148, 163, 184, 0.18);
        --text: #e8edf7;
        --muted: #94a3b8;
        --accent: #7dd3fc;
        --accent-2: #34d399;
        --warn: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(52, 211, 153, 0.12), transparent 24%),
            linear-gradient(180deg, #07111d 0%, #0b1425 42%, #08111f 100%);
        color: var(--text);
    }

    .main {
        background: transparent;
    }

    h1, h2, h3, h4, h5, p, span, label {
        color: var(--text);
        font-family: Inter, "Segoe UI", sans-serif;
    }

    .hero {
        padding: 1.1rem 1.3rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(18, 31, 54, 0.96), rgba(8, 17, 31, 0.88));
        box-shadow: 0 30px 80px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0.35rem 0 0;
        color: var(--muted);
    }

    .glass-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
    }

    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1rem;
    }

    .kpi {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(18, 31, 54, 0.92));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
    }

    .kpi .label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .kpi .value {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .kpi .sub {
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }

    .stButton button {
        border-radius: 14px;
        border: 1px solid rgba(125, 211, 252, 0.25);
        background: linear-gradient(135deg, rgba(29, 78, 216, 0.88), rgba(14, 165, 233, 0.92));
        color: white;
        font-weight: 600;
        padding: 0.7rem 1rem;
    }

    .stButton button:hover {
        border-color: rgba(125, 211, 252, 0.55);
        transform: translateY(-1px);
    }

    .stChatMessage {
        background: rgba(10, 18, 32, 0.66);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 0.4rem 0.75rem;
        margin-bottom: 0.75rem;
    }

    .stProgress > div > div {
        background-color: var(--accent);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8, 17, 31, 0.98), rgba(10, 18, 32, 0.98));
        border-right: 1px solid var(--line);
    }

    .small-note {
        color: var(--muted);
        font-size: 0.84rem;
    }

    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'active_file_name' not in st.session_state:
    st.session_state.active_file_name = None


def stream_query(question: str):
    payload = {"question": question, "stream": True}
    response = requests.post(f"{API_BASE_URL}/query", json=payload, stream=True, timeout=120)
    response.raise_for_status()

    answer_parts = []
    for chunk in response.iter_content(chunk_size=32, decode_unicode=True):
        if chunk:
            answer_parts.append(chunk)
            yield "".join(answer_parts)


def query_once(question: str):
    response = requests.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


def render_kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Sidebar ---
with st.sidebar:
    st.markdown("## FinDoc Risk Engine")
    st.caption("Agentic financial analysis with evidence, validation, and market context.")
    
    st.markdown("---")
    st.markdown("### 1. Upload Document")
    uploaded_file = st.file_uploader("Select PDF Report", type=['pdf'], label_visibility="collapsed")
    
    # --- Sample Selection ---
    st.markdown("---")
    st.markdown("### Try a sample")
    
    sample_map = {
        "Vodafone Idea (Telecom)": "data/financial_docs/VIL-QR-Q1FY25.pdf",
        "Tech Mahindra (IT Service)": "data/financial_docs/Financial Performance Report.pdf"
    }

    selected_sample = st.selectbox("Choose a sample report:", ["-- Select --"] + list(sample_map.keys()))
    
    process_sample = False
    if selected_sample != "-- Select --":
        if st.button(f"📄 Load {selected_sample}", use_container_width=True):
            process_sample = True

    # Logic: Prioritize manual upload, otherwise handle sample load
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file.name:
            with st.status("Ingesting document...", expanded=True) as status:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    # 1. Upload File
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        task_data = response.json()
                        task_id = task_data.get("task_id")
                        st.write("File uploaded. Processing content...")
                        
                        # 2. Poll for Status
                        if task_id:
                            max_retries = 180 # wait max 6 mins (2s * 180) to handle larger files
                            for _ in range(max_retries):
                                time.sleep(2)
                                status_res = requests.get(f"{API_BASE_URL}/upload/status/{task_id}")
                                if status_res.status_code == 200:
                                    s = status_res.json()
                                    if s["status"] == "completed":
                                        st.session_state.uploaded_file = uploaded_file.name
                                        st.session_state.active_file_name = uploaded_file.name
                                        status.update(label="Ready for Analysis", state="complete", expanded=False)
                                        st.success("Analysis Ready! You can now use the dashboard.")
                                        st.rerun() # Refresh to update state
                                        break
                                    elif s["status"] == "failed":
                                        status.update(label="Processing Failed", state="error")
                                        st.error(f"Error: {s.get('message')}")
                                        break
                            else:
                                status.update(label="Timeout", state="error")
                                st.error("Processing timed out. The file might be too large.")
                        else:
                             # Fallback for old API version
                             st.session_state.uploaded_file = uploaded_file.name
                             status.update(label="Ready (Legacy)", state="complete")
                    else:
                        status.update(label="Upload Failed", state="error")
                        st.error(response.text)
                except Exception as e:
                    status.update(label="Connection Error", state="error")
                    st.error(str(e))
        else:
            st.success(f"✅ {uploaded_file.name}")
            
    elif process_sample:
        # Handle Sample Upload
        f_path = sample_map[selected_sample]
        f_name = os.path.basename(f_path)
        
        if st.session_state.uploaded_file != f_name:
             with st.status(f"Ingesting {selected_sample}...", expanded=True) as status:
                try:
                    if not os.path.exists(f_path):
                        st.error(f"Sample file not found at {f_path}")
                        st.stop()
                        
                    with open(f_path, "rb") as f:
                        file_bytes = f.read()

                    files = {"file": (f_name, file_bytes, "application/pdf")}
                    
                    # 1. Upload File
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        task_data = response.json()
                        task_id = task_data.get("task_id")
                        st.write("Sample uploaded. Processing content...")
                        
                        # 2. Poll for Status
                        if task_id:
                            max_retries = 180 # increased for bigger annual reports
                            for _ in range(max_retries):
                                time.sleep(2)
                                status_res = requests.get(f"{API_BASE_URL}/upload/status/{task_id}")
                                if status_res.status_code == 200:
                                    s = status_res.json()
                                    if s["status"] == "completed":
                                        st.session_state.uploaded_file = f_name
                                        st.session_state.active_file_name = f_name
                                        status.update(label="Ready for Analysis", state="complete", expanded=False)
                                        st.success(f"Analysis Ready for {f_name}!")
                                        st.rerun()
                                        break
                                    elif s["status"] == "failed":
                                        status.update(label="Processing Failed", state="error")
                                        st.error(f"Error: {s.get('message')}")
                                        break
                            else:
                                status.update(label="Timeout", state="error")
                        else:
                             st.session_state.uploaded_file = f_name
                             status.update(label="Ready", state="complete")

                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
             st.success(f"✅ {f_name}")
    
    # Show current file if selected via sample and no upload widget active
        if uploaded_file is None and st.session_state.uploaded_file:
            st.info(f"📁 Active File: {st.session_state.uploaded_file}")

        if st.session_state.active_file_name:
            st.caption(f"Working set: {st.session_state.active_file_name}")
    
    st.markdown("---")
    
    # API Status
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1)
        if r.status_code == 200:
            st.caption("🟢 System Online")
            st.caption(f"Documents indexed: {r.json().get('documents_indexed', 0)}")
    except:
        st.caption("🔴 System Offline")

    st.warning("⚠️ **Disclaimer:** This demo runs on a free cloud instance. If the system is offline or slow, please wait 60 seconds for the server to wake up.")

# --- Main Interface ---

if not st.session_state.uploaded_file:
    st.markdown(
        """
        <div class="hero">
            <h1>FinDoc Risk Engine</h1>
            <p>Upload a report to unlock agentic extraction, market-aware validation, and evidence-backed answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload a financial PDF report (Balance Sheet, P&L, Annual Report) using the sidebar to begin.")
    st.stop()

# Tabs for Mode
tab_dashboard, tab_chat = st.tabs(["📊 Risk Dashboard", "💬 Chat Assistant"])

# ================= DASHBOARD TAB =================
with tab_dashboard:
    st.markdown(
        """
        <div class="hero">
            <h1>Financial Risk Dashboard</h1>
            <p>Quick actions, validated metrics, and PDF evidence in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Pre-defined quick actions
    cols = st.columns(3)
    with cols[0]:
        if st.button("🚨 Analyze Credit Risk", use_container_width=True):
            st.session_state.current_query = "Analyze debt, liabilities, and repayment capacity risk."
    with cols[1]:
        if st.button("💰 Summarize Revenue", use_container_width=True):
            st.session_state.current_query = "What is the revenue, profit, and growth?"
    with cols[2]:
        if st.button("⚖️ Compliance Check", use_container_width=True):
            st.session_state.current_query = "Check for compliance issues and red flags."
            
    # Check if a query was triggered
    if 'current_query' in st.session_state:
        with st.spinner("AI agents working: decomposing, retrieving, validating..."):
            try:
                st.session_state.analysis_result = query_once(st.session_state.current_query)
                del st.session_state.current_query # Clear trigger
            except requests.Timeout:
                st.error("⚠️ logic timeout: The file is complex and the AI needed more time. Please try asking a simpler question.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Results if available
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        compliance = res.get("compliance", {})
        market_context = compliance.get("market_context", {}) if compliance else {}
        sources = res.get("sources", [])

        kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
        with kpi_1:
            render_kpi("Confidence", f"{int((res.get('confidence', 0.0) or 0) * 100)}%", "Model-backed score")
        with kpi_2:
            render_kpi("Sources", str(len(sources)), "Unique source pages")
        with kpi_3:
            render_kpi("VIX", f"{market_context.get('vix', 0):.2f}" if market_context else "n/a", "Market temperature")
        with kpi_4:
            render_kpi("10Y Yield", f"{market_context.get('ten_year_yield', 0):.2f}%" if market_context else "n/a", "Rate pressure")
        
        # 1. Metrics Grid
        st.markdown("#### 🔢 Key Financial Indicators")
        metrics = {**res.get("metrics", {}), **res.get("ratios", {})}
        
        if metrics:
            m_cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                # Explicit check: Show even if 0.0, but hide if None/Null
                if v is not None:
                    nice_key = k.replace("_", " ").title()
                    nice_val = f"{v:,.2f}" if isinstance(v, float) else str(v)
                    m_cols[i % 4].metric(label=nice_key, value=nice_val)
        else:
            st.info("No numeric metrics found for this specific query.")

        st.divider()

        # 2. Narrative & Confidence
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("#### 📝 Analysis Report")
            st.write(res.get("answer"))
            
        with c2:
            st.markdown("#### 🎯 Confidence")
            conf = res.get("confidence", 0.0)
            st.metric("AI Confidence Score", f"{int(conf*100)}%")
            st.progress(conf)
            if conf < 0.5:
                st.caption("⚠️ Low confidence: Data might be missing.")

        calculation = res.get("calculation")
        if calculation:
            st.markdown("#### 🧮 Calculation")
            st.code(calculation.get("formula", ""), language="text")
            st.write(f"Result: {calculation.get('result')}")
            if calculation.get("operands"):
                st.caption(f"Operands: {calculation['operands']}")

        verification = res.get("verification")
        if verification:
            st.markdown("#### ✅ Verification")
            st.write(verification.get("status", "Not available"))
            if verification.get("reason"):
                st.caption(verification["reason"])

        if compliance:
            st.markdown("#### 🛡️ Validation Signals")
            st.caption(
                f"Temperature risk: {compliance.get('temperature_risk', 0):.2f} | "
                f"Market context: {compliance.get('market_context', {})}"
            )
            st.write(" • ".join(compliance.get("rule_engine_flags", [])[:4]))

        risk_summary = res.get("risk_summary", [])
        if risk_summary:
            st.markdown("#### ⚠️ Risk Summary")
            for flag in risk_summary[:6]:
                st.write(f"- {flag}")

        st.markdown("#### 📌 Why this answer")
        st.write(res.get("answer"))

        # 3. Sources
        with st.expander("📄 View Source Evidence"):
            for s in res.get("sources", []):
                table_note = f" | Table {s['table_id']}" if s.get("table_id") else ""
                st.markdown(f"**Page {s['page_no']}**{table_note}: {s['snippet']}")
                if s.get("image_base64"):
                    st.image(s["image_base64"], caption=f"Page {s['page_no']} Highlighted")


# ================= CHAT TAB =================
with tab_chat:
    st.markdown(
        """
        <div class="hero">
            <h1>Chat Assistant</h1>
            <p>Ask for explanations, source citations, or follow-up analysis in plain language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **Pg {s['page_no']}**: {s['snippet']}")
            if msg.get("compliance"):
                with st.expander("Validation"):
                    st.caption(" • ".join(msg["compliance"].get("rule_engine_flags", [])[:5]))

    # Input
    if user_input := st.chat_input("Ask about the document..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = {}
                    placeholder = st.empty()
                    answer_text = ""
                    for partial in stream_query(user_input):
                        answer_text = partial
                        placeholder.write(answer_text)

                    if not answer_text:
                        resp = query_once(user_input)
                        answer_text = resp.get("answer", "No answer found.")
                        placeholder.write(answer_text)
                        sources = resp.get("sources", [])
                    else:
                        sources = []
                    
                    # Store logic
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "compliance": resp.get("compliance", {}),
                    })
                    
                    # Show sources immediately for this turn
                    if sources:
                        with st.expander("Sources"):
                             for s in sources:
                                st.markdown(f"- **Pg {s['page_no']}**: {s['snippet']}")
                                if s.get("image_base64"):
                                    st.image(s["image_base64"], caption=f"Page {s['page_no']} Highlighted")
                    if resp.get("compliance"):
                        with st.expander("Validation"):
                            st.write(" • ".join(resp.get("compliance", {}).get("rule_engine_flags", [])[:5]))
                                
                except Exception as e:
                    st.error("Error connecting to agent.")
