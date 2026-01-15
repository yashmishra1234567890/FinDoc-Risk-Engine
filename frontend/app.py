import streamlit as st
import requests
import json
import time

# --- Configuration ---
# Use environment variable for backend URL to support both local and production seamlessly
API_BASE_URL = os.getenv("BACKEND_URL", "https://findoc-risk-engine.onrender.com")

st.set_page_config(
    page_title="FinDoc AI",
    page_icon="🏦",
    layout="wide"  # Changed to wide for better dashboard view
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #0f2c59;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stProgress .st-bo {
        background-color: #0f2c59;
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

# --- Sidebar ---
with st.sidebar:
    st.title("FinDoc AI")
    st.markdown("**Agentic Financial Risk Analyzer**")
    
    st.markdown("---")
    st.header("1️⃣ Upload Document")
    uploaded_file = st.file_uploader("Select PDF Report", type=['pdf'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file.name:
            with st.status("Ingesting Document...", expanded=True) as status:
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
                            max_retries = 60 # wait max 2 mins (2s * 60)
                            for _ in range(max_retries):
                                time.sleep(2)
                                status_res = requests.get(f"{API_BASE_URL}/upload/status/{task_id}")
                                if status_res.status_code == 200:
                                    s = status_res.json()
                                    if s["status"] == "completed":
                                        st.session_state.uploaded_file = uploaded_file.name
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
    
    st.markdown("---")
    
    # API Status
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1)
        if r.status_code == 200:
            st.caption("🟢 System Online") 
    except:
        st.caption("🔴 System Offline")

# --- Main Interface ---

if not st.session_state.uploaded_file:
    st.info("👋 Welcome to FinDoc AI. Please upload a financial PDF report (Balance Sheet, P&L, Annual Report) to the sidebar to begin.")
    st.stop()

# Tabs for Mode
tab_dashboard, tab_chat = st.tabs(["📊 Risk Dashboard", "💬 Chat Assistant"])

# ================= DASHBOARD TAB =================
with tab_dashboard:
    st.subheader("Financial Risk Assessment")
    
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
        with st.spinner("🤖 AI Agents working: Decomposing... Retrieving... Calculating..."):
            try:
                payload = {"question": st.session_state.current_query}
                # Increased timeout to 120s to allow for deep RAG analysis
                response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)
                if response.status_code == 200:
                    st.session_state.analysis_result = response.json()
                    del st.session_state.current_query # Clear trigger
                else:
                    st.error(f"Analysis Failed: {response.text}")
            except requests.Timeout:
                st.error("⚠️ logic timeout: The file is complex and the AI needed more time. Please try asking a simpler question.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Results if available
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        
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

        # 3. Sources
        with st.expander("📄 View Source Evidence"):
            for s in res.get("sources", []):
                st.markdown(f"**Page {s['page_no']}**: {s['snippet']}")


# ================= CHAT TAB =================
with tab_chat:
    st.subheader("💬 Ask Anything")
    
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **Pg {s['page_no']}**: {s['snippet']}")

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
                    payload = {"question": user_input}
                    resp = requests.post(f"{API_BASE_URL}/query", json=payload).json()
                    
                    answer_text = resp.get("answer", "No answer found.")
                    st.write(answer_text)
                    
                    # Store logic
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": resp.get("sources", [])
                    })
                    
                    # Show sources immediately for this turn
                    if resp.get("sources"):
                        with st.expander("Sources"):
                             for s in resp["sources"]:
                                st.markdown(f"- **Pg {s['page_no']}**: {s['snippet']}")
                                
                except Exception as e:
                    st.error("Error connecting to agent.")
