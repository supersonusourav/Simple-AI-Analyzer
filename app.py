import streamlit as st
import pandas as pd
import google.generativeai as genai
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Data Analyzer Pro", layout="wide", page_icon="📊")

# --- SIDEBAR: INPUTS & COPYRIGHT ---
with st.sidebar:
    st.title("🛠️ Configuration")
    
    # 1. API Key Input
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Paste your AIza... key here")
    
    # 2. File Upload
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Dataset (.csv)", type=["csv"])
    
    # 3. Copyright Notice
    st.markdown("---")
    st.caption("© 2026 Your Name/Company. All rights reserved.")

# --- DATA PROCESSING ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def analyze_with_ai(df, prompt, key):
    genai.configure(api_key=key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite') # Switched to Lite
    
    # TRICK: Only send column names and the first 5 rows to stay under the limit
    data_summary = f"Columns: {list(df.columns)}\n\nFirst 5 rows:\n{df.head(5).to_csv(index=False)}"
    
    full_prompt = f"Analyze this data:\n{data_summary}\n\nUser Question: {prompt}"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- MAIN PAGE UI ---
st.title("📊 AI-Powered Data Insights")

if uploaded_file:
    df = load_data(uploaded_file)
    
    tab1, tab2 = st.tabs(["🔍 Data Explorer", "🤖 AI Analyst"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)
        
    with tab2:
        st.subheader("Ask the AI about your data")
        query = st.text_input("Example: 'What are the top 3 trends in this data?'")
        
        if st.button("Generate Analysis"):
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar.")
            elif not query:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing..."):
                    answer = analyze_with_ai(df, query, api_key)
                    st.markdown("### AI Insight")
                    st.write(answer)
else:
    st.info("👈 Please upload a CSV file in the sidebar to begin.")
