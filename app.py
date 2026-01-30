import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Simplified Analytics", page_icon="📊", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configuration")
    # Switch to Gemini API Key
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    st.divider()
    st.caption("© 2026 Data Scientist: Mr. Sonu Sourav | Powered by: Gemini 🚀")

# --- GEMINI AI FUNCTION ---
def analyze_with_gemini(df, user_query, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Strict logic prompt to prevent Rajesh/Ramya hallucinations
        system_prompt = """
        ROLE: You are a Forensic Data Auditor. 
        
        STRICT RULES:
        1. DATA SCOPE: You are analyzing a CSV with 90+ rows. Rows like 'Leeward' and 'Linzess' at the bottom are critical.
        2. INTERSECTION LOGIC: If asked for projects where Name A and Name B worked together, you MUST check if BOTH exist in the SAME 'Team Members' cell.
        3. NO HALLUCINATIONS: If 'Rajesh' is not in the 'PCV HCP' row, you are FORBIDDEN from listing it.
        4. VERIFICATION: For every match, output the project name followed by the exact team list from the data in brackets.
        5. TONE: Formal, professional, and zero conversational filler.
        """
        
        data_full_context = f"DATAFRAME CONTENT:\n{df.to_csv(index=False)}"
        
        # Combine instructions and data
        full_prompt = f"{system_prompt}\n\nDATASET:\n{data_full_context}\n\nUSER QUERY: {user_query}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"System Error: {str(e)}"

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Pre-processing for clean parsing [cite: 1, 5]
        df['Team Members'] = df['Team Members'].replace(to_replace=[r',?\s+and\s+', r'\s+&\s+'], value=', ', regex=True)

        tab1, tab2, tab3 = st.tabs(["📄 Dataset Explorer", "📈 Visual Analytics", "💼 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=500)

        with tab3:
            st.subheader("Consultancy Portal (Gemini Engine)")
            query = st.text_input("Ask about project teams (e.g., 'Projects with Rajesh and Ramya'):")
            if st.button("Generate Verified Insight"):
                if not GEMINI_API_KEY:
                    st.warning("Please enter your Gemini API Key.")
                else:
                    with st.spinner("Gemini is performing a forensic audit..."):
                        answer = analyze_with_gemini(df, query, GEMINI_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Please upload the CSV to initialize the Gemini Analysis Engine.")
