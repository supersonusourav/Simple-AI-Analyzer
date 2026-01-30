import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Simplified Analytics", page_icon="📊", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configuration")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    st.divider()
    st.caption("© 2026 Data Scientist: Mr. Sonu Sourav | Powered by: Gemini 🚀")

# --- GEMINI AI FUNCTION ---
def analyze_with_gemini(df, user_query, api_key):
    try:
        genai.configure(api_key=api_key)
        
        # Use 'gemini-1.5-flash' - ensured compatibility
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = """
        ROLE: You are a Forensic Data Auditor. 
        
        STRICT PROTOCOL:
        1. INTERSECTION CHECK: When asked about two people working together, you MUST verify that BOTH names are in the SAME 'Team Members' cell.
        2. NO HALLUCINATIONS: Do not list 'PCV HCP' or 'AUD HCP' for Rajesh/Ramya unless BOTH are in that row. 
        3. DATA SOURCE: Use ONLY the provided CSV string. Do not use outside knowledge.
        4. OUTPUT: Provide a bulleted list of matches. For each match, include the Team Members in brackets to prove it is a real match.
        """
        
        # Pass the full dataset context
        data_full_context = f"DATAFRAME CONTENT:\n{df.to_csv(index=False)}"
        
        full_prompt = f"{system_prompt}\n\nDATASET:\n{data_full_context}\n\nUSER QUERY: {user_query}"
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # Better error message for debugging
        return f"Consultant Error: {str(e)}. Tip: Try updating the library with 'pip install -U google-generativeai'."

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # PRE-PROCESSING: Convert 'and' to commas for both UI and AI clarity
        df['Team Members'] = df['Team Members'].replace(to_replace=[r',?\s+and\s+', r'\s+&\s+'], value=', ', regex=True)

        tab1, tab2, tab3 = st.tabs(["📄 Dataset Explorer", "📈 Visual Analytics", "💼 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=500)

        with tab2:
            st.subheader("Interactive Visualizations")
            cols = df.columns.tolist()
            c1, c2, c3 = st.columns(3)
            with c1: x_axis = st.selectbox("X-Axis", options=cols, index=0)
            with c2: y_axis = st.selectbox("Y-Axis", options=cols, index=min(1, len(cols)-1))
            with c3: hover = st.selectbox("Hover Info", options=cols, index=min(2, len(cols)-1))
            fig = px.bar(df, x=x_axis, y=y_axis, hover_data=[hover], color=y_axis, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Consultancy Portal (Gemini Engine)")
            query = st.text_input("Enter your query (e.g., 'Projects where Rajesh and Ramya worked together'):")
            if st.button("Generate Verified Insight"):
                if not GEMINI_API_KEY:
                    st.warning("Please enter your Gemini API Key.")
                else:
                    with st.spinner("Gemini is auditing the full dataset..."):
                        answer = analyze_with_gemini(df, query, GEMINI_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Please upload a CSV file to start.")
