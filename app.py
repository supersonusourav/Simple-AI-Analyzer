import streamlit as st
import pandas as pd
from groq import Groq
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Simplified Analytics", page_icon="📊", layout="wide")

# --- SIDEBAR & DATA LOADING ---
with st.sidebar:
    st.header("🛠️ Configuration")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    st.divider()
    st.caption("© 2026 Data Scientist: Mr. Sonu Sourav | Powered by: Groq 🚀")

# --- ENHANCED SYSTEM PROMPT ---
def analyze_with_groq(df, user_query, api_key):
    try:
        client = Groq(api_key=api_key)
        
        # This prompt addresses all previous issues: 
        # 1. Full data visibility 
        # 2. 'And' vs Comma parsing 
        # 3. Co-occurrence logic (Both names must be in the same row)
        system_prompt = """
        ROLE: You are a Senior Data Consultant. Your tone is formal, objective, and precise.

        CONSTRAINTS:
        1. DATA VISIBILITY: You are provided with the ENTIRE dataset. Do not ignore rows at the bottom of the list (e.g., Linzess, Leeward, or PCV).
        2. NAME PARSING: Treat 'and', '&', and commas as separators. 'Sonu and Mahendar' means two distinct individuals.
        3. CO-OCCURRENCE LOGIC: If asked for projects where two people worked TOGETHER, you MUST verify that BOTH names appear in the 'Team Members' column for that SPECIFIC row. 
           - DO NOT list a project if only one person is mentioned.
           - DO NOT list a project based on memory; use ONLY the provided CSV text.
        4. ACCURACY: If a project like 'PCV HCP' has Mahendar but NOT Sonu, you MUST NOT include it in a joint list.
        5. OUTPUT: Provide the project names as a bulleted list. If no matches exist, state that clearly. Avoid conversational filler. Use table format whenever needed.
        """
        
        # Inject the full dataset context
        data_full_context = f"Full Dataset Content:\n{df.to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"DATASET:\n{data_full_context}\n\nUSER QUERY: {user_query}"}
            ],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")

if uploaded_file:
    try:
        # Load data with standard encoding
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # PRE-PROCESSING FIX: Ensure 'and' is treated as a comma for UI consistency and AI parsing
        df['Team Members'] = df['Team Members'].replace(to_replace=r',?\s+and\s+', value=', ', regex=True)
        df['Team Members'] = df['Team Members'].replace(to_replace=r'\s+&\s+', value=', ', regex=True)

        cols = df.columns.tolist()
        tab1, tab2, tab3 = st.tabs(["📄 Dataset Explorer", "📈 Visual Analytics", "💼 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=500)

        with tab2:
            st.subheader("Interactive Visualizations")
            c1, c2, c3 = st.columns(3)
            with c1: x_axis = st.selectbox("X-Axis", options=cols, index=0)
            with c2: y_axis = st.selectbox("Y-Axis", options=cols, index=min(1, len(cols)-1))
            with c3: hover = st.selectbox("Hover Info", options=cols, index=min(2, len(cols)-1))
            fig = px.bar(df, x=x_axis, y=y_axis, hover_data=[hover], color=y_axis, color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Consultancy Portal")
            query = st.text_input("Enter your business query:")
            if st.button("Generate Insight"):
                if not GROQ_API_KEY:
                    st.warning("Please enter your Groq API Key.")
                else:
                    with st.spinner("Analyzing full dataset..."):
                        answer = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Please upload a CSV file to begin.")
