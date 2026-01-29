import streamlit as st
import pandas as pd
from groq import Groq
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="DataInsight AI", page_icon="📊", layout="wide")

# Custom CSS for a "Pro" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configuration")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    st.divider()
    st.caption("🚀 Optimized for 2026 Analysis")

# --- AI FUNCTION ---
def analyze_with_groq(df, user_query, api_key):
    try:
        client = Groq(api_key=api_key)
        # We send a structural summary + head to save tokens and stay precise
        data_context = f"Columns: {list(df.columns)}\nSample Data:\n{df.head(10).to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data tool. Answer with the final result only. No conversational filler."},
                {"role": "user", "content": f"Context:\n{data_context}\n\nQuestion: {user_query}"}
            ],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- MAIN APP INTERFACE ---
st.title("📊 DataInsight AI")
st.markdown("Detailed visual analysis and AI-powered querying in one place.")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        cols = df.columns.tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(df))
        c2.metric("Total Columns", len(cols))
        c3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        tab1, tab2, tab3 = st.tabs(["📄 Explore Data", "📈 Visual Analytics", "🤖 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=450)

        with tab2:
            st.subheader("Interactive Chart Builder")
            col_a, col_b, col_c = st.columns(3)
            with col_a: x_axis = st.selectbox("X-Axis", options=cols, index=0)
            with col_b: y_axis = st.selectbox("Y-Axis", options=cols, index=min(1, len(cols)-1))
            with col_c: tooltip = st.selectbox("Tooltip Info", options=cols, index=min(2, len(cols)-1))

            fig = px.bar(df, x=x_axis, y=y_axis, hover_data=[tooltip], 
                         color=y_axis, color_continuous_scale="Viridis", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Query your Data")
            query = st.text_input("Ask anything (e.g., 'What is the highest rating?')")
            if st.button("Ask AI"):
                if not GROQ_API_KEY:
                    st.warning("Please provide an API key in the sidebar.")
                else:
                    with st.spinner("Analyzing..."):
                        answer = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.success(answer)

    except Exception as e:
        st.error(f"Critical Error: {e}")
else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to start your analysis.")
