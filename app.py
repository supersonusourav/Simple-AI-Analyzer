import streamlit as st
import pandas as pd
from groq import Groq
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Simplified Analytics", page_icon="📊", layout="wide")

# Custom CSS for a clean, professional aesthetic
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #4F8BF9; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configuration")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    st.divider()
    st.caption("© 2026 Data Scientist: Mr. Sonu Sourav | Powered by: Groq 🚀")

# --- PROFESSIONAL AI FUNCTION ---
def analyze_with_groq(df, user_query, api_key):
    try:
        client = Groq(api_key=api_key)
        
        system_prompt = (
            "You are a Senior Data Consultant. Your goal is to provide accurate, "
            "professional, and concise insights based on the PROVIDED DATA. "
            "Scan the entire 'Team Members' column for names provided in the query. "
            "Maintain a formal business tone. Provide direct answers immediately."
        )
        
        # FIX: Pass the ENTIRE dataset to ensure rows like Leeward (Row 71) are visible
        data_full_context = f"Full Dataset:\n{df.to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Data Context:\n{data_full_context}\n\nQuery: {user_query}"}
            ],
            temperature=0  # For data accuracy
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # FIX: Normalize 'and' to commas to ensure AI counts individuals separately
        df['Team Members'] = df['Team Members'].replace(to_replace=r',?\s+and\s+', value=', ', regex=True)
        
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
                    with st.spinner("Analyzing dataset..."):
                        answer = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.info("👋 Please upload a CSV file to begin.")
