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
    # Your Copyright
    st.caption("© 2026 Data Scientist: Mr. Sonu Sourav | Powered by: Groq 🚀")

# --- PROFESSIONAL AI FUNCTION ---
def analyze_with_groq(df, user_query, api_key):
    try:
        client = Groq(api_key=api_key)
        
        # Professional System Prompt
        system_prompt = (
            "You are a Senior Data Consultant. Your goal is to provide accurate, "
            "professional, and concise insights based on the provided data. "
            "Maintain a formal business tone. If the user asks for a specific number, "
            "provide that number immediately, followed by a very brief professional context if necessary. "
            "Avoid slang, introductory filler, or conversational chatter."
        )

        data_full_context = f"Structure: {list(df.columns)}\nFull Data Content:\n{df.to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Data Context:\n{data_full_context}\n\nClient Query: {user_query}"}
            ],
            temperature=0
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")
st.markdown("Automated Visual Analytics and Professional Data Consultancy.")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        df = df.replace(to_replace=r'\s+and\s+', value=', ', regex=True)
        
        cols = df.columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📄 Dataset Explorer", "📈 Visual Analytics", "💼 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=500)

        with tab2:
            st.subheader("Interactive Visualizations")
            c1, c2, c3 = st.columns(3)
            with c1: x_axis = st.selectbox("X-Axis (Labels)", options=cols, index=0)
            with c2: y_axis = st.selectbox("Y-Axis (Values)", options=cols, index=min(1, len(cols)-1))
            with c3: hover = st.selectbox("Hover Information", options=cols, index=min(2, len(cols)-1))

            fig = px.bar(df, x=x_axis, y=y_axis, hover_data=[hover], 
                         color=y_axis, color_continuous_scale="Blues", template="plotly_white")
            fig.update_layout(xaxis_tickangle=-45, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Consultancy Portal")
            query = st.text_input("Enter your business query regarding this dataset:")
            if st.button("Generate Insight"):
                if not GROQ_API_KEY:
                    st.warning("Action Required: Please enter your Groq API Key in the sidebar.")
                else:
                    with st.spinner("Processing analysis..."):
                        answer = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
else:
    st.info("👋 Please upload a CSV file in the sidebar to initialize the analysis engine.")
