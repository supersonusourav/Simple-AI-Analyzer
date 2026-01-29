import streamlit as st
import pandas as pd
from groq import Groq
import plotly.express as px
import io

# --- PAGE SETUP ---
st.set_page_config(page_title="Simple AI Data Analyzer", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Config")
    # Placeholder for your GROQ Key
    GROQ_API_KEY = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.caption("© 2026 Sonu Sourav | Powered by Groq 🚀")

# --- AI ANALYSIS FUNCTION ---
def analyze_with_groq(df, user_query, api_key):
    try:
        client = Groq(api_key=api_key)
        
        data_context = f"Columns: {list(df.columns)}\nData:\n{df.to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise data tool. Answer with the final result only. No explanations, no steps, no introductory text. If the answer is a number, provide only the number."
                },
                {
                    "role": "user", 
                    "content": f"Dataset:\n{data_context}\n\nQuestion: {user_query}"
                }
            ],
            temperature=0  # Set to 0 for maximum consistency and precision
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
# --- MAIN APP ---
st.title("📊 Flexible AI Data Analyzer")

if uploaded_file:
    try:
        # Load the full dataset
        df = pd.read_csv(uploaded_file, encoding='latin1')
        cols = df.columns.tolist()

        tab1, tab2, tab3 = st.tabs(["📄 Data", "📈 Charts", "🤖 AI"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=400)

        with tab2:
            st.subheader("Dynamic Charting")
            # Side-by-side selection for a cleaner look
            c1, c2, c3 = st.columns(3)
            with c1:
                x_axis = st.selectbox("X-Axis (Labels)", options=cols, index=0)
            with c2:
                y_axis = st.selectbox("Y-Axis (Values)", options=cols, index=min(1, len(cols)-1))
            with c3:
                tooltip = st.selectbox("Tooltip (Team Members)", options=cols, index=min(2, len(cols)-1))

            if x_axis and y_axis:
                fig = px.bar(
                    df, x=x_axis, y=y_axis,
                    hover_data=[tooltip],
                    color=y_axis,
                    color_continuous_scale="RdYlGn", # Red-Yellow-Green scale
                    template="plotly_white"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # AI Logic here (from previous steps)
            st.write("Ask your data questions in the sidebar or here!")

    except Exception as e:
        st.error(f"Error: {e}")
    else:
    st.info("Please upload a CSV file to begin.")
