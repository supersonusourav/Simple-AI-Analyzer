import streamlit as st
import pandas as pd
from groq import Groq
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
st.title("📊 Simple AI Data Analyzer")

if uploaded_file:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["📄 Full Dataset", "📈 Visual Insights", "🤖 AI Analyst"])
        
        with tab1:
            # Show entire dataset
            st.dataframe(df, use_container_width=True, height=400)
        
        with tab2:
    st.subheader("Project Ratings & Team Members")
    
    # Identify key columns (Adjust 'Project Name', 'Rating', and 'Member' to match your CSV headers)
    # Assuming standard names, but we can make it flexible
    cols = df.columns.tolist()
    
    # Find the best columns to use
    name_col = st.selectbox("Select Project Name Column", options=cols)
    rating_col = st.selectbox("Select Rating Column", options=cols, index=min(1, len(cols)-1))
    member_col = st.selectbox("Select Member/Team Column", options=cols, index=min(2, len(cols)-1))

    if name_col and rating_col:
        # Plotly Bar Chart
        fig = px.bar(
            df, 
            x=name_col, 
            y=rating_col,
            hover_data=[member_col], # This adds members to the tooltip
            title=f"Ratings by {name_col}",
            template="plotly_white",
            color=rating_col, # Optional: Color bars based on the rating value
            color_continuous_scale="Viridis"
        )
        
        # To make the chart look better
        fig.update_layout(xaxis_tickangle=-45) # Tilt project names if they are long
        
        st.plotly_chart(fig, use_container_width=True)

        with tab3:
            query = st.text_input("Ask a question about your data:")
            if st.button("Run AI Analysis"):
                if not GROQ_API_KEY:
                    st.warning("Please enter your Groq API Key in the sidebar.")
                else:
                    with st.spinner("Groq is thinking..."):
                        result = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.info("### AI Response")
                        st.markdown(result)
                        
    except Exception as e:
        st.error(f"File Loading Error: {e}")
else:
    st.info("Please upload a CSV file to begin.")
