import streamlit as st
import pandas as pd
from groq import Groq
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Simplified Analytics", page_icon="📊", layout="wide")

# --- SIDEBAR ---
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
        
        # This prompt forces the AI to act as a logic gate rather than a narrator
        system_prompt = """
        CRITICAL ROLE: You are a Deterministic Data Auditor. Your output must be 100% verified against the provided CSV.

        RULES TO PREVENT HALLUCINATION:
        1. NO ASSUMPTIONS: If a name is not explicitly in the 'Team Members' cell for a specific project, that project DOES NOT exist for that person.
        2. CO-OCCURRENCE CHECK: When asked about two people (e.g., Sonu AND Mahendar) working together:
           - Scan every row individually.
           - A project is ONLY a match if BOTH names appear in the same 'Team Members' string.
           - If Sonu is in row A and Mahendar is in row B, they did NOT work together on those projects.
        3. DATA SCOPE: You have access to the FULL dataset. Do not stop scanning after 10 rows. Ensure you check projects like 'Linzess', 'Leeward', and 'PCV' found at the end of the file.
        4. PARSING: Treat "and", "&", and "/" as delimiters. 
        5. VERIFICATION: Before answering, double-check your list. For example, in 'PCV HCP', is Sonu actually listed? (Answer: No). Therefore, do not list it.

        OUTPUT FORMAT: Provide a clear bulleted list. If no projects match the criteria, say 'No joint projects found in the record.'
        """
        
        # Injecting full CSV content as a string
        data_full_context = f"DATAFRAME CONTENT:\n{df.to_csv(index=False)}"
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"DATASET:\n{data_full_context}\n\nUSER QUERY: {user_query}"}
            ],
            temperature=0  # Minimum randomness to prevent "creative" hallucinations
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# --- MAIN INTERFACE ---
st.title("📊 Simplified Analytics")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='latin1')
        
        # Pre-processing to clean up separators for the AI
        df['Team Members'] = df['Team Members'].replace(to_replace=[r',?\s+and\s+', r'\s+&\s+'], value=', ', regex=True)

        tab1, tab2, tab3 = st.tabs(["📄 Dataset Explorer", "📈 Visual Analytics", "💼 AI Consultant"])

        with tab1:
            st.dataframe(df, use_container_width=True, height=500)

        with tab3:
            st.subheader("Consultancy Portal")
            query = st.text_input("Enter your query (e.g., 'Projects where Sonu and Mahendar worked together'):")
            if st.button("Generate Insight"):
                if not GROQ_API_KEY:
                    st.warning("Please enter your Groq API Key.")
                else:
                    with st.spinner("Auditing dataset for accurate matches..."):
                        answer = analyze_with_groq(df, query, GROQ_API_KEY)
                        st.info(f"**Consultant Response:**\n\n{answer}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Please upload a CSV file to begin.")
