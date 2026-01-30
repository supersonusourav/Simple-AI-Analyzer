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
        CRITICAL ROLE: You are a Deterministic Data Auditor or a Senior Data Analyst. Your output must be 100% verified against the provided CSV.

        RULES TO PREVENT HALLUCINATION:
        1. NO ASSUMPTIONS: If a name is not explicitly in the 'Team Members' cell for a specific project, that project DOES NOT exist for that person.
        2. CO-OCCURRENCE CHECK: When asked about two people working together:
           - Scan every row individually.
           - A project is ONLY a match if the listed names in the query appear in the same 'Team Members' string.
           - If Sonu is in row A and Mahendar is in row B, they did NOT work together on those projects.
           - Check the sheet in the similar logical operator fashion as asked in the query.
        3. DATA SCOPE: You have access to the FULL dataset. Scan from top to bottom of the sheet. Do not skip any part of the sheet. Ensure you check rows till the end of the file of the sheet.
        4. PARSING: Treat "and", "&", and "/" as delimiters. 
        5. VERIFICATION: Before answering, double-check your list for each query.

        OUTPUT FORMAT: Provide a clear bulleted list. If no projects match the criteria, say 'No joint projects found in the record.'
        STRICT PROTOCOL:
        1. SEPARATION: Treat 'Team Members' as a fixed list. 'And', '&', and commas are separators.
        2. MANDATORY CHECK: When checking for two people (e.g., Rajesh and Ramya):
           - For EVERY project, you must ask: "Is Name A in this cell?" AND "Is Name B in this cell?"
           - If the answer to BOTH is not 'YES', you MUST exclude that project.
        3. NO ASSOCIATIONS: Do not assume that because people work in the same 'Category' or 'Complexity' that they worked together. 
        4. VERIFICATION STEP: Before finalizing your list, look at the 'Team Members' of your chosen projects one last time. If you see 'Ramya' but not 'Rajesh', REMOVE it immediately.
        5. HONESTY: If only 2 projects match, only list 2. Do not try to find a third one to be 'helpful'.
        VERIFICATION PROTOCOL:
        1. IDENTIFY NAMES: Extract the specific names mentioned in the user query.
        2. ROW-BY-ROW SCAN: For every project in the dataset, check if ALL identified names are present in the 'Team Members' string.
        3. CHAIN OF THOUGHT: You MUST follow these steps mentally:
           - "Project X: Does it have [Name 1]? Yes/No. Does it have [Name 2]? Yes/No."
           - Only if BOTH are 'Yes', add to the final list.
        4. ANTI-HALLUCINATION RULES: 
           - Do not assume. For example, if 'Rajesh' is not in the 'PCV HCP' row, DO NOT list it, even if 'Ramya' is there.
           - Ignore previous conversational context; look ONLY at the CSV data provided now.
        5. OUTPUT FORMAT: 
           - Start with: "Based on a row-by-row audit..."
           - List only the valid projects. 
           - If a project was a 'near miss' (only one person present), DO NOT mention it.
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
