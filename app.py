import streamlit as st
import pandas as pd
import requests

# ---------- Data functions (same as before) ----------
def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file, sheet_name=None)

def summarize_data(data):
    summary = ""
    for sheet_name, df in data.items():
        summary += f"## Sheet: {sheet_name}\n"
        summary += f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
        summary += f"- Columns: {', '.join(df.columns)}\n"
        summary += "### First 5 rows:\n"
        summary += df.head(5).to_markdown() + "\n\n"
    return summary

# ---------- LLM function using Groq API ----------
def ask_llm_api(data_summary, question):
    # Your Groq API key is stored in Streamlit secrets
    api_key = st.secrets["groq_api_key"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",  # or "llama3-8b-8192" for smaller
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": f"Data:\n{data_summary}\n\nQuestion: {question}"}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Excel Data Analyst", layout="wide")
st.title("📊 Excel Data Analyst with Llama 3")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    data = load_excel(uploaded_file)
    data_summary = summarize_data(data)

    # Data preview
    st.subheader("Data Preview")
    for sheet_name, df in data.items():
        with st.expander(f"Sheet: {sheet_name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
            st.dataframe(df)

    # Ask question
    st.subheader("Ask a Question")
    question = st.text_input("What would you like to know about this data?")

    if question:
        with st.spinner("Asking Llama 3..."):
            try:
                answer = ask_llm_api(data_summary, question)
                st.markdown("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload an Excel file to begin.")
