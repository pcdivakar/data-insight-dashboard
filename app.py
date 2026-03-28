import streamlit as st
import pandas as pd
import plotly.express as px
import io
from fpdf import FPDF
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFaceEndpoint

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Data Dashboard", page_icon="📊", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

# --- HELPER FUNCTIONS ---
def export_to_pdf(chat_history):
    """Exports chat insights to a PDF format."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Data Insights Report", ln=True, align='C')
    pdf.ln(10)
    
    for item in chat_history:
        # User Question
        pdf.set_font("Arial", 'B', size=11)
        pdf.multi_cell(0, 10, txt=f"Q: {item['question']}")
        
        # LLM Answer
        pdf.set_font("Arial", size=11)
        # Handle encoding issues for PDF
        answer = str(item['answer']).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=f"A: {answer}")
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

def export_to_excel(dataframes_dict):
    """Exports multiple dataframes to a single Excel file with multiple sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in dataframes_dict.items():
            # Excel sheet names can't be longer than 31 characters
            sheet_name = name[:31] 
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    hf_api_token = st.text_input("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", type="password", help="Get this from your Hugging Face account settings.")
    st.markdown("---")
    st.header("Upload Data")
    uploaded_files = st.file_uploader("Upload up to 3 Excel templates", type=["xlsx", "xls"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Please upload a maximum of 3 files.")
        else:
            for file in uploaded_files:
                if file.name not in st.session_state.dataframes:
                    df = pd.read_excel(file)
                    st.session_state.dataframes[file.name] = df
            st.success(f"{len(uploaded_files)} file(s) loaded successfully!")

# --- MAIN APP LAYOUT ---
st.title("📊 AI-Powered Data Dashboard & Assistant")

if not st.session_state.dataframes:
    st.info("👈 Please upload your Excel files in the sidebar to get started.")
else:
    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Dashboard", "🤖 Chat & Insights"])

    # --- TAB 1: DATA PREVIEW ---
    with tab1:
        st.header("Uploaded Datasets")
        for file_name, df in st.session_state.dataframes.items():
            st.subheader(file_name)
            st.dataframe(df.head()) # Show top rows to save memory

    # --- TAB 2: DASHBOARD ---
    with tab2:
        st.header("Interactive Visualizations")
        selected_file = st.selectbox("Select Dataset for Visualization", list(st.session_state.dataframes.keys()))
        df_selected = st.session_state.dataframes[selected_file]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
        with col2:
            x_axis = st.selectbox("X-Axis", df_selected.columns)
        with col3:
            y_axis = st.selectbox("Y-Axis", df_selected.columns)
            
        if st.button("Generate Chart"):
            if chart_type == "Bar Chart":
                fig = px.bar(df_selected, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
            elif chart_type == "Line Chart":
                fig = px.line(df_selected, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            else:
                fig = px.scatter(df_selected, x=x_axis, y=y_axis, title=f"Scatter of {x_axis} vs {y_axis}")
                
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: CHAT & INSIGHTS (LLM Integration) ---
    with tab3:
        st.header("Ask Data Questions")
        
        if not hf_api_token:
            st.warning("Please enter your Hugging Face API Token in the sidebar to use the AI Assistant.")
        else:
            try:
                # Initialize LLM using a powerful open-source model like Mixtral
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    huggingfacehub_api_token=hf_api_token,
                    temperature=0.1,
                    max_new_tokens=512
                )
                
                # Extract dataframes into a list for the agent
                df_list = list(st.session_state.dataframes.values())
                
                # Initialize Pandas Dataframe Agent
                # allow_dangerous_code is set to True because the agent runs python code locally to calculate answers
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df_list, 
                    verbose=True, 
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                )
                
                user_question = st.text_input("Ask a question about your uploaded Excel files:")
                
                if st.button("Get Insight"):
                    with st.spinner("AI is analyzing the data..."):
                        try:
                            # Run the agent
                            response = agent.invoke(user_question)
                            answer = response.get("output", "I could not find an answer.")
                            
                            # Save to chat history
                            st.session_state.chat_history.append({"question": user_question, "answer": answer})
                        except Exception as e:
                            st.error(f"Error analyzing data: {str(e)}")
                            
                # Display Chat History
                if st.session_state.chat_history:
                    st.markdown("### Insight History")
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.chat_message("user"):
                            st.write(chat["question"])
                        with st.chat_message("assistant"):
                            st.write(chat["answer"])
                            
            except Exception as e:
                st.error(f"Failed to initialize AI. Check your API token or model availability. Error: {str(e)}")

    # --- EXPORT SECTION ---
    st.markdown("---")
    st.header("📥 Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # Export Data to Excel
        if st.session_state.dataframes:
            excel_data = export_to_excel(st.session_state.dataframes)
            st.download_button(
                label="📥 Download Combined Excel Data",
                data=excel_data,
                file_name="combined_dashboard_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    with export_col2:
        # Export Chat/Insights to PDF
        if st.session_state.chat_history:
            pdf_data = export_to_pdf(st.session_state.chat_history)
            st.download_button(
                label="📥 Download Insights Report (PDF)",
                data=pdf_data,
                file_name="ai_data_insights.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Ask questions in the 'Chat & Insights' tab to generate a PDF report.")