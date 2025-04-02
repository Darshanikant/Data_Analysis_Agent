import streamlit as st
import pandas as pd
import warnings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit app configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide"
)

def initialize_agent(api_key, df):
    """Initialize the LangChain agent with the provided API key and dataframe"""
    try:
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        
        agent = create_pandas_dataframe_agent(
            llm=chat_model,
            df=df,
            verbose=True,
            allow_dangerous_code=True
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def main():
    # Title and description
    st.title("📊 Data Analysis Agent")
    st.markdown("""
    Upload your dataset and ask questions about your data using natural language.
    Powered by Google's Gemini AI and LangChain.
    """)

    # API Key input
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        st.session_state.df = None

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter your Google Gemini API Key", type="password")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None and api_key:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                # Initialize agent
                st.session_state.agent = initialize_agent(api_key, df)
                st.success("Agent initialized successfully!")
                
                # Display basic info about the dataset
                st.subheader("Dataset Info")
                st.write(f"Number of rows: {df.shape[0]}")
                st.write(f"Number of columns: {df.shape[1]}")
                st.write("Columns:", list(df.columns))
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Main content area
    if st.session_state.agent is not None and st.session_state.df is not None:
        st.header("Ask Your Questions")
        
        # Query input
        query = st.text_area("Enter your question about the dataset", 
                           height=100,
                           placeholder="e.g., 'What is the average value of column X?' or 'Show me summary statistics'")
        
        if st.button("Analyze"):
            if query:
                with st.spinner("Processing your query..."):
                    try:
                        response = st.session_state.agent.run(query)
                        st.subheader("Response")
                        st.write(response)
                        
                        # Display the dataframe if requested
                        if "show" in query.lower() or "display" in query.lower():
                            st.subheader("Dataset Preview")
                            st.dataframe(st.session_state.df.head())
                            
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a query first!")

        # Option to view raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data Preview")
            st.dataframe(st.session_state.df)

    else:
        st.info("Please provide an API key and upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()