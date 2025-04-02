import streamlit as st
import pandas as pd
import os
import warnings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration
warnings.filterwarnings("ignore")
st.set_page_config(page_icon="📊",page_title="Data Analysis Agent", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    google_api_key = st.text_input("Google API Key", type="password", help="Enter your Google Generative AI API key")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    st.markdown("---")
    st.markdown("### Agent Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, help="Controls randomness of responses")
    verbose_mode = st.checkbox("Verbose Mode", value=True, help="Show detailed agent execution")
    
    st.markdown("---")
    st.markdown("### Sample Queries")
    st.code("What is the number of records?")
    st.code("Show summary statistics")
    st.code("Plot distribution of [column_name]")
    st.code("Find correlations between variables")

# Main content area
st.title("📊 Data Analysis Agent")
st.markdown("Ask questions about your data and get insights powered by Gemini")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load data if file is uploaded
if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")
        
        # Show data preview
        with st.expander("View Data Preview"):
            st.dataframe(st.session_state.df.head())
            
        # Initialize agent if API key is provided
        if google_api_key:
            try:
                os.environ["GOOGLE_API_KEY"] = google_api_key
                chat_model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=google_api_key,
                    temperature=temperature
                )
                
                st.session_state.agent = create_pandas_dataframe_agent(
                    llm=chat_model,
                    df=st.session_state.df,
                    verbose=verbose_mode,
                    allow_dangerous_code=True
                )
                st.success("Agent initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Chat interface
if st.session_state.df is not None and st.session_state.agent is not None:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    query = st.chat_input("Ask a question about your data...")
    
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        try:
            # Get agent response
            with st.spinner("Analyzing..."):
                response = st.session_state.agent.run(query)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
else:
    st.info("Please upload a CSV file and provide your Google API key to get started.")