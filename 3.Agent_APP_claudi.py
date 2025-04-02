import streamlit as st
import pandas as pd
import os
import warnings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'history' not in st.session_state:
    st.session_state.history = []

# App title and description
st.title("🤖 Data Analysis Assistant")
st.markdown("Powered by Gemini and LangChain")

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your API key from Google AI Studio")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Shape: {df.shape}")
            
            # Show data preview in sidebar
            st.subheader("Data Preview")
            st.dataframe(df.head(3), use_container_width=True)
            
            st.session_state.df = df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Initialize button
    if st.button("Initialize Agent", disabled=(not api_key or st.session_state.df is None)):
        with st.spinner("Initializing the agent..."):
            try:
                # Create the model
                chat_model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=api_key,
                    temperature=0.1
                )
                
                # Create the agent - ADDED allow_dangerous_code=True
                agent = create_pandas_dataframe_agent(
                    llm=chat_model,
                    df=st.session_state.df,
                    verbose=True,
                    allow_dangerous_code=True  # Added this parameter
                )
                
                st.session_state.api_key = api_key
                st.session_state.agent = agent
                st.session_state.initialized = True
                st.success("Agent successfully initialized!")
            except Exception as e:
                st.error(f"Error initializing agent: {e}")
                st.info("Please check your API key and try again.")

# Main content area
if st.session_state.initialized:
    st.header("Ask questions about your data")
    
    # User query input
    query = st.text_area("Enter your query", height=100, 
                        placeholder="Examples:\n- What is the shape of the dataset?\n- Show summary statistics for numeric columns\n- What are the correlations between features?\n- Plot the distribution of [column]")
    
    run_query = st.button("Run Query")
    
    if run_query and query:
        with st.spinner("Processing your query..."):
            try:
                # Add query to history
                st.session_state.history.append({"query": query, "response": None})
                
                # Run the agent
                response = st.session_state.agent.run(query)
                
                # Update the response in history
                st.session_state.history[-1]["response"] = response
                
                st.success("Query processed successfully!")
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.session_state.history[-1]["response"] = f"Error: {str(e)}"
    
    # Display history
    if st.session_state.history:
        st.header("Query History")
        
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history) - i}: {item['query'][:50]}...", expanded=(i == 0)):
                st.markdown("**Query:**")
                st.markdown(f"```\n{item['query']}\n```")
                
                st.markdown("**Response:**")
                if item['response']:
                    st.markdown(item['response'])
                else:
                    st.info("Processing...")
else:
    # Instructions when not initialized
    st.info("👈 Please enter your Gemini API key and upload a CSV file in the sidebar, then initialize the agent to get started.")
    
    st.header("How to use this app")
    st.markdown("""
    1. Enter your Gemini API key in the sidebar
    2. Upload a CSV dataset
    3. Click "Initialize Agent" to set up the system
    4. Ask natural language questions about your data
    5. View the responses and build on your analysis
    
    **Example queries:**
    - What is the shape of the dataset?
    - Provide summary statistics for all numeric columns
    - What are the top 5 most correlated features?
    - Plot the distribution of [column_name]
    - Are there any missing values in the dataset?
    - What are the unique values in [column_name]?
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Note: Your API key is not stored permanently.")