import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Function to create the LangChain agent
def create_agent(google_api_key, df):
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Using the latest model for better capabilities
        google_api_key=google_api_key,
          # Lower temperature for more deterministic responses
    )
    
    agent = create_pandas_dataframe_agent(
        llm=chat_model,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )
    
    return agent

# Streamlit UI
st.title("Gemini AI Agent for Data Analysis")

# Step 1: User inputs Gemini API key
google_api_key = st.text_input("Enter your Google Gemini API Key", type="password")

# Step 2: Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset into a pandas dataframe
    df = pd.read_csv(uploaded_file)
    st.write("Dataset successfully loaded!")
    st.write(df.head())  # Displaying the first few rows of the dataset
    
    # Step 3: Initialize the agent with the dataset
    if google_api_key:
        agent = create_agent(google_api_key, df)
        
        # Step 4: User asks a question
        query = st.text_input("Ask a question about the dataset")
        
        if query:
            # Get the response from the agent
            response = agent.run(query)
            st.write("Answer:", response)
    else:
        st.warning("Please enter your Gemini API key to initialize the agent.")
else:
    st.info("Please upload a dataset to proceed.")
