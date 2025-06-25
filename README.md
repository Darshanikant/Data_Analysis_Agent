# Data_Analysis_Agent
A Streamlit-based AI-powered data analysis tool using Google Gemini API and LangChain. Users can upload datasets, ask questions, and get AI-driven insights.



## Overview

**Data Analysis Agent** is a Streamlit-based web application that enables users to interact with their datasets using natural language queries. The application utilizes Google Gemini AI models for intelligent data analysis and visualization. Users can upload a dataset, ask questions, and receive AI-driven insights in a user-friendly interface.

## Features
- **User Authentication for API Key**: Users must provide their Google Gemini API key to interact with the agent.
- **Dataset Upload**: Users can upload CSV files for analysis.
- **AI-Powered Data Insights**: The application utilizes the LangChain framework with Gemini AI to answer queries and generate statistical summaries.
- **Multiple AI Model Testing**: The project incorporates implementations from ChatGPT, DeepSeek, Claude, and Grok for comparative analysis.
- **Interactive UI with Streamlit**: A clean and simple UI enables users to seamlessly interact with the data analysis agent.

## Tech Stack
- **Frontend**: Streamlit (for UI)
- **Backend**: Python (FastAPI/Flask for API handling if required)
- **AI Integration**: Google Gemini AI (via LangChain)
- **Data Handling**: Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Support for UI**: Implementations tested using ChatGPT, DeepSeek, Claude, and Grok AI models

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8) and the required libraries:

```bash
pip install requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/Darshanikant/Data_Analysis_Agent.git
cd Data_Analysis_Agent
```

### Run the Application
```bash
streamlit run agent_app_DS.py
```

## Usage
1. **Enter API Key**: Provide your Google Gemini API key.
2. **Upload Dataset**: Upload a CSV file for analysis.
3. **Ask Queries**: Type your data-related questions, and the AI agent will respond with insights.

## Example Queries
- "What is the number of records in the dataset?"
- "Provide summary statistics for all numeric columns."
- "Visualize the distribution of the main feature."


## Contributors
Developed by **Darshanikanta** with insights from multiple AI models including ChatGPT, DeepSeek, Claude, and Grok.



