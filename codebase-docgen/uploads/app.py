# imports
import streamlit as st
import os, tempfile
import plotly.express as px
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.agents import create_pandas_dataframe_agent
import asyncio
from typing import Union, Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import logging


st.set_page_config(page_title="OpenAI Data Analyzer", layout="wide")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



@st.cache_resource()
def retriever_func(uploaded_file):
    if uploaded_file :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200, 
                        add_start_index=True
                        )
        all_splits = text_splitter.split_documents(data)

        
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    if not uploaded_file:
        st.info("Please upload CSV documents to continue.")
        st.stop()
    return retriever, vectorstore

def create_safe_agent(llm, df, **kwargs) -> Any:
    """Safely create a pandas dataframe agent with error handling."""
    try:
        return create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            **kwargs
        )
    except Exception as e:
        logging.error(f"Error creating agent: {str(e)}")
        raise Exception(f"Failed to initialize analysis agent: {str(e)}")


def chat(temperature, model_name):
    st.write("# Talk to CSV")
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    retriever, vectorstore = retriever_func(uploaded_file)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming=True)
        
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    store = {}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Use the following pieces of context to answer the question at the end.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context}""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    runnable = prompt | llm
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    async def chat_message():
        if prompt := st.chat_input():
            if not user_api_key: 
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            contextt = vectorstore.similarity_search(prompt, k=6)
            context = "\n\n".join(doc.page_content for doc in contextt)
            #msg = 
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                text_chunk = ""
                async for chunk in with_message_history.astream(
                        {"context": context, "input": prompt},
                        config={"configurable": {"session_id": "abc123"}},
                    ):
                    text_chunk += chunk.content
                    message_placeholder.markdown(text_chunk)
                    #st.chat_message("assistant").write(text_chunk)
                st.session_state.messages.append({"role": "assistant", "content": text_chunk})
        if reset:
            st.session_state["messages"] = []
    asyncio.run(chat_message())


def summary(model_name, temperature, top_p):
    st.write("# Summary of CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        # encoding = cp1252
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap=100)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)

        os.remove(tmp_file_path)
        gen_sum = st.button("Generate Summary")
        if gen_sum:
            # Initialize the OpenAI module, load and run the summarize chain
            llm = ChatOpenAI(model_name=model_name, temperature=temperature)
            chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",

                return_intermediate_steps=True,
                input_key="input_documents",
                output_key="output_text",
            )
            result = chain({"input_documents": texts}, return_only_outputs=True)

            st.success(result["output_text"])


def analyze(temperature, model_name):
    st.write("# Analyze CSV")
    
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        df = pd.read_csv(tmp_file_path)
        
        # Create the agent with better error handling
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(model=model_name, temperature=temperature),
            df,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True,  # Add this parameter
            max_iterations=3,  # Add retry limit
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you analyze this data?"}]
            
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(placeholder="Ask me anything about the data..."):
            if not user_api_key: 
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            try:
                with st.spinner("Analyzing..."):
                    # Wrap the agent call in better error handling
                    try:
                        response = agent.run({
                            "input": prompt,
                            "chat_history": st.session_state.messages,
                            "handle_parsing_errors": True
                        })
                        
                        # If response is a dictionary or complex object, convert to string
                        if not isinstance(response, str):
                            response = str(response)
                        
                        # Try to enhance the response with visualization if applicable
                        try:
                            # Check if the response contains numeric results
                            if any(char.isdigit() for char in response):
                                st.write("Response:", response)
                                
                                # Add visualization options
                                st.write("Would you like to visualize any numeric columns?")
                                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                                if not numeric_cols.empty:
                                    selected_col = st.selectbox(
                                        "Select column to visualize:",
                                        numeric_cols
                                    )
                                    
                                    chart_type = st.selectbox(
                                        "Select visualization type:",
                                        ["Histogram", "Box Plot", "Line Plot"]
                                    )
                                    
                                    if chart_type == "Histogram":
                                        fig = px.histogram(df, x=selected_col)
                                        st.plotly_chart(fig)
                                    elif chart_type == "Box Plot":
                                        fig = px.box(df, y=selected_col)
                                        st.plotly_chart(fig)
                                    elif chart_type == "Line Plot":
                                        fig = px.line(df, y=selected_col)
                                        st.plotly_chart(fig)
                            else:
                                st.write(response)
                                
                        except Exception as viz_error:
                            st.write(response)  # Fall back to just showing the response
                            logging.error(f"Visualization error: {str(viz_error)}")
                            
                    except Exception as agent_error:
                        # Handle specific agent errors
                        error_msg = str(agent_error)
                        if "parsing" in error_msg.lower():
                            response = "I understood your question but had trouble formatting the response. Here's the raw analysis: " + error_msg
                        else:
                            response = "I encountered an error analyzing your question. Please try rephrasing it."
                        logging.error(f"Agent error: {error_msg}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logging.error(f"Analysis error: {str(e)}")
        
        if reset:
            st.session_state["messages"] = []


async def preprocess_data(df: pd.DataFrame, file_name: str):
    """Preprocess data using OpenAI."""
    try:
        logging.info(f"Starting preprocessing for {file_name}")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Get data info for analysis
        data_info = {
            'filename': file_name,
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        logging.info("Getting column renaming suggestions")
        # Get column renaming suggestions
        rename_prompt = f"""Analyze these columns and suggest business-friendly names:
        Current columns and their info: {json.dumps(data_info, indent=2)}
        Return ONLY a Python dictionary mapping current names to new names.
        Example: {{"old_name": "new_business_friendly_name"}}"""
        
        rename_response = llm.predict(rename_prompt)
        try:
            new_names = eval(rename_response)
            logging.info(f"Column renaming suggestions: {new_names}")
        except Exception as e:
            logging.error(f"Error parsing rename suggestions: {str(e)}")
            new_names = {}
        
        # Rename columns
        if new_names:
            df = df.rename(columns=new_names)
            logging.info("Columns renamed successfully")
        
        # Get cleaning suggestions
        logging.info("Getting cleaning suggestions")
        clean_prompt = f"""Analyze this dataset and suggest cleaning operations:
        Data info: {json.dumps(data_info, indent=2)}
        
        For each column suggest:
        1. How to handle missing values
        2. Data type conversions needed
        3. Inconsistency corrections
        
        Return as Python dictionary: {{"column_name": ["operation1", "operation2"]}}"""
        
        clean_response = llm.predict(clean_prompt)
        try:
            cleaning_ops = eval(clean_response)
            logging.info(f"Cleaning operations suggested: {cleaning_ops}")
        except Exception as e:
            logging.error(f"Error parsing cleaning suggestions: {str(e)}")
            cleaning_ops = {}
        
        # Apply cleaning operations
        cleaning_log = {}
        for col, operations in cleaning_ops.items():
            if col in df.columns:
                cleaning_log[col] = []
                for op in operations:
                    try:
                        if 'convert_numeric' in op.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            cleaning_log[col].append("Converted to numeric")
                            print("converted to numeric")
                        elif 'convert_datetime' in op.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            cleaning_log[col].append("Converted to datetime")
                        elif 'fill_mean' in op.lower() and df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].mean())
                            cleaning_log[col].append("Filled missing values with mean")
                        elif 'fill_mode' in op.lower():
                            df[col] = df[col].fillna(df[col].mode()[0])
                            cleaning_log[col].append("Filled missing values with mode")
                        elif 'standardize_text' in op.lower():
                            df[col] = df[col].str.lower().str.strip()
                            cleaning_log[col].append("Standardized text")
                    except Exception as e:
                        error_msg = f"Error applying {op} to {col}: {str(e)}"
                        logging.error(error_msg)
                        cleaning_log[col].append(error_msg)
        
        logging.info("Preprocessing completed successfully")
        return df, new_names, cleaning_ops
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        return df, {}, {}

# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>OpenAI Data Analyzer</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    global user_api_key
    if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
        user_api_key = os.environ["OPENAI_API_KEY"]
        st.success("API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password", key="openai_api_key"
        )
        if user_api_key:
            st.sidebar.success("API key successfully loaded", icon="🚀")

    os.environ["OPENAI_API_KEY"] = user_api_key

    

    # Model settings
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k","gpt-3.5-turbo-16k","gpt-4-1106-preview"]
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.01)

    uploaded_files = st.file_uploader("Upload your CSV files", type='csv', accept_multiple_files=True)

    if uploaded_files:
        processed_dfs = {}
        
        # Process each file
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            
            with st.spinner("Reading and preprocessing data..."):
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                # Preprocess data
                df_cleaned, new_names, cleaning_ops = asyncio.run(preprocess_data(df, uploaded_file.name))
                processed_dfs[uploaded_file.name] = df_cleaned
                
                # Show preprocessing results
                with st.expander(f"Preprocessing Results for {uploaded_file.name}", expanded=True):
                    st.subheader("Column Name Changes")
                    for old, new in new_names.items():
                        st.write(f"- {old} → {new}")
                    
                    st.subheader("Cleaning Operations")
                    for col, ops in cleaning_ops.items():
                        st.write(f"**{col}**:")
                        for op in ops:
                            st.write(f"- {op}")
                    
                    st.subheader("Data Preview")
                    st.dataframe(df_cleaned.head())

        # After preprocessing, show analysis options
        st.markdown("---")
        analysis_type = st.radio(
            "Choose Analysis Type",
            ["Summarize Data", "Analyze Data"]
        )

        if analysis_type == "Summarize Data":
            for file_name, df in processed_dfs.items():
                with st.expander(f"Summary for {file_name}"):
                    llm = ChatOpenAI(model=model_name, temperature=temperature)
                    
                    # Get summary
                    summary_prompt = f"""Provide a comprehensive summary of this dataset:
                    Columns: {list(df.columns)}
                    Sample Data: {df.head(3).to_dict('records')}
                    Stats: {df.describe().to_dict()}
                    
                    Include:
                    1. Overview of the data
                    2. Key statistics
                    3. Important patterns
                    4. Data quality observations"""
                    
                    summary = llm.predict(summary_prompt)
                    st.write(summary)

        elif analysis_type == "Analyze Data":
            st.warning("""Note: This analysis feature executes Python code to answer your questions. 
                While it's designed to be safe, please be mindful of the queries you submit.""")
            selected_file = st.selectbox("Select file to analyze", list(processed_dfs.keys()))
            df = processed_dfs[selected_file]
            
            query = st.text_area("Enter your analysis question:")
            if st.button("Analyze"):
                agent = create_pandas_dataframe_agent(
                    ChatOpenAI(temperature=temperature, model=model_name),
                    df,
                    verbose=True,
                    allow_dangerous_code=True  # Add this parameter
                )
                
                with st.spinner("Analyzing..."):
                    try:
                        response = agent.run(query)
                        st.write(response)
                        
                        # Add visualizations for numeric results
                        if any(df[col].dtype in ['int64', 'float64'] for col in df.columns):
                            st.subheader("Visualization")
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            selected_col = st.selectbox("Select column to visualize:", numeric_cols)
                            
                            import plotly.express as px
                            fig = px.histogram(df, x=selected_col)
                            st.plotly_chart(fig)
                            
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        logging.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()