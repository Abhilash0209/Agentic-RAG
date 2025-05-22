import os
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, tool
from llama_index.llms.google_genai import GoogleGenAI
from langchain.agents.agent_types import AgentType
import wikipedia
import warnings
from langchain_core._api import LangChainDeprecationWarning

# Suppress specific LangChain deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Load environment variables
load_dotenv()

# Fix Streamlit/Torch file watcher warning
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = ".torch."

# Load the Google API key
api_key = os.getenv("GOOGLE_API_KEY")

st.title("Agentic RAG with PDF Search and Wikipedia")

# Upload PDF
uploaded_file = st.file_uploader(" Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save PDF locally
    filename = f"{uuid.uuid4()}.pdf"
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load and split PDF into documents
        loader = PyPDFLoader(filename)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs)

        # Embed and create FAISS vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectors = FAISS.from_documents(final_documents, embeddings)
        retriever = vectors.as_retriever(search_kwargs={"k": 3})

        # Load LLM and create agent
        from langchain_google_genai import ChatGoogleGenerativeAI
       
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model="gemini-1.5-flash", 
            temperature=0.7
        )
        # Tool 1: PDF Search
        @tool
        def pdf_search(query: str) -> str:
            """
            Use this tool to answer ANY questions based on the uploaded PDF document. 
            The PDF has already been processed and includes information about legal, procedural, or regulatory content.
            
            """
            results = retriever.get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in results])

        # Tool 2: PDF Summary
        @tool
        def summarize_pdf(dummy_input: str) -> str:
            """Return a brief summary of the uploaded PDF."""
            try:
                all_text = "\n".join([doc.page_content for doc in final_documents[:5]])
                prompt = f"Summarize the following content:\n\n{all_text}"
                return llm.invoke(prompt)
            except Exception as e:
                return f"Error generating summary: {e}"

        # Configure Wikipedia to English
        wikipedia.set_lang("en")

        # Wikipedia tool
        @tool
        def wikipedia_search(query: str) -> str:
            """Use ONLY if the question is clearly about general world knowledge, and not likely to be found in the uploaded PDF.
            Prefer pdf_search whenever possible.
            """
            try:
                page = wikipedia.page(query)
                return page.summary
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Your query was ambiguous. Options include: {e.options[:5]}"
            except wikipedia.exceptions.PageError:
                return "No Wikipedia page found for your query."
            except Exception as ex:
                return f"Error: {ex}"

        # Register tools
        tools = [pdf_search, summarize_pdf, wikipedia_search]

        # Initialize the agent_executor once here
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "You are a legal assistant. ALWAYS use the PDF tools first (pdf_search) to answer questions. "
                "Only use other tools (like Wikipedia) if PDF tools don't have the answer or aren't applicable."
                ),
                ("human", "{input}"),
                ("ai", "{agent_scratchpad}")
                ])



        # Get user input
        user_input = st.text_input("Ask a question or type 'summarise PDF'")

        if user_input:
            try:
                start = time.process_time()
                agent_response = agent_executor.run(user_input)
                elapsed = time.process_time() - start

                st.success(f"Response time: {elapsed:.2f} seconds")
                st.markdown(f"**Agent Answer:** {agent_response}")

            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")

    finally:
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename)
else:
    st.info("Please upload a PDF file to begin.")