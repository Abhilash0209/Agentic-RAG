
# Agentic RAG

Agentic RAG is a Streamlit-based AI assistant that allows users to interact with PDF documents through natural language. It uses an agent-based Retrieval-Augmented Generation (RAG) approach powered by LangChain, FAISS vector search, and the gemini-1.5-flash model. The assistant supports PDF content summarization, intelligent question answering, and fallback Wikipedia search when needed.

## Features

- Upload and process PDF documents
- Ask questions about the document using natural language
- Automatically choose between:
  - PDF content search
  - PDF summarization
  - Wikipedia search for general knowledge
- Uses LangChain agent tools with a zero-shot reasoning strategy
- Powered by HuggingFace embeddings and gemini-1.5-flash model

## How It Works

1. You upload a PDF file.
2. The content is split into chunks and embedded using `sentence-transformers/all-mpnet-base-v2`.
3. A FAISS index is created for fast document retrieval.
4. Three tools are available to the agent:
   - `pdf_search`: Searches the PDF content and retrieves relevant sections.
   - `summarize_pdf`: Summarizes the main ideas from the beginning of the document.
   - `wikipedia_search`: Looks up general world knowledge using the Wikipedia API.
5. A zero-shot agent chooses which tool to use based on your query.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Abhilash0209/Agentic_RAG.git
cd Agentic_RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here  
```

### 5. Run the app

```bash
streamlit run app.py
```

## Example Questions

- "What are the key responsibilities mentioned in this document?"
- "Summarise PDF"
- "Who is the issuing authority for this document?"
- "What is the legal definition of the term used on page 3?"


## Project Structure

```
Agentic_RAG/
├── app.py                # Main application code
├── .env                  # Environment variables (not committed)
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- [LangChain](https://www.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
