# PDF Insight 📄💡

Welcome to the **PDF Query Assistant**, a Streamlit-based web application that enables you to upload PDF files and ask detailed questions about their content. The app uses advanced AI models for text processing, embedding generation, and conversational AI to provide accurate and context-based answers.

---

## Features ✨

- **PDF Upload and Text Extraction**: Upload multiple PDF files, and the app will extract their text for processing.
- **Text Chunking**: Splits extracted text into manageable chunks for better processing.
- **Vector Storage**: Uses a vector database to store and retrieve relevant text chunks for answering questions.
- **Conversational AI**: Allows natural language interaction for question-answering.

---

## AI Models Used 🤖

### 1. **OpenAI Embeddings**

- **Model**: `text-embedding-ada-002`
- **Purpose**: Converts text into high-dimensional embeddings for semantic similarity searches.
- **Use Case**: Embedding generation for storing and retrieving chunks of text from a vector database.

### 2. **Google Generative AI (Gemini-Pro)**

- **Model**: `gemini-pro`
- **Purpose**: Provides accurate, context-aware answers to user questions.
- **Use Case**: Generates detailed and accurate conversational responses based on retrieved text.


### 3. LangChain Framework
   - **Purpose**:
     - Combines various components like embeddings, vector databases, and prompt templates into a seamless chain for question answering.
   - **Why It’s Used**:
     - Simplifies the orchestration of different AI components, enabling the app to focus on answering questions accurately and efficiently.

### 4. FAISS(Facebook AI Similarity Search)
   - **Purpose**:
     - Efficiently stores and retrieves embeddings for similarity search.
     - Matches user queries to relevant PDF content chunks.
   - **Why It’s Used**:
     - Ensures fast and scalable search, even for large volumes of text data.


---

### How It All Works Together

1. **PDF Processing**: Extracts raw text from PDFs using PyPDF2.
2. **Text Chunking**: Breaks the text into manageable chunks with overlapping content for better context handling.
3. **Embeddings**: 
   - Converts chunks into vector embeddings using OpenAI's `text-embedding-ada-002`.
4. **Vector Search**:
   - Stores embeddings in FAISS, allowing similarity search to identify the most relevant chunks for a query.
5. **Question-Answering**:
   - Combines relevant text chunks with user queries.
   - Google Generative AI processes the query to generate detailed, context-aware answers.

---

### Why These Models?

- **Accuracy**: OpenAI Embeddings and FAISS ensure high relevance for text retrieval.
- **Conversational Clarity**: Google Generative AI excels at producing clear and human-like responses.
- **Scalability**: FAISS and LangChain handle large volumes of data efficiently, enabling the app to support multiple PDFs with ease.

---
