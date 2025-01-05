---

## AI Models Used

This project uses cutting-edge AI models to provide accurate and efficient query handling for PDF content:

### 1. **OpenAI Embeddings** (`text-embedding-ada-002`)
   - **Purpose**: 
     - Converts text into high-dimensional embeddings for similarity search.
     - Enables efficient and meaningful retrieval of relevant information from the uploaded PDFs.
   - **Why It’s Used**: 
     - This model provides robust text representations, ensuring accurate matching between user queries and document content.

### 2. **Google Generative AI** (Gemini Pro)
   - **Purpose**:
     - Processes user queries in a conversational manner.
     - Generates detailed, context-aware responses to user questions based on the extracted PDF content.
   - **Why It’s Used**: 
     - Provides highly detailed and human-like responses while ensuring correctness when referencing provided content.
     - It has a low temperature (0.3) setup for increased factuality and precision.

### 3. **LangChain Framework**
   - **Purpose**:
     - Combines various components like embeddings, vector databases, and prompt templates into a seamless chain for question answering.
   - **Why It’s Used**:
     - Simplifies the orchestration of different AI components, enabling the app to focus on answering questions accurately and efficiently.

### 4. **FAISS** (Facebook AI Similarity Search)
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
