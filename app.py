import streamlit as st
import langchain_google_genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# from langchain_google_genai import GoogleGenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs: #for every pdf in uploaded pdfs
        pdf_file = io.BytesIO(pdf.read())  # Convert bytes to file-like object
        pdf_reader = PdfReader(pdf_file)
        # pdf_reader=PdfReader(pdf) #read the pdf
        for page in pdf_reader.pages: #for every page extract text
            text+=page.extract_text() #add it to the text
    return text


def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template=""" 
    Answer the question as detailed as possible from the provided context ,make sure to provide all the details ,
    if answer is not in the provided context just say "Answer is not available in the context", don't provide wrong answers\n\n

    Context:\n{context}?\n
    Question:\n{context}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","questions",])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain



def user_input(user_question):
    embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")
    new_db=FAISS.load_local("faiss_index",embeddings)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()

    response=chain({
        "input_documents":docs,"question":user_question }, return_only_outputs=True)

    print(response)
    st.write("Reply:" ,response["output_text"])



def main():
    st.set_page_config(
    page_title="PDF Query Assistant",  # Title of the web page
    page_icon="📄",  # Icon that appears on the browser tab
    layout="wide",  # Layout option
    initial_sidebar_state="expanded"  # Sidebar can be expanded by default
    )

    st.title("Welcome to PDF Query Assistant")  
    # st.write("Upload a PDF and ask questions about its content.")

    user_question=st.text_input("Ask your question from the PDFs")
    if(user_question):
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs=st.file_uploader("Upload your files and click submit",accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Reading Files"):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")

            
if __name__=="__main__":
    main()


