import os  # Importing OS module for environment variables
from langchain_openai import OpenAI  # Importing OpenAI module
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter for text splitting
import streamlit as st  # Importing Streamlit for creating web applications
from dotenv import load_dotenv  # Importing load_dotenv for loading environment variables
from PyPDF2 import PdfReader  # Importing PdfReader for reading PDF files
from langchain.text_splitter import CharacterTextSplitter  # Importing CharacterTextSplitter for text splitting
from langchain_openai.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings for embeddings
from langchain_openai.chat_models import ChatOpenAI  # Importing ChatOpenAI for chat models
from langchain_community.vectorstores import DocArrayInMemorySearch  # Importing DocArrayInMemorySearch for vector storage
from langchain.memory import ConversationBufferMemory  # Importing ConversationBufferMemory for conversation storage
from langchain.chains import ConversationalRetrievalChain  # Importing ConversationalRetrievalChain for conversation chains
from langchain.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate for chat prompts
from langchain_core.output_parsers import StrOutputParser  # Importing StrOutputParser for output parsing
from langchain_core.runnables import RunnablePassthrough  # Importing RunnablePassthrough for running tasks
from htmlTemplates import css, bot_template  # Importing CSS and bot_template HTML templates

# Turns PDF into raw text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Turns raw text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=20)
    chunks = text_splitter.split_text(raw_text)[:5]  # Limiting to 5 chunks
    return chunks

# Adds chunks to vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_texts(text_chunks, embeddings)
    return vectorstore

# Generates Practice Problems
def generate_practice_problems(notesVS, samplesVS, model):
    parser = StrOutputParser()
    template1 = """
        You are a tutor tasked with creating practice problems for your students based on the context provided (textbooks, 
        notes, and lectures for the class) Your goal is to generate 5 practice problems that closely simulate the types of problems 
        typically given by the format below (previous midterms, finals, or practice exams). Do not generate solutions.

        Context: {context}
        Format: {format}

        Your generated practice problems should align with the topics covered in the provided context and 
        adhere to the format commonly used by the professor in assessments.
    """
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain1 = (
        {"context": notesVS.as_retriever(), "format": samplesVS.as_retriever()}
        | prompt1
        | model 
        | parser
    )
    with st.spinner("Creating Problems"):
        problems = chain1.invoke("")
        st.write(bot_template.replace("{{MSG}}", problems), unsafe_allow_html=True)
    template2 = """
        You are an instructor preparing an answer key for the questions provided below. 
        Your goal is to generate accurate answers based on the context provided from textbooks, notes, and lectures for the class.
        
        Context: {context}
        Problems: {problems}
        
        Please provide detailed and correct answers for each question that align with the content covered in the provided context.
    """
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = (
        {"context": notesVS.as_retriever(), "problems": RunnablePassthrough()}
        | prompt2
        | model 
        | parser
    )
    with st.spinner("Creating Solutions"):
        answers = chain2.invoke(problems)
        st.write(bot_template.replace("{{MSG}}", answers), unsafe_allow_html=True)


def main():
    load_dotenv()  # Loading environment variables
    st.set_page_config(page_title="Create Practice Problems", page_icon=":books:")  # Configuring page title and icon
    st.write(css, unsafe_allow_html=True)  # Writing CSS template
    # Checking if session variables exist, if not, initializing them
    if "notesVS" not in st.session_state:
        st.session_state.notesVS = None
    if "samplesVS" not in st.session_state:
        st.session_state.samplesVS = None
    if "notesConvo" not in st.session_state:
        st.session_state.notesConvo = None
    if "samplesConvo" not in st.session_state:
        st.session_state.notesConvo = None
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Retrieving OpenAI API key from environment variables
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")  # Initializing OpenAI chat model
    st.header("Create Practice Problems :books:")  # Writing header
    with st.sidebar:  # Creating sidebar
        st.subheader("Your documents")  # Subheader for documents section
        notes_docs = st.file_uploader(
            "Upload your Notes and Lectures", accept_multiple_files=True)  # Uploader for notes and lectures
        samples_docs = st.file_uploader(
            "Upload, if any, past exams or practice exams", accept_multiple_files=True)  # Uploader for past exams
        if st.button("Process"):  # Button for processing uploaded files
            with st.spinner("Processing"): 
                # Gets pdf texts
                notes_raw_text = get_pdf_text(notes_docs)
                samples_raw_text = get_pdf_text(samples_docs)
                # Gets text chunks
                notes_chunks = get_text_chunks(notes_raw_text)
                samples_chunks = get_text_chunks(samples_raw_text)
                # Creates vector store using embeddings
                st.session_state.notesVS = get_vectorstore(notes_chunks)
                st.session_state.samplesVS = get_vectorstore(samples_chunks)  
    if st.button("Generate Practice Problems"):  # Button for generating practice problems
            generate_practice_problems(st.session_state.notesVS, st.session_state.samplesVS, model)
if __name__ == '__main__':
    main()
