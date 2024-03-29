import os  # Importing OS module for environment variables
#from langchain_openai import OpenAI  # Importing OpenAI module
from dotenv import load_dotenv  # Importing load_dotenv for loading environment variables
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Importing RecursiveCharacterTextSplitter for text splitting
import streamlit as st  # Importing Streamlit for creating web applications\
from PyPDF2 import PdfReader  # Importing PdfReader for reading PDF files
from langchain_openai.embeddings import OpenAIEmbeddings  # Importing OpenAIEmbeddings for embeddings
from langchain_openai.chat_models import ChatOpenAI  # Importing ChatOpenAI for chat models
from langchain_community.vectorstores import DocArrayInMemorySearch  # Importing DocArrayInMemorySearch for vector storage
from langchain.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate for chat prompts
from langchain_core.output_parsers import StrOutputParser  # Importing StrOutputParser for output parsing
from langchain_core.runnables import RunnablePassthrough  # Importing RunnablePassthrough for running tasks
from htmlTemplates import css, bot_template  # Importing CSS and bot_template HTML templates
import streamlit_shadcn_ui as ui
#from auth import *


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
def generate_practice_problems(notesVS, samplesVS, model, number, diff):
    parser = StrOutputParser()
    template1 = """
        You are a professor tasked with creating practice problems for your students based on the context provided (textbooks, 
        notes, and lectures for the class) Your goal is to generate practice problems that closely simulate the types of problems 
        typically given by the format below (previous midterms, finals, or practice exams). The difficulty of these problems should be: {difficulty}
        Do not generate solutions. No other text is need other than the questions.

        Context: {context}
        Format: {format}
        Difficulty: {difficulty}

        In your problems, do not reference charts, images, or anything from the given resources as users will not know what you are referencing.
    """
    prompt1 = ChatPromptTemplate.from_template(template1)
    diff_str = str(diff)
    chain1 = (
        {"context": notesVS.as_retriever(), "format": samplesVS.as_retriever(), "difficulty": RunnablePassthrough() }
        | prompt1
        | model 
        | parser
    )
    with st.spinner("Creating Problems"):
        problems = chain1.invoke(diff_str)
        st.header("Practice Problems:")
        st.write(problems)
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
        st.header("Answer Key:")
        st.write(answers)

def main():
    load_dotenv()  # Loading environment variables
    st.set_page_config(page_title="Generate Practice Problems Using Ai", page_icon=":sparkles:")  # Configuring page title and icon
    st.write(css, unsafe_allow_html=True)  # Writing CSS template
    # Checking if session variables exist, if not, initializing them
    if "notes_docs" not in st.session_state:
        st.session_state.notes_docs = None
    if "samples_docs" not in st.session_state:
        st.session_state.samples_docs = None
    if "notesVS" not in st.session_state:
        st.session_state.notesVS = None
    if "samplesVS" not in st.session_state:
        st.session_state.samplesVS = None
    if "num_of_questions" not in st.session_state:
        st.session_state.num = None
    if "notesConvo" not in st.session_state:
        st.session_state.notesConvo = None
    if "samplesConvo" not in st.session_state:
        st.session_state.notesConvo = None
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = None
    with st.sidebar:
        st.title("StudyGen")
        st.session_state.openaiKey = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.link_button("Give Feedback :arrow_right:", "https://docs.google.com/forms/d/e/1FAIpQLSefz8m7lbE1Q7Fm_iOWw4yDkrN7PSSX_2V9yyJYnJkeg2rXDg/viewform?usp=sf_link")
    OPENAI_API_KEY = st.session_state.openaiKey  # Retrieving OpenAI API key
    #model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")  # Initializing OpenAI chat model
    st.header("Create Practice Problems :book:")  # Writing header
    st.write("Our program generates practice problems using the context you provide, simulating problems that may be on the exam. The more context provide, the better the results!")
    cols = st.columns(2)
    with cols[0]:
        st.session_state.notes_docs = st.file_uploader(
            "Upload your Notes and Lectures in PDF form", accept_multiple_files=True)  # Uploader for notes and lectures
    with cols[1]:
            st.session_state.samples_docs= st.file_uploader(
            "Upload past exams or practice exams in PDF form", accept_multiple_files=True)
    if st.button("Process Documents"):
        with st.spinner("Processing"): 
            notes_raw_text = get_pdf_text(st.session_state.notes_docs)
            notes_chunks = get_text_chunks(notes_raw_text)
            st.session_state.notesVS = get_vectorstore(notes_chunks)
            samples_raw_text = get_pdf_text(st.session_state.samples_docs)
            samples_chunks = get_text_chunks(samples_raw_text)
            st.session_state.samplesVS = get_vectorstore(samples_chunks)
    st.session_state.difficulty = st.radio('Pick a difficulty:', ['Easy','Medium',"Hard"])
    generate_btn = st.button("Generate Practice Problems")
    if generate_btn:  # Button for generating practice problems
        if OPENAI_API_KEY is None or OPENAI_API_KEY == '':
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        if st.session_state.notesVS is None:
           ui.alert_dialog(show=generate_btn, title="Missing Notes", description="Please attach a notes file", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_1")
        elif st.session_state.samplesVS is None:
           ui.alert_dialog(show=generate_btn, title="Missing Samples", description="Please attach a exam sample file", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_2")
        else:
            try:
                model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
                generate_practice_problems(st.session_state.notesVS, st.session_state.samplesVS, model, st.session_state.num, st.session_state.difficulty)
            except Exception as e:
                st.error("Invalid OpenAI API key. Please check your API key and try again.")
                st.stop()

            
if __name__ == '__main__':
    main()