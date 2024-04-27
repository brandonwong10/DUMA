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
from openai import OpenAI
from pydub import AudioSegment
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
def generate_notes(contextVS, style, model):
    parser = StrOutputParser()
    template1 = """
        Generate structured notes for students based on given context, focusing on key topics and using the style provided.
        Ensure that you are only writing notes about the main topics of the lecture.  

        Context: {context}
        Style: {style}
    """
    prompt1 = ChatPromptTemplate.from_template(template1)
    style_str = str(style)
    chain1 = (
        {"context": contextVS.as_retriever(), "style": RunnablePassthrough() }
        | prompt1
        | model 
        | parser
    )
    with st.spinner("Creating Problems"):
        notes = chain1.invoke(style_str)
        st.header("Notes:")
        st.write(notes)

def main():
    load_dotenv()  # Loading environment variables
    client = OpenAI()
    st.set_page_config(page_title="Generate Practice Problems Using Ai", page_icon=":sparkles:")  # Configuring page title and icon
    st.write(css, unsafe_allow_html=True)  # Writing CSS template
    # Checking if session variables exist, if not, initializing them
    if "notes_docs" not in st.session_state:
        st.session_state.notes_docs = None
    if "notesVS" not in st.session_state:
        st.session_state.notesVS = None
    if "notesConvo" not in st.session_state:
        st.session_state.notesConvo = None
    if "audio_docs" not in st.session_state:
        st.session_state.audio_docs = None
    if "audioVS" not in st.session_state:
        st.session_state.audioVS = None
    if "audioConvo" not in st.session_state:
        st.session_state.audioConvo = None
    with st.sidebar:
        st.title("StudyGen")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # Retrieving OpenAI API key
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")  # Initializing OpenAI chat model
    st.header("Create Notes :book:")  # Writing header
    st.write("Our program generates practice problems using the context you provide, simulating problems that may be on the exam. The more context provide, the better the results!")
    cols = st.columns(2)
    with cols[0]:
        st.session_state.notes_docs = st.file_uploader(
            "Upload PDF", accept_multiple_files=True)  # Uploader for notes and lectures
    with cols[1]:
        st.session_state.audio_file= st.file_uploader(
            "Upload Audio File", type=["wav", "mp3", "m4a"])

    if st.button("Process Documents"):
        with st.spinner("Processing"): 
            notes_raw_text = get_pdf_text(st.session_state.notes_docs)
            notes_chunks = get_text_chunks(notes_raw_text)
            st.session_state.notesVS = get_vectorstore(notes_chunks)
    if st.button("Process Audio"):
        with st.spinner("Processing"):
            transcription = client.audio.transcriptions.create(model="whisper-1", file=st.session_state.audio_file)
            st.write(transcription)
            #audio_raw_text = get_pdf_text(st.session_state.audio_file)
            #audio_chunks = get_text_chunks(audio_raw_text)
            #st.session_state.audioVS = get_vectorstore(audio_chunks)
    
    style = st.text_input("Enter your preferred style", None)
    generate_btn = st.button("Generate Notes")
    if generate_btn:  # Button for generating practice problems
        if OPENAI_API_KEY is None or OPENAI_API_KEY == '':
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        if st.session_state.notesVS is None:
           ui.alert_dialog(show=generate_btn, title="Missing Notes", description="Please attach a notes file", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_1")
        else:
            try:
                model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
                generate_notes(st.session_state.notesVS, style, model)
            except Exception as e:
                st.error("Invalid OpenAI API key. Please check your API key and try again.")
                st.stop()

            
if __name__ == '__main__':
    main()