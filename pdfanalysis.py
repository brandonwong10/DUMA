import os
from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    chunks = text_splitter.split_text(raw_text)[:5]
    return chunks

# Adds chunks to vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = DocArrayInMemorySearch.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Generates Practice Problems
def generate_practice_problems(notesVS, samplesVS, model):
    parser = StrOutputParser()
    template = """
        You are a tutor creating a practice exam for your students. Your goal is to generate 5 practice problems
        based on the given context of the exam below and follow the similar format of the pratice exam format below.

        Context: {context}
        Format: {format}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": notesVS.as_retriever(), "format": samplesVS.as_retriever()}
        | prompt 
        | model 
        | parser
    )
    st.write(chain.invoke(""))

notesVS = None
samplesVS = None
def main():
    global notesVS, samplesVS
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")
    st.set_page_config(page_title="Create Practice Problems", page_icon=":books:")
    st.header("Create Practice Problems :books:")
    with st.sidebar:
        st.subheader("Your documents")
        notes_docs = st.file_uploader(
            "Upload your Notes and Lectures", accept_multiple_files=True)
        samples_docs = st.file_uploader(
            "Upload, if any, past exams or practice exams", accept_multiple_files=True)
        #if st.button("Process"):
            #with st.spinner("Processing"): 
                # Gets pdf texts
        notes_raw_text = get_pdf_text(notes_docs)
        samples_raw_text = get_pdf_text(samples_docs)
                # Gets text chunks
        notes_chunks = get_text_chunks(notes_raw_text)
        samples_chunks = get_text_chunks(samples_raw_text)
                # Creates vector store using embeddings
        notesVS = get_vectorstore(notes_chunks)
        samplesVS = get_vectorstore(samples_chunks)
                # Create conversation chain
                # st.session_state.notesConvo = get_conversation_chain(notesVS)
                # st.session_state.samplesConvo = get_conversation_chain(samplesVS)      
    if st.button("Generate Practice Problems"):
        generate_practice_problems(notesVS, samplesVS, model)

if __name__ == '__main__':
    main()
