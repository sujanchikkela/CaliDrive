import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os

# Set Streamlit title and description
st.title("CaliDrive 🚘")
st.write("Enter your current OpenAI API Key and ask questions about the default PDF file.")

# Define the path to the default PDF file
default_pdf_path = 'dmv.pdf'  # Replace with the actual file path

# Add a text input for the OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API Key", type='password')

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def chat_with_pdf(text, openai_key, query):
    try:
        os.environ['OPENAI_API_KEY'] = openai_key
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        from langchain.chains.question_answering import load_qa_chain
        from langchain.llms import OpenAI
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        return result
    except Exception as e:
        st.error(f"Error while processing the PDF: {str(e)}")
        return None

# Add a text input for user questions
user_question = st.text_input("Ask a question")

if st.button("Chat"):
    if openai_api_key and user_question:
        pdf_text = extract_text_from_pdf(default_pdf_path)
        if pdf_text:
            result = chat_with_pdf(pdf_text, openai_api_key, user_question)
            if result:
                st.write("Answer:", result)
        else:
            st.warning("Unable to extract text from the default PDF. Please check the file path.")
    else:
        st.warning("Please provide an OpenAI API Key and enter a question.")

if __name__ == "__main__":
    st.set_option('deprecation.showfileUploaderEncoding', False)
