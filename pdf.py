from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

# Load your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Split text into manageable chunks
def split_text(text, max_tokens=3000):
      words = text.split()
      if len(words) <= max_tokens:
            return [text]

      chunks = []
      current_chunk = []
      current_tokens = 0

      for word in words:
            word_tokens = len(word.split()) 
            if current_tokens + word_tokens <= max_tokens:
                  current_chunk.append(word)
                  current_tokens += word_tokens
            else:
                  chunks.append(' '.join(current_chunk))
                  current_chunk = [word]
                  current_tokens = word_tokens

      if current_chunk:
            chunks.append(' '.join(current_chunk))

      return chunks

# Extract text from uploaded PDF
def extract_pdf(uploaded_file, max_tokens=3000):
      with BytesIO(uploaded_file.read()) as file:
            pdfreader = PdfReader(file)
            raw_text = ''
            for page in pdfreader.pages:
                  content = page.extract_text()
                  if content:
                        raw_text += content

      return split_text(raw_text, max_tokens)

# Convert text chunks to embeddings using HuggingFace
def convert_to_vector(pdf_text_chunks):
      embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
      documents = [Document(page_content=chunk) for chunk in pdf_text_chunks]
      db = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")
      return db

# Use Groq LLM to answer questions
def create_lang():
      llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
      chain = load_qa_chain(llm, chain_type="stuff")
      return chain

# Streamlit frontend
def main():
      st.set_page_config(page_title="PDF Q&A")
      st.header("PDF Q&A")

      uploaded_file = st.file_uploader("Upload File", type="pdf")

      if uploaded_file is not None:
            pdf_text_chunks = extract_pdf(uploaded_file)
            st.write("File uploaded successfully")

            document_search = convert_to_vector(pdf_text_chunks)
            chain = create_lang()

            if "history" not in st.session_state:
                  st.session_state.history = []

            user_input = st.text_input("Ask PDF")

            if st.button("Submit"):
                  if user_input:
                        docs = document_search.similarity_search(user_input)
                  if docs:
                        answer = chain.run(input_documents=docs, question=user_input)
                        st.write(f"Answer: {answer}")
                  else:
                        st.write("No relevant documents found.")
                  
                  st.session_state.history.append((user_input, answer))

if __name__ == "__main__":
      main()
