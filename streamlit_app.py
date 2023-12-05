# streamlit_app.py

import streamlit as st
import fitz  # PyMuPDF
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

# Function to get OpenAI API key from user input
def get_openai_api_key():
    openai_api_key = st.text_input("Enter your OpenAI API key:")
    return openai_api_key

# Function to upload PDF file
def upload_pdf():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    return uploaded_file

# Main Streamlit app
def main():
    st.title("RAG Pipeline Streamlit App")

    # Get OpenAI API key
    openai_api_key = get_openai_api_key()

    # Set OpenAI API key
    if openai_api_key:
        st.write(f"OpenAI API key set: {openai_api_key}")
    else:
        st.warning("Please enter your OpenAI API key.")

    # Upload PDF file
    pdf_file = upload_pdf()

    # Display PDF file details
    if pdf_file:
        st.write("PDF file uploaded:", pdf_file.name)

        # Process PDF file and perform RAG pipeline
        try:
            # Extract text content from PDF using PyMuPDF
            pdf_document = fitz.open(pdf_file)
            text_content = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text_content += page.get_text()

            # Perform RAG pipeline with OpenAI
            document = Document(text=text_content)
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=openai_api_key)
            service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
            index = VectorStoreIndex.from_documents([document], service_context=service_context)
            query_engine = index.as_query_engine()

            # Chat interface
            conversation = st.text_area("Chat with AI:", height=200, max_chars=500)
            if st.button("Send"):
                if conversation:
                    response = query_engine.query(conversation)
                    st.write("AI Response:", str(response))
                else:
                    st.warning("Please enter a message.")

        except Exception as e:
            st.error(f"Error processing the PDF: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
