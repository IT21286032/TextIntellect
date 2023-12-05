# streamlit_app.py

import streamlit as st
import os
import fitz  # PyMuPDF
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI


# Function to get OpenAI API key from user input
def get_openai_api_key():
    openai_api_key = st.text_input("Enter your OpenAI API key:")
    return openai_api_key

# Function to upload PDF file and save it
def upload_pdf():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        # Save the uploaded file
        with open("uploaded_document.pdf", "wb") as f:
            f.write(uploaded_file.read())
        st.success("Document saved successfully.")
    return "uploaded_document.pdf"

# Function to process user input using the chatbot
def process_user_input(user_input):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate bot response
    response = generate_response(user_input)

    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    return response

# Main Streamlit app
def main():
    st.title("RAG Pipeline Streamlit App")

    # Sidebar
    st.sidebar.header("User Input")
    openai_api_key = get_openai_api_key()
    pdf_filename = upload_pdf()

    # Display user input details
    st.sidebar.subheader("Details:")
    if openai_api_key:
        st.sidebar.write(f"OpenAI API key set: {openai_api_key}")
    else:
        st.sidebar.warning("Please enter your OpenAI API key.")

    if pdf_filename:
        st.sidebar.write("PDF file loaded:", pdf_filename)

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Expander for RAG Pipeline and Chatbot
    with st.expander("RAG Pipeline and Chatbot"):
        if openai_api_key and pdf_filename:
            # Process PDF file and perform RAG pipeline
            try:
                # Extract text content from PDF using PyMuPDF
                pdf_document = fitz.open(pdf_filename)
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
                user_input = st.text_input("Ask me anything:")
                if st.button("Send"):
                    if user_input:
                        # Process user input using the chatbot
                        response = process_user_input(user_input)
                        st.write("AI Response:", str(response))
                    else:
                        st.warning("Please enter a question.")

                    # Display chat history
                    for message in st.session_state.chat_history:
                        role, content = message["role"], message["content"]
                        with st.chat_message(role):
                            if role == "user":
                                st.write(f"👤 **You**: {content}")
                            else:
                                st.write(f"🤖 **Assistant**: {content}")

            except Exception as e:
                st.error(f"Error processing the PDF: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
