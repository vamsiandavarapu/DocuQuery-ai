import os

import streamlit as st
import PyPDF2
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=".env.txt")

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="DocuQuery: AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- HELPER FUNCTIONS ---------------- #

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs"""
    text = ""

    for pdf in pdf_docs:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf)

            for page in pdf_reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text

        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")

    return text


def get_text_chunks(text):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


@st.cache_resource
def get_vector_store(text_chunks):
    """Create FAISS vector store"""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vector_store


def get_conversation_chain(vector_store):
    """Create a retrieval-based question-answering function"""

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is missing or still a placeholder. Update .env or .env.txt with a real key."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
    )

    def ask(question):
        docs = vector_store.similarity_search(question, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""You are a helpful assistant answering questions about the uploaded PDF documents.
Use only the information present in the provided context.
If the answer is not in the context, say that you do not know.

Context:
{context}

Question: {question}
Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    return ask


# ---------------- MAIN APP ---------------- #

def main():

    st.title("📄 DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.markdown(
        "Upload PDF documents and ask questions about their contents."
    )

    # Session State

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar

    with st.sidebar:

        st.header("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process Documents"):

            if not pdf_docs:
                st.warning("Please upload PDF files first.")
            else:

                with st.spinner("Processing PDFs..."):

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.warning("No readable text found in PDFs.")

                    else:
                        text_chunks = get_text_chunks(raw_text)

                        vector_store = get_vector_store(
                            text_chunks
                        )

                        st.session_state.conversation_chain = \
                            get_conversation_chain(vector_store)

                        st.session_state.chat_history = []

                        st.success("Documents processed successfully!")

        st.markdown("---")
        st.caption("Built by Vamsi")

    # Display old messages

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input

    user_question = st.chat_input(
        "Ask a question about your documents..."
    )

    if user_question:

        st.chat_message("user").markdown(user_question)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        if st.session_state.conversation_chain is None:

            response = "Please upload and process PDF documents first."

            st.chat_message("assistant").markdown(response)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

        else:

            with st.spinner("Thinking..."):

                try:

                    answer = st.session_state.conversation_chain(user_question)

                    st.chat_message("assistant").markdown(answer)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()