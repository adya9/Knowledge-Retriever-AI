import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGApplication:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.qa_chain = None
        self.memory = None
        
    def process_pdf(self, pdf_file):
        """Process uploaded PDF and create vector store"""
        st.info("Processing PDF...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            st.success(f"PDF processed! Created {len(chunks)} chunks.")
            return True
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def setup_qa_chain(self):
        """Setup conversational question-answering chain"""
        if self.vectorstore is None:
            st.error("Please upload and process a PDF first!")
            return False
        
        # Initialize LLM (using Google Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize conversation memory (remembers last 10 exchanges)
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 question-answer pairs
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create Conversational QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        return True
    
    def ask_question(self, question):
        """Ask a question with conversation context"""
        if self.qa_chain is None:
            return "Please process a PDF first!"
        
        try:
            # Use conversational chain that includes chat history
            result = self.qa_chain({"question": question})
            return result["answer"], result["source_documents"]
        except Exception as e:
            return f"Error: {str(e)}", []
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()

def main():
    st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
    st.title("📚 RAG PDF Question & Answer System")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # Initialize app and chat history
    if "rag_app" not in st.session_state:
        st.session_state.rag_app = RAGApplication()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to start asking questions"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                success = st.session_state.rag_app.process_pdf(uploaded_file)
                if success:
                    st.session_state.rag_app.setup_qa_chain()
                    st.success("PDF processed successfully!")
                    # Clear chat history and memory when new PDF is processed
                    st.session_state.chat_history = []
                    if st.session_state.rag_app.memory:
                        st.session_state.rag_app.clear_memory()
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            # Also clear the conversation memory
            if st.session_state.rag_app.memory:
                st.session_state.rag_app.clear_memory()
            st.rerun()
    
    # Main chat area
    st.header("💬 Chat with your PDF")
    
    if st.session_state.rag_app.vectorstore is not None:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                               
        # Chat input at the bottom
        if question := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(question)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = st.session_state.rag_app.ask_question(question)
                
                st.write(answer)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                })
                
    else:
        st.info("👆 Please upload a PDF file to get started!")
        
        # Show example questions when no PDF is loaded
        st.markdown("### 💡 Example questions you can ask:")
        st.markdown("""
        - What is this document about?
        - Summarize the main points
        - What are the key findings?
        - Explain the methodology used
        - What are the conclusions?
        """)

if __name__ == "__main__":
    main() 