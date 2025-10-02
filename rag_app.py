import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
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
        """Setup question-answering chain"""
        if self.vectorstore is None:
            st.error("Please upload and process a PDF first!")
            return False
        
        # Initialize LLM (using Google Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        return True
    
    def ask_question(self, question):
        """Ask a question and get answer"""
        if self.qa_chain is None:
            return "Please process a PDF first!"
        
        try:
            result = self.qa_chain({"query": question})
            return result["result"], result["source_documents"]
        except Exception as e:
            return f"Error: {str(e)}", []

def main():
    st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
    st.title("📚 RAG PDF Question & Answer System")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # Initialize app
    if "rag_app" not in st.session_state:
        st.session_state.rag_app = RAGApplication()
    
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
    
    # Main area for Q&A
    st.header("❓ Ask Questions")
    
    if st.session_state.rag_app.vectorstore is not None:
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.rag_app.ask_question(question)
                
                st.subheader("Answer:")
                st.write(answer)
                
                if sources:
                    st.subheader("Sources:")
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:**")
                        st.write(doc.page_content[:200] + "...")
                        st.write(f"**Page:** {doc.metadata.get('page', 'Unknown')}")
                        st.divider()
    else:
        st.info("👆 Please upload a PDF file to get started!")

if __name__ == "__main__":
    main() 