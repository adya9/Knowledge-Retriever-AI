# Knowledge-Retriever-AI 📚🤖

A powerful **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and have intelligent conversations about their content. Built with Streamlit, LangChain, and Google Gemini AI.

## ✨ Features

- **📄 PDF Upload & Processing** - Upload any PDF document through the web interface
- **🧠 Intelligent Q&A** - Ask questions about your document content in natural language
- **💬 Conversational Memory** - Maintains context across questions (example - remembers "she", "it", "that company")
- **⚡ Real-time Chat Interface** - Modern chat UI
- **🗑️ Easy Reset** - Clear conversation history or upload new documents anytime
- **🔒 Local Processing** - Your documents are processed locally for privacy

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini AI)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Knowledge-Retriever-AI.git
   cd Knowledge-Retriever-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
   echo "HUGGINGFACE_API_TOKEN=your_huggingface_token_here" >> .env
   ```

5. **Run the application**
   ```bash
   streamlit run rag_app.py
   ```

6. **Open your browser** to `http://localhost:8501`

## 🔑 Getting API Keys

### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### HuggingFace Token (Optional)
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Copy the token to your `.env` file

## 📖 How to Use

1. **Upload a PDF** - Click "Choose a PDF file" in the sidebar
2. **Process Document** - Click "Process PDF" to analyze the document
3. **Start Chatting** - Ask questions in the chat input at the bottom
4. **Continue Conversation** - Ask follow-up questions using pronouns like "she", "it", "that"


**Happy Knowledge Retrieving! 🎉**