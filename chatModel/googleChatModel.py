from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint (
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task ="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

# model = ChatGoogleGenerativeAI( model="gemini-2.5-flash")
model = ChatHuggingFace( llm = llm )
result = model.invoke("what is the capital of india")

print(result.content)