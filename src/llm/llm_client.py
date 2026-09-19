from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Replace this model if your Groq dashboard shows a different available model.
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-20b",
    temperature=0,
)
