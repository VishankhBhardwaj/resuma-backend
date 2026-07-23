import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from rag import get_user_retriever
load_dotenv()

## Get the groq api key
groq_api_key = os.getenv("GROQ_API_KEY")

## Initialize the llm with a valid Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,groq_api_key=groq_api_key)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
         ("system",contextualize_q_system_prompt),
         MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ]
)

systemPrompt = """
You are a professional AI career assistant.

Your role is to help users with:
- Resume-based questions
- Interview preparation
- Generating professional self-introductions
- Answering career-related queries clearly and confidently

Rules:
1. If the question is based on the user's resume, answer strictly using the provided context.
2. If the answer is not present in the context, say: "I could not find this information in the resume."
3. For interview questions, give structured, concise, and professional answers.
4. For introduction requests, generate a confident and natural-sounding self-introduction.
5. Keep responses clear, short, and human-like.
6. Do not make up fake details not present in the context.
7. Maintain a supportive and professional tone.
You are speaking to a college student preparing for placements.
"\n\n"
    "{context}"
"""

## Prompt for the llm
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemPrompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

## Initialize retrieval chain
def get_retrieval_chain(user_id: str):
    retriever = get_user_retriever(user_id)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create and return the retrieval chain
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return retrieval_chain
        
