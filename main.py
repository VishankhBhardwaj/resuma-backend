from llm import llm
from rag import vector_embeddings,load_documents
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from tools import get_interview_prep_tool
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
app = FastAPI()
from pydantic import BaseModel



origins = [
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ddgs_tool = DuckDuckGoSearchRun();

tool_prompt = """
You are an AI placement assistant helping a college student prepare for jobs.
You have access to two tools:

1) Interview Prep Tool:
- Use this when the user asks about their resume, skills, projects, experience, or education.
- This tool will return a COMPLETE answer - do NOT call it again or call any other tool after getting the response.

2) Video Intro Tool:
- Use this when the user wants to create a video introduction, self-intro video, or talking avatar based on their resume.
- This tool will return a video URL or status message - do NOT call it again or call any other tool after getting the response.

CRITICAL RULES:
- If the question asked from the user in not technology related then simply reply back to user with a polite message that you wont be able to help with that question.
- Call a tool ONLY ONCE per user query.
- After receiving a tool's response, that IS your final answer - return it immediately to the user.
- DO NOT call the same tool multiple times.
- DO NOT call another tool after getting a response from the first tool.
- If the question is general career advice (not about their specific resume), answer directly without using any tool.
- Do not make up details not present in the resume.
- Keep answers clear, confident, and professional.
"""
# tools will be initialized dynamically per request



# @app.post("/delete_vectors")
# def delete_vectors():
#     delete_all_vectors()
#     return {"answer":"Deleted Successfully"}

@app.post("/file_upload")
def upload_file(file: UploadFile = File(...),user_id: str = Form(...)):
    
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{user_id}_resume.pdf"
    with open(file_path, "wb") as file_obj:
        shutil.copyfileobj(file.file, file_obj)

    documents = load_documents(file_path)
    vector_embeddings(documents,user_id)

    return {"message": "Embeddings created"}
    

@app.post("/ai_agent")
def invoke_agent(query: str = Form(...),user_id:str = Form(...)):
    
    user_interview_tool = get_interview_prep_tool(user_id)
    tools = [user_interview_tool, ddgs_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", tool_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate", 
        return_intermediate_steps=False,  
    )
    
    response = agent_executor.invoke({"input": query})
    return {"answer": response['output']}







