from langchain.tools import tool
from llm import get_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time


store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



def get_interview_prep_tool(user_id: str):
    @tool(return_direct=True)
    def interview_prep_tool(input: str) -> str:
        """
        Answers questions about the user's resume, skills, projects, experience, or education.
        Use this when the user asks about their background or qualifications.
        
        Args:
            input: The question about the resume or background
        
        Returns:
            str: A complete answer to the user's question
        """
        try:
            retrieval_chain = get_retrieval_chain(user_id)
            conversational_rag_chain = RunnableWithMessageHistory(
                retrieval_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            response = conversational_rag_chain.invoke(
                {"input": input},
                config={
                    "configurable": {"session_id": user_id}
                },
            )
            return response['answer']
        except Exception as e:
            return f"Error retrieving information: {str(e)}"

    return interview_prep_tool

