import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.exceptions import NotFoundException
from pinecone import Pinecone
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "resuma"

# def delete_all_vectors():
#     pc = Pinecone(api_key=pinecone_api_key)
#     index = pc.Index(index_name)

#     try:
#         index.delete(delete_all=True)
#     except NotFoundException:
#         print("No vectors found. Skipping delete.")

def load_documents(pdf_path="uploads/resume.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def vector_embeddings(documents,user_id):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100 
    )
    final_documents = text_splitter.split_documents(documents)
    for doc in final_documents:
        doc.metadata["user_id"] = user_id
    vectorstoredb = PineconeVectorStore.from_documents(
        final_documents,
        embeddings,
        index_name=index_name
    )
    retriever = vectorstoredb.as_retriever(search_kwargs={
        "k": 4,
        "filter": {
            "user_id": user_id
        }
    }) 
    return retriever

def get_user_retriever(user_id):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstoredb = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = vectorstoredb.as_retriever(search_kwargs={
        "k": 4,
        "filter": {
            "user_id": user_id
        }
    }) 
    return retriever



    
    
    
    
    