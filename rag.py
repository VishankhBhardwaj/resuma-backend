import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone.exceptions import NotFoundException
from pinecone import Pinecone
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "resuma"

# Initialize Pinecone Client
pc = Pinecone(api_key=pinecone_api_key)

# Singleton Embedding model (loaded once at module load)
from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)



def delete_user_vectors(user_id: str):
    """Deletes all existing vectors for a specific user namespace before adding new ones."""
    try:
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace=user_id)
        print(f"Successfully cleared previous vectors for namespace: {user_id}")
    except NotFoundException:
        print(f"No existing vectors found for namespace: {user_id}. Skipping delete.")
    except Exception as e:
        print(f"Notice during vector cleanup for user {user_id}: {e}")


def load_documents(pdf_path="uploads/resume.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


def vector_embeddings(documents, user_id: str):
    delete_user_vectors(user_id)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100 
    )
    final_documents = text_splitter.split_documents(documents)
    for doc in final_documents:
        doc.metadata["user_id"] = user_id
    vectorstoredb = PineconeVectorStore.from_documents(
        documents=final_documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=user_id
    )

    retriever = vectorstoredb.as_retriever(search_kwargs={
        "k": 4
    }) 
    return retriever


def get_user_retriever(user_id: str):

    vectorstoredb = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=user_id
    )
    retriever = vectorstoredb.as_retriever(search_kwargs={
        "k": 4
    }) 
    return retriever




    
    
    
    
    