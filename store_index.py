from src.helper import load_pdf_file, split_text, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()
pinecone_api_key = os.environ.get("pinecone_api_key")
os.environ["PINECONE_API_KEY"] = pinecone_api_key

extracted_data = load_pdf_file("data/")
text_chunks = split_text(extracted_data)
embeddings = download_hugging_face_embeddings()



pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbot-genai"


pc.create_index(
    name=index_name,
    dimension=384,  # Dimension of the embeddings
    metric="cosine",  # Similarity metric
    spec=ServerlessSpec(
        cloud="aws",  # Cloud provider
        region="us-east-1",  # Region for the index
    )
)


docseach=PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)