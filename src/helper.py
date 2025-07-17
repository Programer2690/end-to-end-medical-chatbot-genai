from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings


#extract data from pdf file
def load_pdf_file(data):
    loader=DirectoryLoader(data , glob="**/*.pdf" ,loader_cls=PyPDFLoader)
    documments=loader.load()
    
    return documments


#split data into text chunks
def split_text(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


#download huging face embeddings
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings