from flask import Flask, request, jsonify,render_template
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import*
import os



app = Flask(__name__)

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

embeddings = download_hugging_face_embeddings()

index_name="medical-chatbot-genai"


#embed each chunk and upsert emedding into your pinecone index
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(
    model="llama3-8b-8192",  # or another supported model like "mixtral-8x7b-32768"
    temperature=0.4,
    max_tokens=500,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sentence_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["POST"])
def chat():
    msg = request.form.get('msg')
    if not msg:
        return "No input provided."
    try:
        response = rag_chain.invoke({"input": msg})
        return str(response.get("answer", "No answer found."))
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)