from flask import Flask, render_template, jsonify, request
from src.helper import generate_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"]=GROQ_API_KEY
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

embeddings=generate_embeddings()
docsearch=PineconeVectorStore.from_existing_index(embedding=embeddings,index_name="medibot2")

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})
chatmodel=ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know.\n\n{context}"
    ),
    ("human", "{input}")])

question_answer_chain=create_stuff_documents_chain(chatmodel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")
@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    imput=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("response:",response["answer"])
    return str(response["answer"])
if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
