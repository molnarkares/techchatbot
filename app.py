from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms import Anthropic
from langchain_ollama.llms import OllamaLLM
import os
import shutil
from typing import Dict

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# PDF storage
PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# API Keys
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "")
}

# Vector store
embedding_function = OpenAIEmbeddings(api_key=api_keys["openai"])
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# LLM setup
llm = None

class Question(BaseModel):
    text: str

class APIKey(BaseModel):
    key: str

def get_llm():
    if llm is None:
        raise HTTPException(status_code=400, detail="LLM not set")
    return llm

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(PDF_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process and store embeddings
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)

    try:
        vectorstore.add_documents(splits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    return {"message": "PDF uploaded and processed successfully"}

@app.get("/list")
async def list_pdfs():
    pdfs = os.listdir(PDF_DIR)
    return JSONResponse(content=pdfs)

@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        # Delete the PDF file
        os.remove(file_path)

        # Remove the associated embeddings
        try:
            # Assuming the document ID in Chroma is the filename
            vectorstore.delete(ids=[filename])
        except Exception as e:
            print(f"Error deleting embeddings: {str(e)}")

        # Refresh the PDF list
        pdfs = os.listdir(PDF_DIR)
        return JSONResponse(content=pdfs)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/ask")
async def ask_question(text: str = Form(...), current_llm: Dict = Depends(get_llm)):
    qa_chain = RetrievalQA.from_chain_type(current_llm, retriever=vectorstore.as_retriever())
    response = qa_chain({"query": text})
    return {"answer": response['result']}

@app.post("/set_llm")
async def set_llm(llm_type: str):
    global llm
    try:
        if llm_type == "openai":
            llm = OpenAI(api_key=api_keys["openai"])
        elif llm_type == "anthropic":
            llm = Anthropic(api_key=api_keys["anthropic"])
        elif llm_type == "ollama":
            llm = OllamaLLM(model="llama3.2:3b")
        else:
            raise HTTPException(status_code=400, detail="Invalid LLM type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting LLM: {str(e)}")
    return {"message": f"LLM set to {llm_type}", "current_llm": llm_type}

@app.post("/set_api_key/{provider}")
async def set_api_key(provider: str, key: str = Form(...)):
    if provider not in ["openai", "anthropic"]:
        raise HTTPException(status_code=400, detail="Invalid API key provider")
    api_keys[provider] = key
    # Reinitialize the embedding function with the new API key if it's for OpenAI
    if provider == "openai":
        global embedding_function, vectorstore
        embedding_function = OpenAIEmbeddings(api_key=api_keys["openai"])
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return {"message": f"{provider.capitalize()} API key set successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)