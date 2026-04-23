import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def setup_retriever():
    #Load JSON Data
    with open("data/knowledge.json", "r") as f:
        data = json.load(f)
    
    #Convert JSON into a single searchable text block for the retriever
    text_content = f"""
    Company: {data['company_name']}
    Product: {data['product_description']}
    
    Basic Plan: {data['pricing_plans']['Basic Plan']['price']}. Features: {', '.join(data['pricing_plans']['Basic Plan']['features'])}.
    Pro Plan: {data['pricing_plans']['Pro Plan']['price']}. Features: {', '.join(data['pricing_plans']['Pro Plan']['features'])}.
    
    Policies: {', '.join(data['company_policies'])}.
    """
    
    # Creating a langchain Document
    docs = [Document(page_content=text_content)]
    
    # Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 2})