from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Pretend this CSV contains structured college info like "Department, Program, Description, Admission Process, Facilities"
df = pd.read_csv("college_information.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./college_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Department"] + " " + row["Program"] + " " + row["Description"],
            metadata={"admission": row.get("Admission Process", ""), "facility": row.get("Facilities", "")},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="college_info",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
