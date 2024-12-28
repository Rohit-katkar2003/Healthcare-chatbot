
from langchain.document_loaders import PyPDFLoader ,DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import pickle

def load_pdf(data): 
    loader = DirectoryLoader(data, 
                             glob="*.pdf", 
                             loader_cls=PyPDFLoader) 
    document = loader.load() 
    return document  


def text_split(extracted_data) : 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=20) 
    text_chunks = text_splitter.split_documents(extracted_data) 
    return text_chunks 


def download_hugging_face_embeddings(): 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings  

def load_faiss_with_metadata(embeddings, metadata_filename="faiss_metadata.pkl"):
    # Load FAISS index from file
    index = faiss.read_index("faiss_index")
    
    # Load metadata (docstore and index_to_docstore_id)
    with open(metadata_filename, "rb") as f:
        metadata = pickle.load(f)
    
    docstore = metadata["docstore"]
    index_to_docstore_id = metadata["index_to_docstore_id"]
    
    # Recreate the FAISS vectorstore with the loaded index, docstore, and index_to_docstore_id
    vectorstore = FAISS(embedding_function=embeddings, index=index, 
                        docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    
    print("FAISS index and metadata loaded successfully!")
    return vectorstore 


