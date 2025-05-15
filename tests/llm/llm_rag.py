"""
RAG implementation with local Phi-3-mini-4k-instruct-onnx and embeddings
"""

import os
from typing import List
import numpy as np

# LangChain imports
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# HuggingFace and ONNX imports
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, pipeline

# ------------------------------------------------------
# 1. LOAD THE LOCAL PHI-3 MODEL
# ------------------------------------------------------

# Set up paths to the local model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "cpu_and_mobile", "cpu-int4-rtn-block-32-acc-level-4")
print(f"Loading Phi-3 model from: {model_path}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True
)
model = ORTModelForCausalLM.from_pretrained(
    model_id=model_path,
    provider="CPUExecutionProvider",
    trust_remote_code=True
)
model.name_or_path = model_path

# Create the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)

# Create the LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# ------------------------------------------------------
# 2. LOAD THE EMBEDDING MODEL
# ------------------------------------------------------

# Initialize the embedding model - using a small, efficient model
# Options:
# - "BAAI/bge-small-en-v1.5" (385MB, good performance/size ratio)
# - "sentence-transformers/all-MiniLM-L6-v2" (91MB, very small)
# - "intfloat/e5-small-v2" (134MB, good performance)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model loaded")

# ------------------------------------------------------
# 3. CREATE A SAMPLE DOCUMENT COLLECTION
# ------------------------------------------------------

# Sample documents about artificial intelligence
docs = [
    Document(
        page_content="Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals, which involves consciousness and emotionality.",
        metadata={"source": "AI Definition"}
    ),
    Document(
        page_content="Machine learning is a subset of AI focused on building systems that learn from data. Deep learning is a subset of machine learning based on neural networks.",
        metadata={"source": "Machine Learning"}
    ),
    Document(
        page_content="Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. They form the basis of many modern AI systems.",
        metadata={"source": "Neural Networks"}
    ),
    Document(
        page_content="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
        metadata={"source": "NLP"}
    ),
    Document(
        page_content="Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a reward signal.",
        metadata={"source": "Reinforcement Learning"}
    ),
    Document(
        page_content="Computer vision is a field of AI that enables computers to derive meaningful information from digital images, videos and other visual inputs.",
        metadata={"source": "Computer Vision"}
    ),
]

# ------------------------------------------------------
# 4. SPLIT DOCUMENTS AND CREATE VECTOR STORE
# ------------------------------------------------------

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Split the documents
split_docs = text_splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(split_docs)} chunks")

# Create a FAISS vector store from the chunks
vectorstore = FAISS.from_documents(split_docs, embeddings)
print("Vector store created")

# ------------------------------------------------------
# 5. CREATE RAG PROMPT TEMPLATE FOR PHI-3
# ------------------------------------------------------

# Phi-3 specific RAG prompt template
rag_prompt_template = """<|user|>
Use the following context to answer the question. If you don't know the answer based on the context, just say you don't know.

Context:
{context}

Question: {question}
<|assistant|>"""

# Create the prompt
prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "question"]
)

# ------------------------------------------------------
# 6. CREATE RAG CHAIN
# ------------------------------------------------------

# Create the retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" method puts all retrieved docs into one prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 results
    return_source_documents=True,  # Return source docs for transparency
    chain_type_kwargs={"prompt": prompt}  # Use our custom prompt
)

# ------------------------------------------------------
# 7. QUERY FUNCTIONS
# ------------------------------------------------------

def ask_rag(question: str):
    """Query the RAG system with a question"""
    print(f"\nQuestion: {question}")
    
    # Get response from the chain
    response = qa_chain.invoke({"query": question})
    
    # Print the answer
    print("\nAnswer:")
    print(response["result"])
    
    # Print the source documents
    print("\nSources:")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nSource {i+1}: {doc.metadata['source']}")
        print(f"Content: {doc.page_content}")
    
    return response

# ------------------------------------------------------
# 8. FUNCTION TO LOAD CUSTOM DOCUMENTS
# ------------------------------------------------------

def load_docs_from_files(file_paths: List[str]):
    """Load documents from files"""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    
    all_docs = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} document(s) from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_docs

def create_rag_from_files(file_paths: List[str]):
    """Create a new RAG system from the provided files"""
    # Load documents
    loaded_docs = load_docs_from_files(file_paths)
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(loaded_docs)
    
    # Create vector store
    new_vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    # Create QA chain
    new_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=new_vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}  # Use our custom prompt
    )
    
    return new_qa_chain

# ------------------------------------------------------
# 9. EXAMPLE USAGE
# ------------------------------------------------------

if __name__ == "__main__":
    # Test with sample questions
    print("\n===== RAG System Demo =====")
    
    # Example 1: Basic retrieval
    ask_rag("What is the difference between machine learning and deep learning?")
    
    # Example 2: Testing knowledge boundaries
    ask_rag("What are the key components of a neural network?")
    
    # Example 3: Question outside the knowledge base
    ask_rag("What is the capital of France?")
    
    print("\n===== Demo Complete =====")

# ------------------------------------------------------
# 10. SAVE AND LOAD VECTOR STORE FOR FUTURE USE
# ------------------------------------------------------

def save_vectorstore(vectorstore, directory="faiss_index"):
    """Save the FAISS vector store to disk"""
    vectorstore.save_local(directory)
    print(f"Vector store saved to {directory}")

def load_vectorstore(directory="faiss_index"):
    """Load a FAISS vector store from disk"""
    if os.path.exists(directory):
        loaded_vectorstore = FAISS.load_local(directory, embeddings)
        print(f"Vector store loaded from {directory}")
        return loaded_vectorstore
    else:
        print(f"No vector store found at {directory}")
        return None