from langchain import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate

embedding_model = HuggingFaceEmbeddings(
    model_name = 'intfloat/e5-small-v2'
)

texts = [
    'text1',
    'text2'
]

db = FAISS.from_texts(texts, embedding_model)

template = """<|user|>
Relevant information:
{context}

Provide a concise answer to the 
"""

prompt = PromptTemplate.from_template(
    template=template
)
prompt.format(context="")



retriever = VectorStoreRetriever(vectorstore=FAISS(...))
retrievalQA = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever)


retriever = ...  # Your retriever
llm = ChatOpenAI()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

chain.invoke({"input": query})