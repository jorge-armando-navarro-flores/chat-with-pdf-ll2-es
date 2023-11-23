import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama


if __name__ == "__main__":
    print("Hello VectorStore!")

    loader = TextLoader("mediumblogs/mediumblog1")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="orca-mini")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=Ollama(
            model="orca-mini", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        ),
        chain_type="stuff",
        retriever=new_vectorstore.as_retriever(),
        return_source_documents=True
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)

    # llama2 vdim=5120
