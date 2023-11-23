import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import pinecone

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="gcp-starter",
)

if __name__ == "__main__":
    print("Hello VectorStore!")

    loader = TextLoader("mediumblogs/mediumblog1")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="orca-mini")
    docsearch = Pinecone.from_documents(
        texts,
        embeddings,
        index_name="langchainpinecone"
    )

    qa = RetrievalQA.from_chain_type(
        llm=Ollama(
            model="orca-mini", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)

    # llama2 vdim=5120
