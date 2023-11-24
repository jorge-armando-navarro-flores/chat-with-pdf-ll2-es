import os
import multiprocessing
import torch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama


def process_chunk(chunk, embeddings):
    return embeddings.embed(chunk)


if __name__ == "__main__":
    print("Hello VectorStore!")

    loader = TextLoader("mediumblogs/mediumblog1")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="orca-mini")
    vectorstore = FAISS()

    # Specify the number of GPUs to use
    num_gpus = torch.cuda.device_count()

    # Create a pool of processes for parallel processing
    pool = multiprocessing.Pool(processes=num_gpus)

    # Calculate the number of chunks per GPU
    chunks_per_gpu = len(texts) // num_gpus

    # Process the chunks in parallel using multiple GPUs
    embeddings_list = []
    for i in range(num_gpus):
        start_idx = i * chunks_per_gpu
        end_idx = start_idx + chunks_per_gpu if i < num_gpus - 1 else len(texts)
        gpu_texts = texts[start_idx:end_idx]
        embeddings_list.extend(pool.starmap(process_chunk, [(chunk, embeddings) for chunk in gpu_texts]))

    # Add the embeddings to the vector store
    for i, embedding in enumerate(embeddings_list):
        vectorstore.add_document(embedding, texts[i])

    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react")

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

    # llama2 vdim=5120" beginner friendly