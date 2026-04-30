from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

#Setup Ollama
Settings.llm=Ollama(model="mistral",request_timeout=120.0)
Settings.embed_model=OllamaEmbedding(model_name="mistral")

#Create a sample document
import osos.makedirs("data",exit_ok=True)

with open("data/sample.txt","w")as f:
    f.write("Artificial Intelligence is Transforming the world.Machine Learning is a subset of AI.Python is the most popular language for AI development")

    #Load and index
    documents=SimpleDirectoryReader("data").load_data()
    index=VectorStoreIndex.from_documents(documents)

    #query
    query_engine=index.as_query_engine()
    response=query_engine.query("What is MachineLearning?")
    print(response)