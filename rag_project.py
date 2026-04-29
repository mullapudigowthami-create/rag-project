from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

Settings.llm = Ollama(model="mistral", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

os.makedirs("data", exist_ok=True)
with open("data/sample.txt", "w") as f:
    f.write("Artificial Intelligence is transforming the world. Machine Learning is a subset of AI. Python is the most popular language for AI development.")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What is Machine Learning?")
print(response)
resopnse = query_engine.query("What is AI transforming?")
print(response)