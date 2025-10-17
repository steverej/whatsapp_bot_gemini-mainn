from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Load your text file
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.create_documents([text])

# Create embeddings (no API required)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Save to local Chroma database
db = Chroma.from_documents(docs, embeddings, persist_directory="knowledge_db")
db.persist()

print("✅ Training complete! Knowledge saved in 'knowledge_db'")
