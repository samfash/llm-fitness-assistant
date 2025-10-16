import chromadb
from sentence_transformers import SentenceTransformer

# Initialize embedding model and Chroma DB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# Add your knowledge base (this could be a list of paragraphs, diet plans, etc.)
docs = [
    "Protein-rich meals help repair muscle tissue after workouts.",
    "Hydration improves recovery and energy levels.",
    "Cardio workouts like running and cycling improve heart health.",
    "Resistance training builds muscle strength and tone.",
    "Eating complex carbs before a workout provides sustained energy."
]

# Add them to the vector store (once)
for i, d in enumerate(docs):
    collection.add(
        ids=[f"doc_{i}"],
        documents=[d],
        embeddings=[embedding_model.encode(d).tolist()]
    )

def retrieve_context(query, top_k=2):
    """Retrieve top similar documents based on semantic similarity."""
    query_emb = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    # Combine retrieved docs into one context string
    if results["documents"]:
        context = " ".join(results["documents"][0])
    else:
        context = ""
    return context
