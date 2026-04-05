import sys
import json
import hashlib
import chromadb

DB_PATH = "./chroma_db"

def _make_id(topic: str) -> str:
    return hashlib.md5(topic.lower().strip().encode()).hexdigest()

def main():
    data = json.loads(sys.stdin.read())
    action = data["action"]

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name="research_memory",
        metadata={"hnsw:space": "cosine"}
    )

    if action == "store":
        doc_id = _make_id(data["topic"])
        collection.upsert(
            ids=[doc_id],
            documents=[data["research"]],
            metadatas=[{"topic": data["topic"], "summary": data["summary"]}]
        )

    elif action == "retrieve":
        count = collection.count()
        if count == 0:
            print(json.dumps([]))
            return
        results = collection.query(
            query_texts=[data["topic"]],
            n_results=min(data["n_results"], count)
        )
        similar = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            if similarity > 0.2:
                similar.append({
                    "topic": results["metadatas"][0][i]["topic"],
                    "summary": results["metadatas"][0][i]["summary"],
                    "similarity": round(similarity, 3)
                })
        print(json.dumps(similar))

if __name__ == "__main__":
    main()