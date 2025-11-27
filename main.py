import os
import pickle
import uvicorn
import numpy as np
import faiss
import networkx as nx
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# DEPENDENCIES AND SETUP
# ---------------------------------------------------------
# To run this code, you need to install the following:
# pip install fastapi uvicorn networkx faiss-cpu sentence-transformers numpy pydantic
#
# To start the server:
# python main.py

app = FastAPI(title="Vector + Graph Native DB Prototype")

# Constants for persistence
DB_FOLDER = "db_data"
GRAPH_FILE = os.path.join(DB_FOLDER, "graph.pkl")
INDEX_FILE = os.path.join(DB_FOLDER, "vector.index")
META_FILE = os.path.join(DB_FOLDER, "metadata.pkl")

# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------

class NodeItem(BaseModel):
    id: str
    text: str
    type: str = "generic"

class EdgeItem(BaseModel):
    source: str
    target: str
    type: str = "related_to"

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    connections: List[str]

# ---------------------------------------------------------
# CORE DATABASE LOGIC
# ---------------------------------------------------------

class VectorGraphDB:
    def __init__(self):
        # 1. Embedding Model (Small & Fast)
        print("Loading Embedding Model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2

        # 2. Graph Storage (NetworkX)
        self.graph = nx.Graph()

        # 3. Vector Storage (FAISS)
        # using IndexFlatL2 for exact Euclidean search
        self.index = faiss.IndexFlatL2(self.dimension)

        # 4. ID Mappings
        # FAISS uses integer IDs. We need to map:
        # Int ID (FAISS) <-> String ID (User/NetworkX)
        self.int_to_id: Dict[int, str] = {}
        self.id_to_int: Dict[str, int] = {}
        self.next_int_id = 0

        # Attempt to load existing data
        self.load_db()

    def _get_embedding(self, text: str):
        # Generate embedding and normalize for consistency
        vector = self.encoder.encode([text])[0]
        # FAISS expects float32
        return np.array([vector], dtype='float32')

    def add_node(self, node: NodeItem):
        if node.id in self.id_to_int:
            raise HTTPException(status_code=400, detail=f"Node {node.id} already exists")

        # 1. Vectorize
        vector = self._get_embedding(node.text)

        # 2. Add to FAISS
        self.index.add(vector)
        
        # 3. Update Mappings
        curr_int_id = self.next_int_id
        self.int_to_id[curr_int_id] = node.id
        self.id_to_int[node.id] = curr_int_id
        self.next_int_id += 1

        # 4. Add to NetworkX (Store Metadata)
        self.graph.add_node(
            node.id, 
            text=node.text, 
            type=node.type,
            faiss_id=curr_int_id
        )
        return {"status": "success", "id": node.id, "faiss_id": curr_int_id}

    def add_edge(self, edge: EdgeItem):
        # Ensure both nodes exist
        if not self.graph.has_node(edge.source) or not self.graph.has_node(edge.target):
            raise HTTPException(status_code=404, detail="Source or Target node not found")
        
        self.graph.add_edge(edge.source, edge.target, relation=edge.type)
        return {"status": "success", "edge": f"{edge.source} -> {edge.target}"}

    def hybrid_search(self, query_text: str, top_k: int = 5):
        """
        Performs a vector search first, then re-ranks based on graph connectivity.
        Logic: If a node is semantically similar AND connected to other semantically 
        similar nodes, it gets a score boost.
        """
        if self.index.ntotal == 0:
            return []

        # 1. Vector Search (Fetch more candidates than top_k to allow for graph re-ranking)
        search_vector = self._get_embedding(query_text)
        candidate_k = min(self.index.ntotal, 50) # Fetch top 50 candidates
        distances, indices = self.index.search(search_vector, candidate_k)

        # 2. Normalize Vector Scores
        # Convert L2 distance to a similarity score (0 to 1)
        # Simple inversion: 1 / (1 + distance)
        candidates = []
        found_ids = set()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            node_id = self.int_to_id.get(idx)
            if not node_id: continue
            
            dist = distances[0][i]
            sim_score = 1 / (1 + dist)
            
            candidates.append({
                "id": node_id,
                "base_score": sim_score,
                "final_score": sim_score
            })
            found_ids.add(node_id)

        # 3. Graph Logic: "Neighbor Boost"
        # If a candidate is connected to another candidate in the `found_ids` set,
        # it implies a "cluster" of relevance. Boost the score.
        
        for candidate in candidates:
            node_id = candidate["id"]
            # Get neighbors from NetworkX
            neighbors = list(self.graph.neighbors(node_id))
            
            # Count how many neighbors are also in the top vector results
            relevant_neighbors = [n for n in neighbors if n in found_ids]
            
            # Boost logic: +5% score for every relevant neighbor (capped at 50% boost)
            boost_factor = 1.0 + min(len(relevant_neighbors) * 0.05, 0.5)
            
            candidate["final_score"] = candidate["base_score"] * boost_factor
            candidate["relevant_connections"] = len(relevant_neighbors)

        # 4. Sort by Final Score and Slice
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = candidates[:top_k]

        # 5. Format Output
        formatted_results = []
        for res in top_results:
            node_data = self.graph.nodes[res["id"]]
            neighbors = list(self.graph.neighbors(res["id"]))
            formatted_results.append(SearchResult(
                id=res["id"],
                text=node_data.get("text", ""),
                score=round(res["final_score"], 4),
                connections=neighbors
            ))

        return formatted_results

    def save_db(self):
        if not os.path.exists(DB_FOLDER):
            os.makedirs(DB_FOLDER)
        
        # Save FAISS Index
        faiss.write_index(self.index, INDEX_FILE)
        
        # Save Graph
        with open(GRAPH_FILE, 'wb') as f:
            pickle.dump(self.graph, f)
            
        # Save Mappings (Metadata)
        meta_data = {
            "int_to_id": self.int_to_id,
            "id_to_int": self.id_to_int,
            "next_int_id": self.next_int_id
        }
        with open(META_FILE, 'wb') as f:
            pickle.dump(meta_data, f)
            
        print("Database saved successfully.")

    def load_db(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(GRAPH_FILE) and os.path.exists(META_FILE):
            print("Loading existing database...")
            # Load FAISS
            self.index = faiss.read_index(INDEX_FILE)
            
            # Load Graph
            with open(GRAPH_FILE, 'rb') as f:
                self.graph = pickle.load(f)
                
            # Load Mappings
            with open(META_FILE, 'rb') as f:
                meta = pickle.load(f)
                self.int_to_id = meta["int_to_id"]
                self.id_to_int = meta["id_to_int"]
                self.next_int_id = meta["next_int_id"]
            print(f"Loaded {self.index.ntotal} nodes.")
        else:
            print("No existing database found. Starting fresh.")

# Initialize the DB wrapper
db = VectorGraphDB()

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Vector+Graph DB is running. Visit /docs for Swagger UI."}

@app.post("/nodes")
def create_node(node: NodeItem):
    """
    Ingest a node:
    1. Generates embedding from text.
    2. Stores vector in FAISS.
    3. Stores metadata in NetworkX.
    """
    result = db.add_node(node)
    # Auto-save for prototype safety
    db.save_db() 
    return result

@app.post("/edges")
def create_edge(edge: EdgeItem):
    """
    Create a relationship between two existing nodes.
    """
    result = db.add_edge(edge)
    db.save_db()
    return result

@app.post("/search/hybrid")
def search(query: SearchQuery):
    """
    Perform hybrid search:
    1. FAISS Search for semantic similarity.
    2. NetworkX traversal to boost scores of connected clusters.
    """
    results = db.hybrid_search(query.query, query.top_k)
    return {"results": results}

@app.post("/admin/save")
def manual_save():
    db.save_db()
    return {"status": "saved"}

# ---------------------------------------------------------
# RUNNER
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure the DB folder exists
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)