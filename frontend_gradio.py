import gradio as gr
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000"

def add_node(node_id, text, node_type):
    """Sends data to your backend to create a node."""
    if not node_id or not text:
        return "❌ Error: ID and Text are required."
    
    payload = {"id": node_id, "text": text, "type": node_type}
    try:
        response = requests.post(f"{API_URL}/nodes", json=payload)
        if response.status_code == 200:
            return f"✅ Success! Node '{node_id}' created."
        else:
            return f"❌ Error: {response.text}"
    except Exception as e:
        return f"❌ Connection Error: Is the backend running? ({e})"

def add_edge(source, target, relation):
    """Sends data to your backend to link two nodes."""
    if not source or not target:
        return "❌ Error: Source and Target IDs are required."
        
    payload = {"source": source, "target": target, "type": relation}
    try:
        response = requests.post(f"{API_URL}/edges", json=payload)
        if response.status_code == 200:
            return f"✅ Success! Linked '{source}' -> '{target}'."
        else:
            return f"❌ Error: {response.text}"
    except Exception as e:
        return f"❌ Connection Error: Is the backend running? ({e})"

def search_hybrid(query):
    """Asks the backend for hybrid search results."""
    if not query:
        return "Please enter a query."
        
    payload = {"query": query, "top_k": 5}
    try:
        response = requests.post(f"{API_URL}/search/hybrid", json=payload)
        data = response.json()
        
        # Format the results nicely for the demo
        results = data.get("results", [])
        if not results:
            return "No results found."
            
        output_text = ""
        for i, res in enumerate(results):
            score = res.get('score', 0)
            node_id = res.get('id', 'Unknown')
            content = res.get('text', 'No content')
            output_text += f"{i+1}. [ID: {node_id}] (Score: {score:.2f})\n   Content: {content}\n\n"
            
        return output_text
    except Exception as e:
        return f"❌ Connection Error: {e}"

# --- Build the UI ---
with gr.Blocks(title="Vector + Graph DB Demo") as demo:
    gr.Markdown("# 🚀 Hybrid Vector + Graph Database")
    gr.Markdown("This system uses **FAISS** for vector similarity and **NetworkX** for graph relationships.")
    
    with gr.Tab("1. Add Data (Ingestion)"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Add a Node")
                n_id = gr.Textbox(label="Node ID (Unique Name)", placeholder="e.g., apple_inc")
                n_text = gr.Textbox(label="Node Text (Content)", placeholder="e.g., Apple is a technology company.")
                n_type = gr.Dropdown(["concept", "entity", "document"], label="Type", value="entity")
                btn_node = gr.Button("Create Node", variant="primary")
                out_node = gr.Textbox(label="Status")
                
                btn_node.click(add_node, inputs=[n_id, n_text, n_type], outputs=out_node)
                
            with gr.Column():
                gr.Markdown("### Add a Relationship")
                e_source = gr.Textbox(label="Source Node ID", placeholder="e.g., apple_inc")
                e_target = gr.Textbox(label="Target Node ID", placeholder="e.g., iphone")
                e_type = gr.Textbox(label="Relationship Type", placeholder="e.g., manufactures")
                btn_edge = gr.Button("Link Nodes", variant="primary")
                out_edge = gr.Textbox(label="Status")
                
                btn_edge.click(add_edge, inputs=[e_source, e_target, e_type], outputs=out_edge)

    with gr.Tab("2. Hybrid Search"):
        gr.Markdown("### Search your Knowledge Graph")
        search_box = gr.Textbox(label="Search Query", placeholder="Enter your question...")
        btn_search = gr.Button("Run Hybrid Search", variant="primary")
        search_results = gr.TextArea(label="Ranked Results", lines=10)
        
        btn_search.click(search_hybrid, inputs=search_box, outputs=search_results)

# Launch the app
if __name__ == "__main__":
    demo.launch()