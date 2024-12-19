from paths import AGENTS_DIR
VERSION = "v4"

MERMAID_DIR = AGENTS_DIR / "graph_mermaids"
# Create the directory if it doesn't exist
MERMAID_DIR.mkdir(parents=True, exist_ok=True)


def save_graph(app = None, version = VERSION):
    filename = f"agent_graph_{version}.png"
    try:
        if app is None:
            from tangents.tan_graph import create_workflow
            app = create_workflow()
        
        MERMAID_DIR.mkdir(parents=True, exist_ok=True)
        file_path = MERMAID_DIR / filename
        app.get_graph().draw_mermaid_png(output_file_path=file_path)
        print(f"Graph saved to {file_path}")
    except Exception as e:
        print("Failed to save graph")
        print(e)
        pass

if __name__ == "__main__":
    save_graph()
