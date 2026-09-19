import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Papeer - Research Paper Assistant", page_icon="📄", layout="wide")
st.title("📄 Papeer: Research Paper Assistant")
st.markdown("Upload research papers via **File, URL, or ArXiv** and chat with them using **Google Gemini 1.5-flash** & LangGraph.")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

# Sidebar for Setup & Multi-Source Paper Loading
with st.sidebar:
    st.header("⚙️ Setup & Paper Loading")
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY missing in .env file!")
    else:
        st.success("✅ Google API Key loaded")

    st.divider()
    source_type = st.radio("Choose Source Type", ["File Upload", "Web URL", "ArXiv ID"])

    if source_type == "File Upload":
        uploaded_file = st.file_uploader("Choose a PDF, TXT, or MD file", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            docs_dir = os.path.join(os.path.dirname(__file__), "documents")
            os.makedirs(docs_dir, exist_ok=True)
            file_path = os.path.join(docs_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("🚀 Process File"):
                with st.spinner("Processing chunks & local embeddings..."):
                    try:
                        from backend.vector_store import add_paper
                        from backend.paper_loader import load_and_split_pdf
                        chunked_docs = load_and_split_pdf(file_path, uploaded_file.name)
                        add_paper(chunked_docs, st.session_state.session_id)
                        st.success("File successfully indexed!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif source_type == "Web URL":
        url_input = st.text_input("Enter Web URL")
        if url_input and st.button("🚀 Process URL"):
            with st.spinner("Scraping and indexing URL content..."):
                try:
                    from backend.vector_store import add_paper
                    from backend.paper_loader import load_from_url
                    chunked_docs = load_from_url(url_input)
                    add_paper(chunked_docs, st.session_state.session_id)
                    st.success("URL successfully indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif source_type == "ArXiv ID":
        arxiv_input = st.text_input("Enter ArXiv ID (e.g. 2303.08774)")
        if arxiv_input and st.button("🚀 Fetch & Index ArXiv Paper"):
            with st.spinner("Fetching from ArXiv and indexing..."):
                try:
                    from backend.vector_store import add_paper
                    from backend.paper_loader import load_from_arxiv
                    chunked_docs = load_from_arxiv(arxiv_input)
                    add_paper(chunked_docs, st.session_state.session_id)
                    st.success("ArXiv paper successfully indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

# Main Chat Interface
st.subheader("💬 Chat Interface")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "state" in message and message["state"]:
            with st.expander("🔍 LangGraph State Inspector"):
                st.json(message["state"])

if query := st.chat_input("Ask a question, use /btw for off-topic, or type 'Verify claim...'"):
    # Check for /btw command
    is_btw = query.startswith("/btw")
    display_query = query[4:].strip() if is_btw else query

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving context..."):
            try:
                if is_btw:
                    from backend.btw_handler import handle_btw_query
                    response_stream = handle_btw_query(display_query)
                    response_text = st.write_stream(response_stream)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    from backend.rag_graph import app_graph
                    initial_state = {"query": query, "documents": [], "generation": "", "route": ""}
                    
                    # Run graph and capture final state for inspector
                    final_state = {}
                    for event in app_graph.stream(initial_state):
                        for node_name, node_output in event.items():
                            final_state.update(node_output)

                    response = final_state.get("generation", "No response generated.")
                    st.markdown(response)
                    
                    with st.expander("🔍 LangGraph State Inspector"):
                        st.json(final_state)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "state": final_state
                    })
            except Exception as e:
                st.error(f"Error generating response: {e}")