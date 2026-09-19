import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Papeer — Research Paper Assistant", 
    page_icon="📄", 
    layout="wide"
)

# Initialize session state management
if "sessions" not in st.session_state:
    st.session_state.sessions = {"default_session": "Default Research Chat"}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default_session"
if "messages_dict" not in st.session_state:
    st.session_state.messages_dict = {"default_session": []}

session_id = st.session_state.current_session

# Sidebar Layout matching Himanshu's design
with st.sidebar:
    # New Chat / New Session Buttons
    if st.button("+ New Chat", use_container_width=True):
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.messages_dict[new_id] = []
        st.session_state.current_session = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Sessions")
    
    # Session switcher list
    selected_session = st.selectbox(
        "Active Session",
        options=list(st.session_state.sessions.keys()),
        format_func=lambda x: st.session_state.sessions[x],
        label_visibility="collapsed"
    )
    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        st.rerun()

    if st.button("New Session", type="primary", use_container_width=True):
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.messages_dict[new_id] = []
        st.session_state.current_session = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Documents")
    st.markdown("**Upload Files**")
    
    source_type = st.radio("Choose Source Type", ["File Upload", "Web URL", "ArXiv ID"], label_visibility="collapsed")

    if source_type == "File Upload":
        uploaded_file = st.file_uploader("Upload PDF, TXT, MD", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            docs_dir = os.path.join(os.path.dirname(__file__), "documents")
            os.makedirs(docs_dir, exist_ok=True)
            file_path = os.path.join(docs_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("🚀 Process File", use_container_width=True):
                with st.spinner("Processing chunks & local embeddings..."):
                    try:
                        from backend.vector_store import add_paper
                        from backend.paper_loader import load_and_split_pdf
                        chunked_docs = load_and_split_pdf(file_path, uploaded_file.name)
                        add_paper(chunked_docs, session_id)
                        st.success("File successfully indexed!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif source_type == "Web URL":
        url_input = st.text_input("Enter Web URL")
        if url_input and st.button("🚀 Process URL", use_container_width=True):
            with st.spinner("Scraping and indexing URL..."):
                try:
                    from backend.vector_store import add_paper
                    from backend.paper_loader import load_from_url
                    chunked_docs = load_from_url(url_input)
                    add_paper(chunked_docs, session_id)
                    st.success("URL successfully indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    elif source_type == "ArXiv ID":
        arxiv_input = st.text_input("Enter ArXiv ID (e.g. 2303.08774)")
        if arxiv_input and st.button("🚀 Fetch ArXiv", use_container_width=True):
            with st.spinner("Fetching from ArXiv..."):
                try:
                    from backend.vector_store import add_paper
                    from backend.paper_loader import load_from_arxiv
                    chunked_docs = load_from_arxiv(arxiv_input)
                    add_paper(chunked_docs, session_id)
                    st.success("ArXiv paper indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

# Main Chat Interface matching Himanshu's clean layout
st.markdown("<h1>📁 Papeer — Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "🔍 **Ask questions** from your uploaded papers · ✅ **Verify claims** against recent literature · 🌐 **Search the web** for the latest findings",
    unsafe_allow_html=True
)
st.markdown("---")

# Retrieve current session messages
current_messages = st.session_state.messages_dict[session_id]

if not current_messages:
    st.markdown("<p style='color: gray; text-align: center;'>Upload documents in the sidebar and start chatting below.</p>", unsafe_allow_html=True)

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "state" in message and message["state"]:
            with st.expander("🔍 LangGraph State Inspector"):
                st.json(message["state"])

if query := st.chat_input("Ask about your papers, verify a claim, or search the web..."):
    is_btw = query.startswith("/btw")
    display_query = query[4:].strip() if is_btw else query

    current_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and retrieving context..."):
            try:
                if is_btw:
                    from backend.btw_handler import handle_btw_query
                    response_stream = handle_btw_query(display_query)
                    response_text = st.write_stream(response_stream)
                    current_messages.append({"role": "assistant", "content": response_text})
                else:
                    from backend.rag_graph import app_graph
                    initial_state = {"query": query, "documents": [], "generation": "", "route": ""}
                    
                    final_state = {}
                    for event in app_graph.stream(initial_state):
                        for node_name, node_output in event.items():
                            final_state.update(node_output)

                    response = final_state.get("generation", "No response generated.")
                    st.markdown(response)
                    
                    with st.expander("🔍 LangGraph State Inspector"):
                        st.json(final_state)

                    current_messages.append({
                        "role": "assistant", 
                        "content": response,
                        "state": final_state
                    })
            except Exception as e:
                st.error(f"Error generating response: {e}")