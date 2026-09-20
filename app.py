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
if "loaded_docs_dict" not in st.session_state:
    st.session_state.loaded_docs_dict = {"default_session": []}

session_id = st.session_state.current_session

# Exact Sidebar Layout Matching Target Screenshots
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        import uuid
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = f"Session {len(st.session_state.sessions) + 1}"
        st.session_state.messages_dict[new_id] = []
        st.session_state.loaded_docs_dict[new_id] = []
        st.session_state.current_session = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Sessions")
    
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
        st.session_state.loaded_docs_dict[new_id] = []
        st.session_state.current_session = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("### 📄 Documents")
    
    # Upload Files Section
    st.markdown("**Upload Files**")
    uploaded_file = st.file_uploader("Upload PDF, TXT, MD, MARKDOWN", type=["pdf", "txt", "md", "markdown"], label_visibility="collapsed")
    st.caption("200MB per file • PDF, TXT, MD, MARKDOWN")
    
    if st.button("Add Files", use_container_width=True) and uploaded_file is not None:
        docs_dir = os.path.join(os.path.dirname(__file__), "documents")
        os.makedirs(docs_dir, exist_ok=True)
        file_path = os.path.join(docs_dir, uploaded_file.name)

        current_session_docs = st.session_state.loaded_docs_dict.get(session_id, [])
        
        if uploaded_file.name in current_session_docs:
            st.info(f"'{uploaded_file.name}' is already loaded in this session!")
        else:
            file_already_existed = os.path.exists(file_path)
            if not file_already_existed:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            if file_already_existed:
                st.session_state.loaded_docs_dict[session_id].append(uploaded_file.name)
                st.success(f"Reused existing embeddings for: {uploaded_file.name} (Zero re-processing!)")
            else:
                with st.spinner("Processing file & local embeddings..."):
                    try:
                        from backend.vector_store import add_paper
                        from backend.paper_loader import load_and_split_pdf
                        chunked_docs = load_and_split_pdf(file_path, uploaded_file.name)
                        add_paper(chunked_docs, session_id)
                        st.session_state.loaded_docs_dict[session_id].append(uploaded_file.name)
                        st.success(f"Added and indexed: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # Web Pages Section
    st.markdown("**Web Pages**")
    url_input = st.text_area("https://example.com/paper", height=70, label_visibility="collapsed")
    
    if st.button("Load URLs", use_container_width=True) and url_input.strip():
        with st.spinner("Scraping and indexing URL..."):
            try:
                from backend.vector_store import add_paper
                from backend.paper_loader import load_from_url
                chunked_docs = load_from_url(url_input.strip())
                add_paper(chunked_docs, session_id)
                st.session_state.loaded_docs_dict[session_id].append(url_input.strip())
                st.success("URL successfully indexed!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # ArXiv Papers Section
    st.markdown("**ArXiv Papers**")
    arxiv_input = st.text_input("1706.03762 or Attention Is All You Need", label_visibility="collapsed")
    
    if st.button("Load ArXiv Paper", use_container_width=True) and arxiv_input.strip():
        with st.spinner("Fetching from ArXiv..."):
            try:
                from backend.vector_store import add_paper
                from backend.paper_loader import load_from_arxiv
                chunked_docs = load_from_arxiv(arxiv_input.strip())
                add_paper(chunked_docs, session_id)
                st.session_state.loaded_docs_dict[session_id].append(f"ArXiv: {arxiv_input.strip()}")
                st.success("ArXiv paper indexed!")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### Loaded Documents")
    current_docs = st.session_state.loaded_docs_dict.get(session_id, [])
    if current_docs:
        for doc in current_docs:
            st.text(f"• {doc}")
    else:
        st.caption("No documents loaded yet.")

# Main Chat Interface
st.markdown("<h1>📁 Papeer — Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "🔍 **Ask questions** from your uploaded papers · ✅ **Verify claims** against recent literature · 🌐 **Search the web** for the latest findings",
    unsafe_allow_html=True
)
st.markdown("---")

current_messages = st.session_state.messages_dict[session_id]

if not current_messages:
    st.markdown("<p style='color: gray; text-align: center;'>Upload documents in the sidebar and start chatting below.</p>", unsafe_allow_html=True)

# Render stored history messages
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "state" in message and message["state"]:
            with st.expander("🔍 LangGraph State Inspector"):
                st.json(message["state"])

# Handle new user input
if query := st.chat_input("Ask about your papers, verify a claim, or search the web..."):
    # Display user query on screen immediately
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and routing query..."):
            try:
                from backend.rag_graph import app_graph
                initial_state = {"query": query, "session_id": session_id, "documents": [], "generation": "", "route": ""}
                
                final_state = {}
                for event in app_graph.stream(initial_state):
                    for node_name, node_output in event.items():
                        final_state.update(node_output)

                response = final_state.get("generation", "No response generated.")
                route = final_state.get("route", "")
                
                # Render the response
                st.markdown(response)
                
                if route == "command_a":
                    st.caption("🔒 **CommandA Side-Channel**: This exchange is ephemeral and was not saved to your session history.")
                else:
                    with st.expander("🔍 LangGraph State Inspector"):
                        st.json(final_state)

                # EPHEMERAL CHECK: Only append to session history if it's NOT a command_a side-channel query!
                if route != "command_a":
                    current_messages.append({"role": "user", "content": query})
                    current_messages.append({
                        "role": "assistant", 
                        "content": response,
                        "state": final_state
                    })

            except Exception as e:
                st.error(f"Error generating response: {e}")