import streamlit as st
import requests
import json
import time
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import torch
import os
from io import BytesIO
from dotenv import load_dotenv, find_dotenv

# Try to import ocrmypdf, disable OCR functionality if not available
ocr_available = True
try:
    import ocrmypdf
except ImportError:
    ocr_available = False

# Custom class to mimic StreamlitUploadedFile
class CustomUploadedFile:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]  # Fix for torch classes not found error
load_dotenv(find_dotenv())  # Loads .env file contents into the application based on key-value pairs defined therein, making them accessible via 'os' module functions like os.getenv().

OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_API_URL = f"{OLLAMA_BASE_URL}/api/generate"
MODEL= os.getenv("MODEL", "deepseek-r1:7b")                                                      #Make sure you have it installed in ollama
EMBEDDINGS_MODEL = "mxbai-embed-large"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

reranker = None                                                        # 🚀 Initialize Cross-Encoder (Reranker) at the global level 
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")


st.set_page_config(page_title="DeepGraph RAG-Pro", layout="wide")      # ✅ Streamlit configuration

# Add MathJax for LaTeX support
st.markdown("""
    <script type="text/javascript">
        window.MathJax = {
            tex: {
                inlineMath: [['$','$']],
                displayMath: [['$$','$$']],
                processEscapes: true,
                processEnvironments: true,
                packages: ['base', 'ams', 'noerrors', 'noundefined']
            },
            svg: {
                fontCache: 'global'
            },
            startup: {
                typeset: true
            }
        };
    </script>
    <script type="text/javascript" id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
    </script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
        .st-emotion-cache-bm2z3a { background-color: black; }
        details {
            background-color: black;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            display: block;
        }
        .st-emotion-cache-acwcvw{
            background-color: gray;   
        }
        details summary {
            cursor: pointer;
            color: #00AAFF;
            font-weight: bold;
        }
        .thinking-section summary {
            color: #808080;
        }
        details[open] summary {
            margin-bottom: 10px;
        }
        details pre {
            color: #e0e0e0;
            background-color: #1e1e1e;
            padding: 8px;
            border-radius: 4px;
            margin: 5px 0;
            overflow-x: auto;
        }
    </style>
""", unsafe_allow_html=True)


                                                                                    # Manage Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "ocr_processed_pdf" not in st.session_state:
    st.session_state.ocr_processed_pdf = None
if "ocr_processed_name" not in st.session_state:
    st.session_state.ocr_processed_name = None
if "final_think_times" not in st.session_state:
    st.session_state.final_think_times = {}
if "ocr_available" not in st.session_state:
    st.session_state.ocr_available = ocr_available
if "num_ctx" not in st.session_state:
    st.session_state.num_ctx = 4096
if "chat_history_limit" not in st.session_state:
    st.session_state.chat_history_limit = 5
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "current_files" not in st.session_state:
    st.session_state.current_files = []

with st.sidebar:                                                                        # 📁 Sidebar
    st.header("📁 Document Management")
    # Get whitelist and blacklist from environment
    WHITELIST_EXTENSIONS = set(os.getenv('WHITELIST_EXTENSIONS', '').split(',')) if os.getenv('WHITELIST_EXTENSIONS') else None
    BLOCKED_EXTENSIONS = set(os.getenv('BLOCKED_EXTENSIONS', '.exe,.dll,.so,.dylib,.bat,.cmd,.sh,.com,.msi,.sys,.vbs,.ps1,.jar,.bin').split(','))

    uploaded_files = st.file_uploader(
        "Upload any document",
        type=None,  # Allow all file types
        accept_multiple_files=True
    )
    
    # Clear current_files if no files are uploaded
    if not uploaded_files:
        st.session_state.current_files = []
    # Process files only if they've changed
    elif [f.name for f in uploaded_files] != st.session_state.current_files:
        # Update list of current files
        st.session_state.current_files = [f.name for f in uploaded_files]
        # Filter files based on extensions
        if WHITELIST_EXTENSIONS:
            # If whitelist is set, only allow those extensions
            allowed_files = [f for f in uploaded_files if os.path.splitext(f.name.lower())[1] in WHITELIST_EXTENSIONS]
            rejected_files = [f.name for f in uploaded_files if os.path.splitext(f.name.lower())[1] not in WHITELIST_EXTENSIONS]
            if rejected_files:
                st.error(f"❌ Only files with these extensions are allowed: {', '.join(WHITELIST_EXTENSIONS)}\nRejected files: {', '.join(rejected_files)}")
            uploaded_files = allowed_files
        else:
            # Use blacklist if no whitelist is set
            blocked_files = [f.name for f in uploaded_files if os.path.splitext(f.name.lower())[1] in BLOCKED_EXTENSIONS]
            if blocked_files:
                st.error(f"❌ Cannot process executable or binary files for security reasons: {', '.join(blocked_files)}")
                st.session_state.documents_loaded = False
                uploaded_files = [f for f in uploaded_files if os.path.splitext(f.name.lower())[1] not in BLOCKED_EXTENSIONS]
            
        if uploaded_files:  # Process remaining files if any
            # Create columns for progress bar and status
            progress_col, status_col = st.columns([2, 3])
            
            with progress_col:
                progress_bar = st.progress(0)
            with status_col:
                status_text = st.empty()
                
            try:
                process_documents(
                    uploaded_files,
                    reranker,
                    EMBEDDINGS_MODEL,
                    OLLAMA_BASE_URL,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                st.success("✅ Documents processed successfully!")
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
            finally:
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
    
    st.markdown("---")
    st.header("📑 PDF OCR")
    if st.session_state.ocr_available:
        ocr_file = st.file_uploader("Upload PDF for OCR", type=["pdf"], key="ocr_upload")
    else:
        st.error("PDF OCR functionality is not available. Please install ocrmypdf package to enable this feature.")
    
    if st.session_state.ocr_available and ocr_file and (st.session_state.ocr_processed_name != ocr_file.name):
        try:
            # Save uploaded file temporarily
            with st.spinner("Processing OCR, this will take a while..."):
                temp_input = f"temp_{ocr_file.name}"
                temp_output = f"temp_ocr_{ocr_file.name}"
                
                # Save uploaded file
                with open(temp_input, "wb") as f:
                    f.write(ocr_file.getbuffer())
                
                # Process OCR
                ocrmypdf.ocr(temp_input, temp_output, deskew=True, force_ocr=True)
                
                # Read processed file for download
                with open(temp_output, "rb") as f:
                    st.session_state.ocr_processed_pdf = f.read()
                    st.session_state.ocr_processed_name = ocr_file.name
                
                # Clean up temp files
                os.remove(temp_input)
                os.remove(temp_output)
        except Exception as e:
            st.error(f"Error during OCR processing: {str(e)}")
            st.session_state.ocr_processed_pdf = None
            st.session_state.ocr_processed_name = None

    # Show download button and use file button if we have processed content
    if st.session_state.ocr_processed_pdf is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download OCR'd PDF",
                data=st.session_state.ocr_processed_pdf,
                file_name=f"ocr_{st.session_state.ocr_processed_name}",
                mime="application/pdf"
            )
        with col2:
            if st.button("Use File for RAG"):
                # Create a CustomUploadedFile instance
                custom_file = CustomUploadedFile(
                    name=st.session_state.ocr_processed_name,
                    data=st.session_state.ocr_processed_pdf
                )
                
                # Process the file with document management
                try:
                    process_documents(
                        [custom_file],
                        reranker,
                        EMBEDDINGS_MODEL,
                        OLLAMA_BASE_URL,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    st.success("OCR'd PDF added to document management!")
                    st.session_state.documents_loaded = True
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")
    
    # Document Processing Settings
    with st.expander("📄 Document Processing Settings", expanded=False):
        st.session_state.chunk_size = st.number_input(
            "Chunk Size",
            value=1000,
            min_value=100,
            max_value=8192,
            help="Size of text chunks in characters. Larger chunks provide more context but use more tokens."
        )
        st.session_state.chunk_overlap = st.number_input(
            "Chunk Overlap",
            value=200,
            min_value=0,
            max_value=st.session_state.chunk_size - 100,
            help="Number of characters that overlap between chunks to maintain context continuity."
        )
    
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("Enable GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.number_input("Max Contexts", value=3, min_value=1, help="Number of context passages to retrieve")
    st.session_state.chat_history_limit = st.number_input("Chat History Messages", value=5, min_value=1, max_value=20, help="Number of previous messages to keep in chat history")
    st.session_state.num_ctx = st.number_input("Context Window Size", value=4096, min_value=512, help="Size of the model's context window (in tokens)")
    
    if st.button("Reset Embeddings"):
        st.session_state.retrieval_pipeline = None
        st.session_state.documents_loaded = False
        st.success("Embeddings have been reset. You can now upload new documents.")
        st.rerun()

    if st.button("Save Chat"):
        if st.session_state.messages:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"chat_session_{timestamp}.txt"
            chat_content = "=== Chat Session Settings ===\n"
            
            # Save RAG settings
            chat_content += "\n[RAG SETTINGS]\n"
            chat_content += f"RAG Enabled: {st.session_state.rag_enabled}\n"
            chat_content += f"HyDE Enabled: {st.session_state.enable_hyde}\n"
            chat_content += f"Neural Reranking: {st.session_state.enable_reranking}\n"
            chat_content += f"GraphRAG: {st.session_state.enable_graph_rag}\n"
            chat_content += f"Temperature: {st.session_state.temperature}\n"
            chat_content += f"Max Contexts: {st.session_state.max_contexts}\n"
            chat_content += f"Context Window Size: {st.session_state.num_ctx}\n"
            chat_content += f"Chunk Size: {st.session_state.chunk_size}\n"
            chat_content += f"Chunk Overlap: {st.session_state.chunk_overlap}\n\n"
            
            chat_content += "=== Chat Messages ===\n\n"
            
            for msg in st.session_state.messages:
                chat_content += f"[{msg['role'].upper()}]:\n{msg['content']}\n"
                
                # Include RAG information for assistant messages
                if msg['role'] == "assistant":
                    # First add the actual RAG context that was used
                    if "rag_context" in msg:
                        chat_content += "\n[RAG CONTEXT USED]:\n"
                        chat_content += msg["rag_context"] + "\n"
                    
                    # Then extract GraphRAG sections
                    content = msg['content']
                    sections = [
                        "GraphRAG Search",
                        "GraphRAG Matches",
                        "Related Nodes",
                        "Retrieved Nodes"
                    ]
                    
                    chat_content += "\n[GRAPHRAG DETAILS]:\n"
                    for section in sections:
                        start = content.find(section)
                        if start != -1:
                            chat_content += f"--- {section} ---\n"
                            # Find the content between this section and the next one
                            next_start = float('inf')
                            for next_section in sections:
                                temp = content.find(next_section, start + len(section))
                                if temp != -1 and temp < next_start:
                                    next_start = temp
                            
                            if next_start == float('inf'):
                                section_content = content[start:]
                            else:
                                section_content = content[start:next_start]
                            
                            chat_content += f"{section_content}\n"
                
                chat_content += "\n" + "="*50 + "\n\n"
            
            # Create download button for the saved chat
            st.download_button(
                label="📥 Download Chat",
                data=chat_content,
                file_name=filename,
                mime="text/plain"
            )
            st.success(f"Chat ready for download!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # 🚀 Footer (Bottom Right in Sidebar) For some Credits :)
    st.sidebar.markdown("""
        
            <b>Original Source Developed by:</b> N Sai Akhil &copy; All Rights Reserved 2025
        
    """, unsafe_allow_html=True)

# 💬 Chat Interface
st.title("🤖 DeepGraph RAG-Pro")
st.caption("Advanced RAG System with GraphRAG, Hybrid Retrieval, Neural Reranking and Chat History")

# Display messages
def format_sections(content):
    formatted = content
    
    # Format thinking sections
    thinking_start = formatted.find("<think>")
    while thinking_start != -1:
        thinking_end = formatted.find("</think>", thinking_start)
        if thinking_end != -1:
            think_content = formatted[thinking_start + 7:thinking_end]
            think_elapsed = st.session_state.final_think_times.get(thinking_start, 0.00)
            formatted = (
                formatted[:thinking_start] +
                f'<details class="thinking-section"><summary>Thinking Process ({think_elapsed:.2f}s)</summary><pre style="white-space: pre-wrap; font-family: monospace;">{think_content}</pre></details>' +
                formatted[thinking_end + 8:]
            )
            thinking_start = formatted.find("<think>")
        else:
            break
    
    # Format GraphRAG sections
    sections = [
        ("Searching GraphRAG for", "GraphRAG Search"),
        ("GraphRAG Matched Nodes", "GraphRAG Matches"),
        ("GraphRAG Retrieved Related Nodes", "Related Nodes"),
        ("GraphRAG Retrieved Nodes", "Retrieved Nodes")
    ]
    
    for search_text, display_text in sections:
        section_start = formatted.find(search_text)
        while section_start != -1:
            next_section = float('inf')
            # Find the start of the next section or the end of the content
            for other_text, _ in sections:
                next_pos = formatted.find(other_text, section_start + len(search_text))
                if next_pos != -1 and next_pos < next_section:
                    next_section = next_pos
                    
            if next_section == float('inf'):
                section_end = len(formatted)
            else:
                section_end = next_section
                
            section_content = formatted[section_start:section_end]
            formatted = (
                formatted[:section_start] +
                f'<details class="graphrag-section"><summary>{display_text}</summary><pre style="white-space: pre-wrap; font-family: monospace;">{section_content}</pre></details>' +
                formatted[section_end:]
            )
            section_start = formatted.find(search_text, section_start + len(display_text))
    
    return formatted

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "assistant":
            content = format_sections(content)
        st.markdown(content, unsafe_allow_html=True)

if prompt := st.chat_input("Ask about your documents..."):
    # Reset think times for new conversation
    st.session_state.final_think_times = {}
    
    # Only use messages from the current conversation
    current_conversation = []
    for msg in st.session_state.messages[::-1]:  # Reverse to get latest messages first
        if msg["role"] == "user" and msg["content"] == prompt:
            break
        content = msg["content"]
        # Strip out RAG-related sections for assistant messages
        if msg["role"] == "assistant":
            for section in ["GraphRAG Search", "GraphRAG Matches", "Related Nodes", "Retrieved Nodes"]:
                section_start = content.find(section)
                if section_start != -1:
                    next_section_start = float('inf')
                    for next_section in ["GraphRAG Search", "GraphRAG Matches", "Related Nodes", "Retrieved Nodes"]:
                        pos = content.find(next_section, section_start + len(section))
                        if pos != -1 and pos < next_section_start:
                            next_section_start = pos
                    if next_section_start == float('inf'):
                        content = content[:section_start].strip()
                    else:
                        content = content[:section_start].strip() + "\n" + content[next_section_start:].strip()
        current_conversation.insert(0, {"role": msg["role"], "content": content})
    
    # Take last n messages from current conversation based on user setting
    chat_history = "\n".join([msg["content"] for msg in current_conversation[-st.session_state.chat_history_limit:]])
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 🚀 Build context
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                docs = retrieve_documents(prompt, OLLAMA_API_URL, MODEL, chat_history)
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
        
        # 🚀 Structured Prompt
        latex_examples = """
1. Inline math: Use $...$ for inline equations
   Examples: $E = mc^2$, $\\sqrt{x^2 + y^2}$, $f(x) = ax^2 + bx + c$
2. Display math: Use $$....$$ for centered equations
   Example:
   $$
   F(x) = \\int_{-\\infty}^x f(t) dt
   $$
3. Common notation:
   - Fractions: $\\frac{numerator}{denominator}$
   - Subscripts: $x_1$, $x_2$, $x_n$
   - Superscripts: $x^2$, $e^x$
   - Greek letters: $\\alpha$, $\\beta$, $\\gamma$, $\\theta$
   - Sums and products: $\\sum_{i=1}^n x_i$, $\\prod_{i=1}^n x_i$"""

        system_prompt = f"""Use the chat history to maintain context:
            Chat History:
            {chat_history}

            Analyze the question and context through these steps:
            1. Identify key entities and relationships
            2. Check for contradictions between sources
            3. Synthesize information from multiple contexts
            4. Formulate a structured response

            LaTeX Instructions:
            Use LaTeX syntax for all mathematical expressions. Here are examples:
            {latex_examples}
            Always wrap mathematical expressions in proper LaTeX delimiters ($...$ or $$...$$).

            Context:
            {context}

            Question: {prompt}
            Answer:"""
        
        # Initialize dictionary to store thinking section start times
        thinking_times = {}
        
        # Stream response
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL,
                "prompt": system_prompt,
                "stream": True,
                "keep_alive": "10m",
                "options": {
                    "temperature": st.session_state.temperature,  # Use dynamic user-selected value
                    "num_ctx": st.session_state.num_ctx
                }
            },
            stream=True
        )
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    
                    # Store start time if this is a new thinking section
                    thinking_start = full_response.find("<think>")
                    if thinking_start != -1 and thinking_start not in thinking_times:
                        thinking_times[thinking_start] = time.time()
                        
                    # Calculate and store timing for any completed thinking sections
                    current_think_start = full_response.find("<think>")
                    while current_think_start != -1:
                        current_think_end = full_response.find("</think>", current_think_start)
                        if current_think_end != -1 and current_think_start not in st.session_state.get('final_think_times', {}):
                            if 'final_think_times' not in st.session_state:
                                st.session_state.final_think_times = {}
                            st.session_state.final_think_times[current_think_start] = time.time() - thinking_times[current_think_start]
                            current_think_start = full_response.find("<think>", current_think_end)
                        else:
                            break
                            
                    # Format both thinking and GraphRAG sections
                    formatted_response = format_sections(full_response)
                    
                    # Display the formatted response
                    response_placeholder.markdown(formatted_response + "▌", unsafe_allow_html=True)
                    
                    # Stop if we detect the end token
                    if data.get("done", False):
                        break
            
            response_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            # Store the original response with think tags and RAG context in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "rag_context": context if context else "No RAG context used"
            })
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})

