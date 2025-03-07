import streamlit as st
import os
import time
import uuid

# Try to import ocrmypdf, disable OCR functionality if not available
ocr_available = True
try:
    import ocrmypdf
except ImportError:
    ocr_available = False

# Streamlit configuration
st.set_page_config(page_title="PDF OCR Tool", layout="wide")

# Custom CSS to match main app style
st.markdown("""
    <style>
        .stApp { background-color: #black; }
        h1 { color: #00FF99; text-align: center; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

# Create temp directory if it doesn't exist
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Main title
st.title("📑 PDF OCR Tool")
st.caption("Upload PDF files for OCR processing")

# Initialize session state for processed file and current file
if "ocr_processed_pdf" not in st.session_state:
    st.session_state.ocr_processed_pdf = None
if "ocr_processed_name" not in st.session_state:
    st.session_state.ocr_processed_name = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Main interface
if ocr_available:
    # Use a key that changes when clearing to force file uploader reset
    upload_key = "pdf_upload_" + str(int(time.time())) if st.session_state.get('clearing') else "pdf_upload"
    new_file = st.file_uploader("Upload PDF for OCR", type=["pdf"], key=upload_key)
    
    # Handle new file upload
    if new_file is not None and new_file != st.session_state.current_file:
        # Clear previous file data
        st.session_state.ocr_processed_pdf = None
        st.session_state.ocr_processed_name = None
        st.session_state.current_file = new_file
    
    ocr_file = st.session_state.current_file
else:
    st.error("PDF OCR functionality is not available. Please install ocrmypdf package to enable this feature.")

if ocr_available and ocr_file and (st.session_state.ocr_processed_name != ocr_file.name) and not st.session_state.get('clearing', False):
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update status
        status_text.text("Starting OCR process...")
        progress_bar.progress(10)
        
        # Generate unique filenames with UUID
        unique_id = str(uuid.uuid4())
        temp_input = os.path.join(TEMP_DIR, f"{unique_id}_input_{ocr_file.name}")
        temp_output = os.path.join(TEMP_DIR, f"{unique_id}_output_{ocr_file.name}")
        
        # Save uploaded file
        with open(temp_input, "wb") as f:
            f.write(ocr_file.getbuffer())
        
        if not os.path.exists(temp_input):
            raise Exception("Input file was removed before processing could begin")
            
        progress_bar.progress(30)
        status_text.text("Processing OCR (this may take a while)...")
        
        # Process OCR
        ocrmypdf.ocr(temp_input, temp_output, deskew=True, force_ocr=True)
        
        # Check if input file still exists after OCR
        if not os.path.exists(temp_input):
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise Exception("Input file was removed during processing")
            
        progress_bar.progress(70)
        status_text.text("Finalizing...")
        
        # Read processed file for download
        if os.path.exists(temp_output):
            with open(temp_output, "rb") as f:
                st.session_state.ocr_processed_pdf = f.read()
                st.session_state.ocr_processed_name = ocr_file.name
        else:
            raise Exception("Output file was not created successfully")
        
        # Clean up temp files
        try:
            if os.path.exists(temp_input):
                os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e:
            st.warning(f"Warning: Could not remove temporary files: {str(e)}")
        
        progress_bar.progress(100)
        status_text.text("OCR processing complete!")
        time.sleep(1)  # Brief pause to show completion
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ PDF successfully processed with OCR!")
        
    except Exception as e:
        st.error(f"Error during OCR processing: {str(e)}")
        st.session_state.ocr_processed_pdf = None
        st.session_state.ocr_processed_name = None
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()

# Show download button if we have processed content
if st.session_state.ocr_processed_pdf is not None:
    st.download_button(
        label="📥 Download OCR'd PDF",
        data=st.session_state.ocr_processed_pdf,
        file_name=f"ocr_{st.session_state.ocr_processed_name}",
        mime="application/pdf"
    )
    
    # Add a clear button to reset the state
    if st.button("Clear"):
        # Set clearing flag and reset all states
        st.session_state.clearing = True
        st.session_state.ocr_processed_pdf = None
        st.session_state.ocr_processed_name = None
        st.session_state.current_file = None
        # Clear the file uploader
        st.session_state["uploaded_files"] = None
        st.rerun()

# Reset clearing flag at the start of next run
if st.session_state.get('clearing'):
    st.session_state.clearing = False