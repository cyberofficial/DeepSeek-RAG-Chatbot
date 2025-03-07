call venv\Scripts\activate
:: Set Upload size to 1000M (1GB)
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
streamlit run ocr_app.py