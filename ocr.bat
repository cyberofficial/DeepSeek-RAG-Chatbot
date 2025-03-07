:: ---- Convert PDF to OCR PDF ----

REM pip install ocrmypdf
REM Requires: https://github.com/UB-Mannheim/tesseract/wiki & https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10040/gs10040w64.exe
call venv\Scripts\activate
python ocr.py
pause