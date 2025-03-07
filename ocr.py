import ocrmypdf
import os

# Ask for user input
input_pdf = input("Enter the path to the PDF file: ").strip()

# Ensure the file exists
if not os.path.isfile(input_pdf):
    print("Error: File not found!")
else:
    # Create output filename by appending '_ocr' before the extension
    base_name, ext = os.path.splitext(input_pdf)
    output_pdf = f"{base_name}_ocr{ext}"

    print(f"Processing OCR... Output file will be: {output_pdf}")

    # Perform OCR
    try:
        ocrmypdf.ocr(input_pdf, output_pdf, deskew=True, force_ocr=True)
        print("OCR process completed successfully!")
    except Exception as e:
        print(f"Error during OCR processing: {e}")
