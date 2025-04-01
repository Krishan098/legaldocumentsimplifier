Legal Text Summary Generator

Overview

The Legal Text Summary Generator is a web application that utilizes a fine-tuned BART model (BillSum dataset) to generate concise summaries of legal documents. The app provides two input methods:

Text Input - Users can enter legal text manually.

PDF Upload - Users can upload a PDF document, and the app will extract and summarize its content.

The application is built using Streamlit for an interactive UI and Hugging Face Transformers for text summarization.

Features

Generate summaries for legal text using a BART-based model.

Extract text from PDF documents for summarization.

User-friendly interface with tabs for different input methods.

Provides warnings for scanned or protected PDFs where text extraction may not be possible.

Built-in ROUGE metric support for potential evaluation.

Installation

To run the application locally, follow these steps:

1. Clone the Repository

git clone https://github.com/krishan098/legaldocumentsimplifier.git
cd legaldocumentsimplifier

2. Install Dependencies

Ensure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run app.py

Usage

Enter text manually in the provided text area and click "Generate Summary."

Upload a PDF document and click "Extract and Summarize" to generate a summary.

Expand the "View Extracted Text" section to preview the extracted content from a PDF.

Read the summary output under the "Generated Summary" section.

Dependencies

torch - For deep learning computations.

transformers - For BART-based text summarization.

streamlit - For the web UI.

rouge_score - For summary evaluation.

PyPDF2 - For extracting text from PDFs.

Model Details

The summarization model used in this project is BART-large-CNN fine-tuned on BillSum, a dataset specialized in legal and legislative documents.

Limitations

The model works best on well-formatted legal documents.

Scanned PDFs may not be processed correctly as they require OCR.

Summarization may vary based on the length and complexity of input text.

Future Improvements

Implement OCR support for scanned PDFs.

Provide an option for customizable summary length.

Enhance the text preprocessing pipeline for better summarization results.

License

This project is licensed under the MIT License.

Contact

For questions or contributions, feel free to open an issue or reach out!

Author: Krishan Mittal
GitHub: krishan098
Email: krishanmittal798@gmail.com
![image](https://github.com/user-attachments/assets/c4b233ee-e09b-47a0-8623-ad5a05caa39b)
![image](https://github.com/user-attachments/assets/6b41362d-82b9-4a5a-af07-9511e1949199)
![image](https://github.com/user-attachments/assets/eace5b30-b497-4c02-a8d1-01786bcd54c6)
![image](https://github.com/user-attachments/assets/e779df9b-723b-40c5-9227-0f8f950bbffc)


