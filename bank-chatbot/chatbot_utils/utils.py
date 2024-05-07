from PyPDF2 import PdfReader
import json
def read_pdf_file(pdf_path):
    pages_text = []
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        document = ""
        for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                document+=page.extract_text()
    return document
def read_json_file(file_path):
    """
    Read data from a JSON file.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        a string json file
    """
    with open(file_path, 'r') as file:
        json_string = file.read()
    return json_string

def read_text_file(file):
    with open(file, "r") as file:
        file_contents = file.read()
    return file_contents
    