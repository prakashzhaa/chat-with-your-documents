import pypdfium2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from  utils import load_config
from llm_chains import load_vectordb, create_embeddings

config = load_config()  

def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list] 


def extract_text_from_pdf(pdf_bytes):
    pdf_file =  pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_num).get_textpage().get_text_range() for page_num in range(len(pdf_file)))


def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"],
                                                    chunk_overlap=config["pdf_text_splitter"]["chunk_overlap"],
                                                    separators=config["pdf_text_splitter"]["seperators"])
    return text_splitter.split_text(texts)
 
def get_documents_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))   
    return documents


def add_documents_to_db(pdf_bytes):
    texts = get_pdf_texts(pdf_bytes)
    documents = get_documents_chunks(texts)
    vectordb = load_vectordb(create_embeddings())
    vectordb.add_documents(documents)
    print("Documents added to DB")

    