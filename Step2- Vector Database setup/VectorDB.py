import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "your-api-key"
'''You can get your api key from openai website.'''



'''loding pdf by PyPDFLoader'''
loader = PyPDFLoader("pdfs\Introduction to Machine Learning with Python ( PDFDrive.com )-min (1) (1).pdf")
pages = loader.load()
print(f"Total pages in pdf : {len(pages)}")



'''Breaking the text in chunks by CharacterTextSplitter'''
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)


'''embedding model'''
embeddings = OpenAIEmbeddings()



'''Setting up Chroma VectorDB'''
db = Chroma.from_documents(texts, OpenAIEmbeddings())



'''Similarity search on ChromaDB'''

query = "What is Machine learning?"
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)