import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "Your_openai_api_key"
'''You can get your api key from openai website.'''

'''loding pdf by PyPDFLoader'''
loader = PyPDFLoader("Introduction to Machine Learning with Python ( PDFDrive.com )-min (1) (1).pdf")
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


'''Now as our text splitted in and stored in list where each element is 1000 charaters of pdf. now we just 
the first 1000 charaters and convert them into vectors.'''
embeddings = OpenAIEmbeddings().embed_query(texts[0].page_content)
print(embeddings)


'''Further we will discuss how to embedded this whole pdf and store it to vector data base'''
