import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore




load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    #TextLoader
    loader = TextLoader("/Users/sugiharto/Desktop/langchain-course/mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f'created {len(texts)} chunks')

    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001'
    )

    print("Ingesting...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ['INDEX_NAME'],
    )

    print('Finish')

