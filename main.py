import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()
print("Initializing Components...")

#create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001'
    )

#create llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

#connect langchain to your pinecone index
vector_store = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings,
)

#retriever
#it converts pinecone into a retriever
# When a question comes in,
    # LangChain:
    # Embeds the question
    # Searches Pinecone
    # Returns the 3 most similar chunks
retriever = vector_store.as_retriever(search_kwargs={'k':3})

#tells gemini how to use the retrieved document
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves document

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    #step 1: retrieve relevant documents
    #this will give us the most relevant document based on the query
    #list of 3 langchain document
    docs = retriever.invoke(query)

    #step 2: format documents into a string
    context = format_docs(docs)

    #step 3: format the prompt with context and question
    messages = prompt_template.format_messages(
        context=context,
        question=query,
    )

    #step 4: invoke LLM with the formatted messages
    response=llm.invoke(messages)

    #step 5: return the context
    return response.content

# ========================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ========================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with ("question":"...")

    Advantages over non-LCEL approach
    - Declarative and composable: easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools

    """
    retrieval_chain = (
        RunnablePassthrough.assign(
         context=itemgetter('question') | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain




if __name__ == '__main__':
    print("Retrieving...")

    query = "what is the Pinecone in machine larning?"


    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 1: Without LCEL")
    # print("=" * 70)
    # result_without_lcel = retrieval_chain_without_lcel(query)
    # print("\nAnswer:")
    # print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: WITH LCEL")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({'question': query})
    print(result_with_lcel)







