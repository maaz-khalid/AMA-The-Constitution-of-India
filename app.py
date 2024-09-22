import os
import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
# from langchain_objectbox.vectorstores import ObjectBox

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

st.title("AMA : Constitution of India 📔")

llm =ChatGroq(groq_api_key=groq_api_key,model="Llama3-70b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
Also give me the document name where you are referring from.
<context>
{context}
<context>
Questions: {input}
"""
)

prompt3 = ChatPromptTemplate.from_template(
    """
You are a law expert who is well versed with constituion of india.
Please provide the most accurate response after reading the entire Database based on the question.
First answer the question using context and if there is no answer in the context then give a general answer.
Dont say that you dont know, you have to give the answer at all cost.
<context>
{context}
<context>
Questions: {input}
"""
)

# Vector Embedding and ObjectBox Vectorstore DB

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./constitution")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        for i, doc in enumerate(st.session_state.docs):
            doc.metadata["name"] = f"Document_{i+1}"

        for doc in st.session_state.final_documents:
            doc.metadata["name"] = doc.metadata.get("name", "Unknown Document")

        index_name = "coi-index"

        # Connect to Pinecone index and insert the chunked docs as contents
        st.session_state.vectors = PineconeVectorStore.from_documents(st.session_state.final_documents, st.session_state.embeddings, index_name=index_name)    

        # st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768)

col1, col2 = st.columns(2)

with col1:
    if st.button("Read the Documents"):
        vector_embedding()
        st.write("Memory Updated") 

input_prompt = st.text_input("Enter your question:")

if st.button("Answer"):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(f"Document Name: {doc.metadata.get('name', 'Unknown Document')}")
            st.write(doc.page_content)
            st.write("--------------------------------")

input_prompt2 = st.text_input("Follow Up Question")

if st.button("Go"):
    document_chain = create_stuff_documents_chain(llm, prompt3)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt2})

    print("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(f"Document Name: {doc.metadata.get('name', 'Unknown Document')}")
            st.write(doc.page_content)
            st.write("--------------------------------")

if st.button("Project Information"): 
    tech_summary = """
    **Technologies Used:**
    - **Language:** Python
    - **Libraries:** Streamlit, Langchain, dotenv, PyPDF2
    - **Model Used:** ChatGroq (Llama3-70b-8192)
    - **APIs Used:** GOOGLE API, GROQ API, PINECONE API
    """
    st.markdown(tech_summary)

# Add this at the end of your Streamlit app code to display the footer
st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
    }
</style>
<div class="footer">
    <p>Designed by <a href="https://www.linkedin.com/in/maaz-khalid-73bba21b3/" target="_blank">Maaz Khalid</a></p>
</div>
""", unsafe_allow_html=True)