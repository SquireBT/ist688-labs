import sys

# ------------------------------------------------------------ sqlite fix
# A fix for working with ChromaDB on Streamlit Community Cloud.
# This MUST run before chromadb is imported -- chromadb checks the
# sqlite3 version at import time.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

# ------------------------------------------------------------ config
# Paths are relative to this file, not the working directory Streamlit
# was launched from (the app is run from the repo root).
PDF_FOLDER = Path(__file__).parent / 'Lab-04-Data'
CHROMA_PATH = Path(__file__).parent / 'ChromaDB_for_Lab'
COLLECTION_NAME = 'Lab4Collection'
EMBED_MODEL = 'text-embedding-3-small'

# ------------------------------------------------------------ openai client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# ------------------------------------------------------------ extract text from pdf
# This function extracts text from each syllabus
# to pass to add_to_collection
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += (page.extract_text() or '') + '\n'
    return text

# ------------------------------------------------------------ add documents to collection
# collection = ChromaDB collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_to_collection(collection, text, file_name):

    # Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )

    # Get the embedding
    embedding = response.data[0].embedding

    # Add embedding and document to ChromaDB
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{'filename': file_name}]
    )

# ------------------------------------------------------------ populate collection with pdfs
# This function uses extract_text_from_pdf
# and add_to_collection to put syllabi in ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):
    loaded = []
    for pdf_path in sorted(Path(folder_path).glob('*.pdf')):
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            st.warning(f'No extractable text in {pdf_path.name}, skipping.')
            continue
        add_to_collection(collection, text, pdf_path.name)
        loaded.append(pdf_path.name)
    return loaded

# ------------------------------------------------------------ build the vector db
# Stored in st.session_state so the ChromaDB is only created once
def create_chromaDB():

    # Create ChromaDB client
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    # Check if collection is empty and load PDFs
    if collection.count() == 0:
        loaded = load_pdfs_to_collection(PDF_FOLDER, collection)
        st.sidebar.success(f'Embedded {len(loaded)} PDFs into {COLLECTION_NAME}')

    return collection


if 'Lab4_VectorDB' not in st.session_state:
    with st.spinner('Building the Lab4Collection vector database...'):
        st.session_state.Lab4_VectorDB = create_chromaDB()

collection = st.session_state.Lab4_VectorDB

# ------------------------------------------------------------ main app
st.title('Lab 4: Chatbot using RAG')

st.sidebar.caption(f'`{COLLECTION_NAME}`: {collection.count()} documents')

# ------------------------------------------------------------ querying a collection -- only used for testing
# topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

# if topic:
#     client = st.session_state.openai_client
#     response = client.embeddings.create(
#         input=topic,
#         model=EMBED_MODEL)

#     # Get the embedding
#     query_embedding = response.data[0].embedding

#     # Get the text related to this question (this prompt)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3  # The number of closest documents to return
#     )

#     # Display the results
#     st.subheader(f'Results for: {topic}')

#     for i in range(len(results['documents'][0])):
#         doc = results['documents'][0][i]
#         doc_id = results['ids'][0][i]

#         st.write(f'**{i+1}. {doc_id}**')

# else:
#     st.info('Enter a topic in the sidebar to search the collection')

# ------------------------------------------------------------ part b
# Comment out the "querying a collection" section above and
# implement the chatbot here once Part A testing is done.
# ------------------------------------------------------------ part b: retrieval
# Embeds the user's question and returns the closest documents
# from the collection as (doc_id, text) pairs

CHAT_MODEL = 'gpt-5-mini'
N_RESULTS = 3          # number of closest documents to send to the LLM
MAX_DOC_CHARS = 6000 

def retrieve_context(question):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=question,
        model=EMBED_MODEL)
 
    # Get the embedding
    query_embedding = response.data[0].embedding
 
    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS
    )
 
    retrieved = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        retrieved.append((doc_id, doc[:MAX_DOC_CHARS]))
    return retrieved

# ------------------------------------------------------------ part b: prompt
BASE_PROMPT = (
    "You are a course advisor for the iSchool. Answer the student's question using "
    "the course syllabi provided below. When your answer comes from those syllabi, "
    "state which syllabus you used, for example: 'According to IST 256 Syllabus.pdf...'. "
    "If the syllabi do not contain the answer, say plainly that the retrieved course "
    "documents do not cover it, then answer from general knowledge and make clear that "
    "the extra information did not come from the syllabi. Keep answers clear and concise."
)
 
def build_system_prompt(retrieved):
    parts = [BASE_PROMPT, "\n\nRetrieved course syllabi:"]
    for doc_id, text in retrieved:
        parts.append(f"\n=== {doc_id} ===\n{text}")
    return "\n".join(parts)
 
# ------------------------------------------------------------ part b: chat loop
st.write("Ask a question about the course syllabi. Each question is embedded and "
"used to retrieve the most relevant syllabi from the vector database, which are "
"then added to the prompt sent to the LLM.")
 
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Ask me anything about the iSchool course syllabi.'}
    ]
 
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg['role'])
    chat_msg.write(msg['content'])
 
if question := st.chat_input('Ask a question about the courses'):
    st.session_state.messages.append({'role': 'user', 'content': question})
 
    with st.chat_message('user'):
        st.markdown(question)
 
    # Fetch the relevant documents for this question and add them to the prompt
    retrieved = retrieve_context(question)
    system_prompt = build_system_prompt(retrieved)
 
    context = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages
    stream = st.session_state.openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=context,
        stream=True)
 
    with st.chat_message('assistant'):
        response = st.write_stream(stream)
    st.session_state.messages.append({'role': 'assistant', 'content': response})
 
    # Show which documents were sent to the LLM for this question
    with st.sidebar.expander(f'Retrieved for: {question[:40]}'):
        for i, (doc_id, _) in enumerate(retrieved, start=1):
            st.write(f'**{i}. {doc_id}**')