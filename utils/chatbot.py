from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory


from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

COLLECTION_NAME = "base_de_conhecimentos"
host = os.getenv("QDRANT_HOST")
port = int(os.getenv("QDRANT_PORT"))
client = QdrantClient(host=host, port=port)

def index_chunks(chunks):
    """Insere os chunks no Qdrant"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    collections = client.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    vectors = embeddings.embed_documents(chunks)

    points = []
    for chunk, vector in zip(chunks, vectors):
        if chunk and chunk.strip():
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"page_content": chunk.strip()}
            ))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Inseridos {len(points)} chunks no Qdrant.")
    else:
        print("Nenhum chunk válido para inserir.")

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    ).as_retriever()

def create_conversation_chain(retriever):
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain

    llm = ChatOpenAI()
    history = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer" 
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
