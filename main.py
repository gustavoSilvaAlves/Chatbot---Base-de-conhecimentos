import streamlit as st
from utils import text, chatbot
import os

st.set_page_config(
    page_title='Mbot - Bot da Mtec',
    page_icon='🤖',
    layout='wide'
)

st.header('🤖 Olá, eu sou o Mbot. Em que posso te ajudar?')

st.markdown("""
🔎 **Exemplo de informações que o Mbot pode responder:**
- Informações relacionadas ao Wifi Corporativo
- Como criar um Pedido de compra no Salesforce
- Como abrir um chamado na TI
""")


if 'conversation' not in st.session_state or st.session_state.conversation is None:
    retriever = chatbot.get_retriever()
    st.session_state.conversation = chatbot.create_conversation_chain(retriever)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header('🔒 Área de Admin')

    admin_password = st.text_input("Digite a senha de admin", type="password")
    correct_password = os.getenv("ADMIN_PASSWORD", "senha123") 

    if admin_password == correct_password:
        st.success("Acesso de admin concedido!")
        pdf_docs = st.file_uploader(
            "📄 Upload de arquivos",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button('Processar') and pdf_docs:
            with st.spinner("Processando arquivos..."):
                all_files_text = text.process_files(pdf_docs)

                if not all_files_text:
                    st.warning("Nenhum texto foi extraído dos arquivos.")
                else:
                    chunks = text.create_text_chunks(all_files_text)
                    if not chunks:
                        st.warning("Nenhum conteúdo válido foi gerado.")
                    else:
                        chatbot.index_chunks(chunks)
                        retriever = chatbot.get_retriever()
                        st.session_state.conversation = chatbot.create_conversation_chain(retriever)
                        st.success("Base de conhecimento atualizada!")
    else:
        st.info("Entre com a senha para acessar a área de upload de arquivos.")

question = st.chat_input("Digite sua pergunta aqui...")

if question:
    for msg in st.session_state.chat_history:
        st.chat_message("user" if msg.type == "human" else "ai").write(msg.content)

    st.chat_message("user").write(question)
    with st.spinner("Pensando..."):
        result = st.session_state.conversation.invoke({"question": question})
        st.session_state.chat_history = result["chat_history"]
        st.chat_message("ai").write(result["answer"])
