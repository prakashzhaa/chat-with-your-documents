import streamlit as st
import sqlite3
import os 
import yaml
from llm_chains import load_noraml_chain, load_pdf_chat_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from utils import save_chat_history_json, get_timestamp, load_chat_history_json, load_config, get_icon
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from image_handler import handle_image
from pdf_handler import add_documents_to_db
from html_templates import css
from database_operations import load_last_k_text_message, save_text_message, save_image_message,save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history
config = load_config()
@st.cache_resource

def load_chain():
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain()
    return load_noraml_chain()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"
    
    
def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def clear_cache():
    st.cache_resource.clear()


def main():
    st.title("Chat With Your Documents")
    st.write(css, unsafe_allow_html=True)
    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1  
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None      
    
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)

    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    pdf_toggle_column, voice_recording_column = st.sidebar.columns(2)
    pdf_toggle_column.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt = "Start recording", stop_prompt = "Stop recording", just_once=True)

    delete_chat_column, clear_cache_column = st.sidebar.columns(2)
    delete_chat_column.button("Delete Chat History", on_click=delete_chat_session_history)
    clear_cache_column.button("Clear Cache", on_click=clear_cache)

    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key= "user_input")

    uploaded_pdfs = st.sidebar.file_uploader("Upload your PDFs", accept_multiple_files=True, key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat)
    uploaded_audio = st.sidebar.file_uploader("Upload your audio file", type=["wav", "mp3", "ogg"], key=st.session_state.audio_uploader_key)
    uploaded_image = st.sidebar.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
    
    if uploaded_pdfs:
        with st.spinner("Processing PDFs..."):
            add_documents_to_db(uploaded_pdfs)
            st.session_state.pdf_uploader_key += 2

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input="Summerize this text: " + transcribed_audio, chat_history=[])
        save_audio_message(get_session_key(), "human", uploaded_audio.getvalue())
        save_text_message(get_session_key(), "ai", llm_answer)
        st.session_state.audio_uploader_key += 2

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input=transcribed_audio, chat_history=load_last_k_text_message(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_audio_message(get_session_key(), "human", voice_recording["bytes"])
        save_text_message(get_session_key(), "ai", llm_answer)

    if user_input:
        if uploaded_image:
            with st.spinner("Processing Image..."):
                llm_answer = handle_image(uploaded_image.getvalue(), user_input)
                save_text_message(get_session_key(), "human", user_input)
                save_image_message(get_session_key(), "human", uploaded_image.getvalue())
                save_text_message(get_session_key(), "ai", llm_answer)
                user_input = None

        if user_input:
            llm_chain = load_chain()
            llm_answer = llm_chain.run(user_input=user_input, chat_history=load_last_k_text_message(get_session_key(), config["chat_config"]["chat_memory_length"]))
            save_text_message(get_session_key(), "human", user_input)
            save_text_message(get_session_key(), "ai", llm_answer)
            user_input = None
    
    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_message = load_messages(get_session_key())
            
            for message in chat_history_message:
                with st.chat_message(name=message["sender_type"], avatar=get_icon(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")
                    
        if (st.session_state.session_key =='new_session') and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()





