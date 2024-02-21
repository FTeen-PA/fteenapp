__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
# import pandas as pd




def load_llm_models(llm_name):
    llm = ChatOpenAI(model_name=llm_name, temperature=0.5)
    return llm

@st.cache_resource
def load_db(dir, _embeddings):
    db = Chroma(persist_directory=dir, embedding_function=_embeddings)
    return db

@st.cache_resource
def load_embedding(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def load_retriever(db, k=5, search_type="similarity"):
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})
    return retriever

def load_qa_chain(llm, chain_type, retriever):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa_chain

def send_message():
    user_input = st.session_state.user_input
    if user_input:
        # Process the user input and get the response
        bot_response = chatbot.get_response(user_input)
        # Append user input and bot response to chat history
        st.session_state.chat_history.append({"sender": "user", "message": user_input})
        st.session_state.chat_history.append({"sender": "bot", "message": bot_response})
        # Clear the input field
        st.session_state.user_input = ''


emb = load_embedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = load_db("./jawazat_db33", emb)
llm = load_llm_models("gpt-3.5-turbo")
retriever = load_retriever(db)
qa_chain = load_qa_chain(llm, "stuff", retriever)


class ChatBot:
    def __init__(self, qa):
        self.qa = qa

    def get_response(self, query):
        result = self.qa({"question": query, "chat_history": []})
        return result["answer"]

chatbot = ChatBot(qa_chain)


def send_message():
    user_input = st.session_state.user_input
    if user_input:
        bot_response = chatbot.get_response(user_input)
        st.session_state.chat_history.append({"sender": "user", "message": user_input})
        st.session_state.chat_history.append({"sender": "bot", "message": bot_response})
        st.session_state.user_input = ''  # Clear the input



# Define Streamlit app
def main():
    # Custom CSS for RTL layout, specific fonts for title and subtitle, and font import
    rtl_and_custom_font_style = """
        <style>
            @import url('/app/static/font/IBMPlexSansArabic-Bold.ttf');
            
            body {
                direction: rtl;
                text-align: right;
                font-family: "IBM Plex Arabic", sans-serif;
            }
            .stApp {
                background-image: url('https://l.top4top.io/p_2971orcvd1.jpg'); /* Add your image URL */
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
                direction: rtl;
                text-align: right;
            }
            @media (max-width: 768px) {
                body {
                    background-image: url('https://j.top4top.io/p_2973qhc431.jpg') !important;
                 }
            }
            .custom-title {
                font-family: 'IBM Plex Arabic', sans-serif; /* Use the imported font for the title */
                font-size: 24px; /* You can adjust the size as needed */
                color: #00651d; /* Adjust color as needed */
            }
            .custom-subtitle {
                font-family: 'IBM Plex Arabic', sans-serif; /* Use the imported font for the subtitle */
                font-size: 20px; /* You can adjust the size as needed */
                color: #303030; /* Adjust color as needed */
            }
                        /* New style for the input placeholder */
            ::placeholder { /* Chrome, Firefox, Opera, Safari 10.1+ */
                font-family: 'IBM Plex Arabic', sans-serif;
                font-size: 16px; /* You can adjust the size as needed */
                color: #A9A9A9; /* Adjust color as needed */
                opacity: 1; /* Firefox */
            }
            .stTextInput>div>div>input {
                font-family: 'IBM Plex Arabic', sans-serif !important;
            }
            .stButton>button {
                font-family: 'IBM Plex Arabic', sans-serif !important;
                font-size: 18px; /* Adjust the size as needed */
                color: #FFFFFF; /* Adjust color as needed */
                border: 2px solid #4CAF50; /* Optional: Adjust border style */
                border-radius: 5px; /* Optional: Adjust border radius */
                background-color: #4CAF50; /* Optional: Adjust background color */
            }
            .custom-response {
                font-family: 'IBM Plex Arabic', sans-serif;
                display: -webkit-box;
                -webkit-box-orient: vertical;
                -webkit-line-clamp: 2;
                overflow: hidden;
                text-overflow: ellipsis;
                padding: 4px;
                margin-bottom: 10px; /* Add some space below the response */
            }
            .chat-message {
                display: flex;
                align-items: center;
                font-family: 'IBM Plex Arabic', sans-serif;
            }
            .chat-icon {
                width: 30px;
                height: 30px;
                border-radius: 15px; /* جعل الأيقونة دائرية */
                margin-right: 10px;
            }
            .chat-text {
                flex: 1;
                border: 0px solid #eee;
                border-radius: 5px;
                padding: 10px;
            }
            stButton>button {
            font-family: 'IBM Plex Arabic', sans-serif;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            padding: 10px 24px;
            color: white;
            background-color: #008CBA;
            }
            .chat-container {
            height: 200px; /* حدد الارتفاع المطلوب لمنطقة المحادثة */
            overflow-y: auto; /* تمكين التمرير الجانبي عند الحاجة */
            border: 0px solid #ffffff; /* خط الحدود (اختياري) */
            margin-bottom: 20px; /* مسافة بين الحدود وبقية العناصر */
            padding: 10px; /* التباعد داخل الحدود */
            background-color: #ffffff; /* لون خلفية منطقة المحادثة */
            }
            .tab-content > div > div > div > div > div > div > div:first-child {
            font-family: 'IBM Plex Arabic', sans-serif;
            }
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size:12px;
            font-family: 'IBM Plex Arabic', sans-serif;
            }
            .stText {
            font-size:14px;
            font-family: 'IBM Plex Arabic', sans-serif;
            }
        </style>
    """
    st.title("Fateen App")

    tab1, tab2, tab3 = st.tabs(["اسألني", "تواصل معنا", "عن النظام"])
    with tab1:

        # Initialize session state variables
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ''
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []


        st.markdown(rtl_and_custom_font_style, unsafe_allow_html=True)

        # Set title and subtitle with custom CSS classes
        st.markdown('<h1 class="custom-title">المساعد الشخصي (فـطـين)</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="custom-subtitle">مرحبًا بك! كيف أقدر اساعدك:</h2>', unsafe_allow_html=True)

        # User input with the on_change callback
        st.text_input("", placeholder="أكتب سؤالك هنا ...",
                    key="user_input", on_change=send_message, value=st.session_state.user_input)

        logo_usr = 'https://imgg.io/images/2024/02/18/c02a6e3b5dc7c491086c0fc6c593a595.png'  
        logo_bot = 'https://e.top4top.io/p_2970wzumc1.png'
        
        # Chatbox style
        st.write('<style>.chatbox {height: 300px; overflow-y: scroll; border: 0px solid #ccc; margin-bottom: 10px; padding: 5px;}</style>', unsafe_allow_html=True)
        chatbox = st.empty()

        # Display chat history
        chat_history_html = "<div class='chatbox'>"
        for chat in st.session_state.chat_history:
            chat_history_html += f"<div class='chat-message {('user' if chat['sender'] == 'user' else 'bot')}'>"
            chat_history_html += f"<img src='{logo_usr if chat['sender'] == 'user' else logo_bot}' class='chat-icon'>"
            chat_history_html += f"<div class='chat-text'>{chat['message']}</div>"
            chat_history_html += "</div>"
        chat_history_html += "</div>"
        chatbox.markdown(chat_history_html, unsafe_allow_html=True)
    with tab2:
        st.markdown('<h2 class="custom-subtitle">نسعد بالتواصل معكم عبر أحد القنوات التالية :</h2>', unsafe_allow_html=True)
        st.markdown(rtl_and_custom_font_style, unsafe_allow_html=True)
        st.markdown('<div class="stText">يمكن التواصل معنا عبر الإيميل التالي : t5.32hhf@gmail.com</div>', unsafe_allow_html=True)
        st.markdown('<div class="stText">يمكن التواصل معنا عبر حسابنا في منصة إكس : Fateen@x.com</div>', unsafe_allow_html=True)



    with tab3:
        st.markdown(rtl_and_custom_font_style, unsafe_allow_html=True)
        st.markdown('<div class="stText">فطين .. بداية جديدة للرد التفاعلي ، يهدف لرفع مستوى رضا المستخدمين نقلة نوعية إلى مستوى مختلف تماماً مما جعل التواصل أكثر كفاءة وفعالية ، فطين سيصبح نموذجاً مثيراً لروبوت الدردشة لخدمة العملاء بوزارة الداخلية.</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="stText">كيف تعلم فطين ؟ تم تغذية فطين كمرحلة ( مبدئية ) من خلال دمج بيانات حساب خدمة العملاء للمديرية العامة للجوازات بمنصة تويتر وكذلك التعليمات والإجراءات الرسمية للوزارة . و سيتبعها بمشيئة الله مرحلة جمع بيانات جميع القطاعات ( بعد تمكيننا من ذلك ) ليصبح فطين أكثر شمولية وتكامل .</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="stText">مستقبل فطين .. مستقبل واعد إذا توفرت البيانات اللازمة من قبل الوزارة سيتم التوسع تدريجياً ليشمل فطين خدمة العملاء لجميع قطاعات الوزارة</div>', unsafe_allow_html=True)

# The main function call
if __name__ == "__main__":
    main()