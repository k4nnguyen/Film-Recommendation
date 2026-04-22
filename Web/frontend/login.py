import streamlit as st
import requests

def load_css():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stSidebarNav"] { display: none; }
        .block-container {
            max-width: 450px !important;
            padding: 2.5rem 2rem !important;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
            margin-top: 5vh;
        }
        .stApp { background-color: #f4f7f6; }
        h1, h2, h3 {
            text-align: center !important;
            color: #2c3e50;
            padding-bottom: 10px;
        }
        .stTextInput input {
            border-radius: 8px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 10px 15px !important;
        }
        .stTextInput input:focus {
            border-color: #ff4b4b !important;
            box-shadow: 0 0 0 1px #ff4b4b !important;
        }
        .stButton button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            height: 45px !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-color: transparent !important;
        }
        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }
        div[data-testid="stTextInput"] label {
            padding-bottom: 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. SETUP DATABASE & FILE
# ==========================================

API_URL = "http://127.0.0.1:8000"
# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
def main():
    st.set_page_config(page_title="Hệ thống Đăng nhập")
    load_css()
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    if not st.session_state['logged_in']:
        st.title("Đăng Nhập")
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 20px;'>Vui lòng điền thông tin để tiếp tục</p>", unsafe_allow_html=True)
        
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type='password')
        
        st.write("") 
        col1, col2 = st.columns(2)
        with col1:
            btn_login = st.button("Đăng nhập", type="primary", use_container_width=True)
        with col2:
            btn_register = st.button("Đăng ký", use_container_width=True)

        if btn_login:
            if username == "" or password == "":
                st.warning("Vui lòng nhập tên đăng nhập và mật khẩu!")
            else:
                try:
                    # GỌI API ĐĂNG NHẬP
                    res = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = data['username']
                        st.session_state['user_id'] = data['user_id']
                        # Giữ lại flag just_registered nếu có (để cold start biết user mới)
                        st.switch_page("pages/app.py") 
                    else:
                        st.error(res.json().get("detail", "Sai tên đăng nhập hoặc mật khẩu!"))
                except Exception as e:
                    st.error("Không thể kết nối đến Backend! Vui lòng kiểm tra lại!")
        if btn_register:
            if username == "" or password == "":
                st.warning("Vui lòng điền thông tin để đăng ký!")
            else:
                try:
                    # GỌI API ĐĂNG KÝ
                    res = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
                    if res.status_code == 200:
                        st.success("Đăng ký thành công! Bấm Đăng nhập để tiếp tục.")
                        # Đánh dấu tài khoản này là MỚI để cold start hiện lên đúng lúc
                        st.session_state['just_registered'] = True
                    else:
                        st.error(res.json().get("detail", "Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác."))
                except Exception as e:
                    st.error("Không thể kết nối đến Backend! Vui lòng kiểm tra lại!")

if __name__ == '__main__':
    main()