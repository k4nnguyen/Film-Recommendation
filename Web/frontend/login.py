import streamlit as st
import sqlite3
import bcrypt
import os

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
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Hàm lấy ID tiếp theo và lưu vào u.info
def save_to_u_info(username, password):
    file_path = '../../crawl_data/data/u.info'
    next_id = 1
    
    # Kiểm tra nếu file đã tồn tại để lấy ID lớn nhất
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if lines:
                # Lấy dòng cuối cùng và tách để lấy ID
                last_line = lines[-1].strip()
                if last_line:
                    try:
                        last_id = int(last_line.split('\t')[0])
                        next_id = last_id + 1
                    except ValueError:
                        next_id = len(lines) + 1

    # Ghi đè hoặc thêm mới vào file với định dạng id\tuser\tpass
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{next_id}\t{username}\t{password}\n")

# ==========================================
# 2. HÀM MÃ HÓA & THAO TÁC DB
# ==========================================
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        # Lưu vào SQLite (có mã hóa để bảo mật hệ thống login)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hash_password(password)))
        conn.commit()
        
        # Lưu vào u.info (không mã hóa theo yêu cầu)
        save_to_u_info(username, password)
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    if data:
        return check_password(password, data[0])
    return False

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
def main():
    st.set_page_config(page_title="Hệ thống Đăng nhập")
    load_css()
    init_db()

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
            elif login_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.switch_page("pages/app.py")
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")

        if btn_register:
            if username == "" or password == "":
                st.warning("Vui lòng điền thông tin để đăng ký!")
            elif add_user(username, password):
                st.success("Đăng ký thành công! Bạn có thể bấm Đăng nhập.")
            else:
                st.error("Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")

if __name__ == '__main__':
    main()