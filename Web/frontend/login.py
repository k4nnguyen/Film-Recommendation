import streamlit as st
import sqlite3
import bcrypt
def load_css():
    st.markdown("""
        <style>
        /* 1. Thu hẹp form lại và tạo hiệu ứng thẻ (Card) nổi lên giữa màn hình */
        .block-container {
            max-width: 450px !important; /* Độ rộng của form đăng nhập */
            padding: 2.5rem 2rem !important;
            background-color: #ffffff; /* Màu nền trắng cho form */
            border-radius: 15px; /* Bo góc form */
            box-shadow: 0 8px 20px rgba(0,0,0,0.08); /* Đổ bóng nhẹ */
            margin-top: 5vh; /* Đẩy xuống một chút từ viền trên */
        }
        
        /* 2. Đổi màu nền của toàn bộ trang web sang màu xám nhạt để làm nổi bật form */
        .stApp {
            background-color: #f4f7f6;
        }
        
        /* 3. Căn giữa các Tiêu đề */
        h1, h2, h3 {
            text-align: center !important;
            color: #2c3e50;
            padding-bottom: 10px;
        }
        
        /* 4. Bo góc và làm đẹp ô nhập liệu (Input) */
        .stTextInput input {
            border-radius: 8px !important;
            border: 1px solid #e0e0e0 !important;
            padding: 10px 15px !important;
        }
        .stTextInput input:focus {
            border-color: #ff4b4b !important;
            box-shadow: 0 0 0 1px #ff4b4b !important;
        }
        
        /* 5. Làm đẹp Nút bấm (Button) và thêm hiệu ứng Hover */
        .stButton button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            height: 45px !important;
            transition: all 0.3s ease !important;
        }
        
        /* Hiệu ứng nảy lên nhẹ khi di chuột vào nút */
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-color: transparent !important;
        }
                
        /* 6. Ẩn dòng chữ "Press Enter to apply" trong ô input */
        div[data-testid="InputInstructions"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Ẩn thêm khoảng trống dư thừa do dòng chữ để lại (nếu có) */
        div[data-testid="stTextInput"] label {
            padding-bottom: 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)
# ==========================================
# 1. SETUP DATABASE (CƠ SỞ DỮ LIỆU)
# ==========================================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Tạo bảng users nếu chưa có
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# 2. HÀM MÃ HÓA & KIỂM TRA MẬT KHẨU
# ==========================================
def hash_password(password):
    # Mã hóa mật khẩu với bcrypt
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    # So sánh mật khẩu nhập vào với mã hash trong DB
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ==========================================
# 3. THAO TÁC VỚI DATABASE
# ==========================================
def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Lỗi xảy ra nếu username đã tồn tại
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
# 4. GIAO DIỆN STREAMLIT CHÍNH
# ==========================================
def main():
    st.set_page_config(page_title="Hệ thống Đăng nhập")
    
    # GỌI HÀM CSS ĐỂ TRANG TRÍ GIAO DIỆN
    load_css()

    # Khởi tạo DB khi chạy app
    init_db()

    # Quản lý trạng thái đăng nhập
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''

    if not st.session_state['logged_in']:
        st.title(" Đăng Nhập")
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 20px;'>Vui lòng điền thông tin để tiếp tục</p>", unsafe_allow_html=True)
        
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type='password')
        
        st.write("") 
        
        # 1. CHỈ VẼ GIAO DIỆN NÚT BẤM VÀ GÁN VÀO BIẾN
        col1, col2 = st.columns(2)
        with col1:
            btn_login = st.button("Đăng nhập", type="primary", use_container_width=True)
        with col2:
            btn_register = st.button("Đăng ký", use_container_width=True)

        # 2. XỬ LÝ LOGIC VÀ HIỂN THỊ THÔNG BÁO Ở NGOÀI CỘT (FULL WIDTH)
        
        if btn_login:
            if username == "" or password == "":
                st.warning("Vui lòng nhập tên đăng nhập và mật khẩu!")
            elif login_user(username, password):
                # Lưu trạng thái đăng nhập
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                
                
                st.switch_page("app.py")
                
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")

        if btn_register:
            # [Code phần đăng ký của bạn giữ nguyên không đổi...]
            if username == "" or password == "":
                st.warning("Vui lòng điền thông tin để đăng ký!")
            elif add_user(username, password):
                st.success("Đăng ký thành công! Bạn có thể bấm Đăng nhập.")
            else:
                st.error("Tên đăng nhập đã tồn tại. Vui lòng chọn tên khác.")

if __name__ == '__main__':
    main()