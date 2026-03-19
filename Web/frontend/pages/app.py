import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# KIỂM TRA ĐĂNG NHẬP
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Vui lòng đăng nhập để sử dụng hệ thống!")
    st.stop()

# --- 1. CẤU HÌNH & CSS ---
st.set_page_config(page_title="Movie Recommender System", layout="wide", page_icon=None)

style_css = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    [data-testid="stImage"] img {
        height: 350px !important;
        object-fit: cover !important;
        border-radius: 10px;
        width: 100% !important;
    }

    .movie-title {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        height: 45px;
        font-weight: bold;
        margin-top: 10px;
        line-height: 1.3;
    }

    .movie-genre {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #888;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #ff4b4b;
    }

    /* CSS cho Footer */
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888;
        text-align: center;
        padding: 30px 0;
        margin-top: 50px;
        border-top: 1px solid #333;
    }
    </style>
"""
st.markdown(style_css, unsafe_allow_html=True)

# Giải pháp cuộn trang bằng mẹo onerror
def scroll_to_top():
    js_scroll = """
        <img src="x" onerror="
            var mainSection = window.parent.document.querySelector('section.main');
            if (mainSection) {
                mainSection.scrollTo({ top: 0, behavior: 'auto' });
            }
            this.parentNode.removeChild(this);
        " style="display:none;">
    """
    st.markdown(js_scroll, unsafe_allow_html=True)

# --- 2. THANH SIDEBAR: PROFILE & ĐĂNG XUẤT ---
with st.sidebar:
    st.markdown("### Thông Tin Người Dùng")
    username = st.session_state.get('username', 'User')
    st.write(f"Chào ngày mới {username}!")
    
    st.markdown("---")
    if st.button("Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.switch_page("login.py")

# --- 3. LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        # Giữ nguyên đường dẫn theo yêu cầu
        df = pd.read_csv('../../crawl_data/data/movies_with_posters.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('../../crawl_data/data/movies_metadata_encoded.csv', encoding='utf-8-sig')
    
    features = df.select_dtypes(include=['number'])
    sim_matrix = cosine_similarity(features)
    return df, sim_matrix

df, sim_matrix = load_data()

# --- 4. QUẢN LÝ TRẠNG THÁI ---
if 'page' not in st.session_state:
    st.session_state.page = "Trang chủ"
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None

def nav_to(page):
    st.session_state.page = page
    st.session_state.selected_idx = None
    st.rerun()

# --- 5. THANH NAVBAR ---
nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([2, 1, 1, 3])
with nav_c1:
    st.markdown("### Hệ thống gợi ý phim")
with nav_c2:
    if st.button("Trang chủ"): nav_to("Trang chủ")
with nav_c3:
    if st.button("Danh sách"): nav_to("Danh sách")
with nav_c4:
    search = st.text_input("", placeholder="Tìm phim...", label_visibility="collapsed")

st.divider()

# --- 6. HÀM HIỂN THỊ LƯỚI PHIM ---
def display_grid(indices, cols=5):
    if not indices:
        st.warning("Không tìm thấy phim nào phù hợp.")
        return
        
    for i in range(0, len(indices), cols):
        columns = st.columns(cols)
        for j in range(cols):
            if i + j < len(indices):
                idx = indices[i+j]
                movie = df.iloc[idx]
                with columns[j]:
                    p_url = movie['poster_url'] if 'poster_url' in movie and pd.notna(movie['poster_url']) else "https://via.placeholder.com/300x450"
                    st.image(p_url)
                    st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-genre">{movie["genre"]}</div>', unsafe_allow_html=True)
                    
                    if st.button("Chi tiết", key=f"btn_{idx}"):
                        st.session_state.selected_idx = idx
                        st.session_state.page = "Chi tiết"
                        st.rerun()

# --- 7. LOGIC ĐIỀU HƯỚNG TRANG ---

if search:
    res = df[df['title'].str.contains(search, case=False)].index.tolist()
    st.title(f"Kết quả cho: {search}")
    display_grid(res)

elif st.session_state.page == "Danh sách":
    scroll_to_top()
    st.title("Tất cả phim trong hệ thống")
    display_grid(list(range(len(df))))

elif st.session_state.page == "Chi tiết":
    if st.session_state.selected_idx is not None:
        scroll_to_top()
        movie = df.iloc[st.session_state.selected_idx]
        
        if st.button("Quay lại"): 
            nav_to("Trang chủ")
        
        c1, c2 = st.columns([1, 2])
        with c1: 
            st.image(movie['poster_url'])
        with c2:
            st.title(movie['title'])
            st.write(f"**Thể loại:** {movie['genre']}")
            st.write(f"**Quốc gia:** {movie['country'] if pd.notna(movie['country']) else 'N/A'}")
            st.write(f"**Ngày chiếu:** {movie['release_date']}")
            st.link_button("Xem trên Momo", movie['url'])
        
        st.divider()
        st.subheader("Gợi ý tương tự dựa trên nội dung")
        sim_scores = list(enumerate(sim_matrix[st.session_state.selected_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        display_grid([x[0] for x in sim_scores])
    else:
        nav_to("Trang chủ")

else:
    st.title("Phim mới đề xuất")
    display_grid(list(range(len(df)))[:15])

# --- 8. FOOTER ---
st.markdown(
    """
    <div class="footer">
        <p>Hệ thống gợi ý phim - Đồ án thực hành</p>
        <p>Sinh viên thực hiện: Nguyễn Kim An - Nguyễn Tiến Đạt - Trần Đức Lâm - Học viện Công nghệ Bưu chính Viễn thông (PTIT)</p>
        <p>© 2026 Movie Recommender System</p>
    </div>
    """,
    unsafe_allow_html=True
)