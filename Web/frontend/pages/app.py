import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests

# KIỂM TRA ĐĂNG NHẬP
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Vui lòng đăng nhập để sử dụng hệ thống!")
    st.switch_page("login.py") # Chuyển người dùng về trang login nếu chưa đăng nhập
    
# --- 1. CẤU HÌNH & CSS ---
st.set_page_config(
    page_title="Hệ thống Gợi ý Phim", 
    layout="wide", 
    page_icon=None,
    initial_sidebar_state="expanded"
)

style_css = """
    <style>
    /* 1. Đẩy nội dung chính lên sát phía trên */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* 2. Ẩn menu mặc định và footer của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {display: none !important; pointer-events: none !important;}

    /* 3. KHÓA CUỘN SIDEBAR & ĐẨY PROFILE XUỐNG ĐÁY */
    /* Ẩn menu điều hướng mặc định */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Ẩn nút đóng Sidebar */
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }

    /* Ép container sidebar chiếm trọn chiều cao và KHÔNG CHO SCROLL */
    div[data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        height: calc(100vh - 20px);
        overflow: hidden !important; /* KHÓA CUỘN TẠI ĐÂY */
    }

    .sidebar-footer {
        border-top: 1px solid #e0e0e0;
        padding-top: 20px;
        padding-bottom: 20px;
        width: 100%;
    }

    /* 4. Định dạng ảnh phim và lưới */
    [data-testid="stImage"] img {
        height: 350px !important;
        object-fit: cover !important;
        border-radius: 10px;
        width: 100% !important;
        transition: all 0.3s ease-in-out !important; /* Thêm transition cho mọi thay đổi */
    }

    /* Thêm hiệu ứng hover cho ảnh */
    [data-testid="stImage"]:hover img {
        transform: translateY(-8px) !important; /* Đẩy ảnh lên trên 8px */
        box-shadow: 0px 15px 25px rgba(0,0,0,0.15) !important; /* Đổ bóng phía dưới */
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

    /* 5. Hệ thống ngôi sao */
    .star-button {
        background: none;
        border: none;
        font-size: 35px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
        margin-right: 5px;
    }
    .star-filled { color: #FFD700; }
    .star-empty { color: #CCCCCC; }

    div.stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #ff4b4b;
    }

    .footer {
        text-align: center;
        padding: 30px 0;
        margin-top: 50px;
        border-top: 1px solid #333;
        color: #888;
    }
    </style>
"""
st.markdown(style_css, unsafe_allow_html=True)

# Giải pháp cuộn trang (Main page)
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

# --- 2. LOAD DỮ LIỆU TỪ BACKEND ---
API_URL = "http://127.0.0.1:8000"

@st.cache_data
def load_base_data():
    # 1. Gọi Backend lấy danh sách phim
    resp_movies = requests.get(f"{API_URL}/movies")
    df = pd.DataFrame(resp_movies.json())
    
    # 2. Gọi Backend lấy danh sách thể loại
    resp_genres = requests.get(f"{API_URL}/genres")
    all_genres = resp_genres.json()
    
    return df, all_genres

df, all_genres = load_base_data()

# Lấy thông tin username từ session_state
username = st.session_state.get('username', 'Người dùng')

# Lấy ID thực tế từ phiên đăng nhập
current_user_id = st.session_state.get('user_id')

# 3. Gọi Backend lấy lịch sử đánh giá của ĐÚNG user này
if current_user_id:
    resp_ratings = requests.get(f"{API_URL}/user-ratings/{current_user_id}")
    data_json = resp_ratings.json()
    
    # KÍCH HOẠT KHIÊN BẢO VỆ: Kiểm tra xem list có rỗng không
    if len(data_json) > 0:
        u_data = pd.DataFrame(data_json)
    else:
        u_data = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
else:
    u_data = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

# --- 3. THANH SIDEBAR  ---
with st.sidebar:
    st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
    st.markdown("### Thông tin cá nhân")
    st.write(f"Chào ngày mới, **{username}**!")
    
    if st.button("Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        if 'cs_selected_titles' in st.session_state:
            st.session_state.cs_selected_titles = []
        st.switch_page("login.py")
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. HÀM HỖ TRỢ ---

def update_rating(u_id, m_id, new_rate):
    API_URL = "http://127.0.0.1:8000"
    
    # Đóng gói dữ liệu gửi lên API phải khớp với class RatingData bên backend
    payload = {
        "user_id": int(u_id),
        "movie_id": int(m_id),
        "rating": int(new_rate)
    }
    
    # Gửi qua phương thức POST
    response = requests.post(f"{API_URL}/update-rating", json=payload)
    
    if response.status_code == 200:
        st.cache_data.clear() # Xóa cache giao diện để cập nhật lại số sao
    else:
        st.error("Lỗi khi lưu đánh giá!")
    
def go_home():
    st.session_state.page = "Trang chủ"
    st.session_state.selected_idx = None
    if 'search_input' in st.session_state:
        st.session_state.search_input = ""

def go_list():
    st.session_state.page = "Danh sách"
    st.session_state.selected_idx = None
    if 'search_input' in st.session_state:
        st.session_state.search_input = ""

def go_detail(idx):
    st.session_state.selected_idx = idx
    st.session_state.page = "Chi tiết"
    if 'search_input' in st.session_state:
        st.session_state.search_input = ""
        
def display_grid(indices, cols=5, key_prefix="grid"):
    if not indices:
        st.warning("Không tìm thấy phim phù hợp.")
        return
    for i in range(0, len(indices), cols):
        columns = st.columns(cols)
        for j in range(cols):
            if i + j < len(indices):
                idx = indices[i+j]
                movie = df.iloc[idx]
                with columns[j]:
                    p_url = movie['poster_url'] if 'poster_url' in movie and pd.notna(movie['poster_url']) else "https://via.placeholder.com/300x450"
                    st.image(p_url, use_container_width=True)
                    st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-genre">{movie["genre"]}</div>', unsafe_allow_html=True)
                    st.button("Chi tiết", key=f"{key_prefix}_btn_{idx}", on_click=go_detail, args=(idx,))

# --- 5. LOGIC COLD START ---
# is_new_user = (current_user_id is None) or (current_user_id not in u_data['user_id'].values)
is_new_user = u_data.empty

if is_new_user:
    st.title("Chào mừng bạn đến với hệ thống")
    st.write("Vui lòng chọn sở thích ban đầu:")
    selected_genres = st.multiselect("Bước 1: Chọn 3 thể loại yêu thích", all_genres, max_selections=3)
    
    if len(selected_genres) == 3:
        st.divider()
        st.markdown("#### Bước 2: Chọn 3 bộ phim bạn thấy ấn tượng nhất:")
        resp = requests.post(f"{API_URL}/movies/by-genres", json={"genres": selected_genres})
        filtered_indices = resp.json().get("result_indices", [])
        num_cols = 5
        if 'cs_selected_titles' not in st.session_state: st.session_state.cs_selected_titles = []
        
        for i in range(0, len(filtered_indices), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(filtered_indices):
                    # Lấy index thực tế từ danh sách API trả về
                    real_idx = filtered_indices[i + j]
                    movie = df.iloc[real_idx]
                    with cols[j]:
                        st.image(movie['poster_url'] if pd.notna(movie['poster_url']) else "https://via.placeholder.com/300x450", use_container_width=True)
                        st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                        cb_key = f"cb_cs_{movie.name}"
                        disabled = len(st.session_state.cs_selected_titles) >= 3 and movie['title'] not in st.session_state.cs_selected_titles
                        is_selected = st.checkbox("Chọn", key=cb_key, value=movie['title'] in st.session_state.cs_selected_titles, disabled=disabled)
                        if is_selected and movie['title'] not in st.session_state.cs_selected_titles: st.session_state.cs_selected_titles.append(movie['title'])
                        elif not is_selected and movie['title'] in st.session_state.cs_selected_titles: st.session_state.cs_selected_titles.remove(movie['title'])

        if len(st.session_state.cs_selected_titles) == 3:
            if st.button("Hoàn tất thiết lập"):
                # Gom 3 cái ID phim lại thành 1 list
                selected_ids = [int(df[df['title'] == title].index[0] + 1) for title in st.session_state.cs_selected_titles]
                
                # Gọi API nhờ Backend lưu hộ
                payload = {
                    "user_id": int(current_user_id),
                    "selected_movie_ids": selected_ids
                }
                requests.post(f"{API_URL}/cold-start", json=payload)
                
                # Reset giao diện
                st.session_state.cs_selected_titles = []
                st.rerun()
    st.stop()

# --- 6. GIAO DIỆN CHÍNH ---
if 'page' not in st.session_state:
    st.session_state.page = "Trang chủ"
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None

# Navbar
nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([2, 1, 1, 3])
with nav_c1: st.markdown("### Hệ thống gợi ý phim")
with nav_c2: 
    st.button("Trang chủ", on_click=go_home)
with nav_c3:
    st.button("Danh sách", on_click=go_list)
with nav_c4: search = st.text_input("", placeholder="Tìm kiếm phim...", label_visibility="collapsed", key="search_input")
st.divider()

if search:
    search_res = requests.get(f"{API_URL}/search", params={"query":search})
    res_indices = search_res.json().get("result_indices", [])
    st.title(f"Kết quả cho: '{search}'")
    display_grid(res_indices, key_prefix="search")
elif st.session_state.page == "Danh sách":
    scroll_to_top()
    st.title("Tất cả phim")
    display_grid(list(range(len(df))), key_prefix="all")
elif st.session_state.page == "Chi tiết" and st.session_state.selected_idx is not None:
    scroll_to_top()
    movie = df.iloc[st.session_state.selected_idx]
    movie_id = st.session_state.selected_idx + 1
    
    st.button("Quay lại", on_click=go_home)

    c1, c2 = st.columns([1, 2])
    with c1: st.image(movie['poster_url'], use_container_width=True)
    with c2:
        st.title(movie['title'])
        st.write(f"**Thể loại:** {movie['genre']}")
        st.write(f"**Quốc gia:** {movie['country'] if pd.notna(movie['country']) else 'N/A'}")
        st.write(f"**Ngày chiếu:** {movie['release_date']}")
        if pd.notna(movie['url']):
            st.link_button("Xem Review trên Momo", movie['url'])
        
        st.divider()
        st.write("**Đánh giá của bạn**")
        existing_rating = u_data[(u_data['user_id'] == current_user_id) & (u_data['movie_id'] == movie_id)]
        curr_stars = int(existing_rating.iloc[0]['rating']) if not existing_rating.empty else 0
        
        star_cols = st.columns([1,1,1,1,1,10])
        for i in range(1, 6):
            with star_cols[i-1]:
                icon = "★" if i <= curr_stars else "☆"
                if st.button(icon, key=f"star_{i}", help=f"Đánh giá {i} sao"):
                    update_rating(current_user_id, movie_id, i)
                    st.rerun()
        st.caption(f"Bạn đang đánh giá {curr_stars} sao." if curr_stars > 0 else "Hãy bấm vào ngôi sao để đánh giá.")
    st.divider()
    st.subheader("Có thể bạn cũng thích")
    try:
        # Gọi API Collaborative Filtering cho User hiện tại
        cf_res = requests.get(f"{API_URL}/cf-recommend/{current_user_id}")
        cf_indices = cf_res.json().get("recommend_indices", [])
        
        # Lọc bỏ phim hiện tại ra khỏi danh sách gợi ý và chỉ lấy đúng 5 phim đầu tiên
        if st.session_state.selected_idx in cf_indices:
            cf_indices.remove(st.session_state.selected_idx)
        cf_indices = cf_indices[:5]
    except Exception as e:
        st.warning("Hệ thống gợi ý CF đang bảo trì.")
        cf_indices = []
    if cf_indices:
        display_grid(cf_indices, key_prefix="cf")
    else:
        st.info("Hãy đánh giá thêm vài bộ phim để hệ thống học được gu của bạn nhé!")
    st.divider()
    st.subheader("Gợi ý phim tương tự")
    try:
        # Nhờ Backend tính toán và lấy về danh sách ID phim tương tự
        recommend_res = requests.get(f"{API_URL}/recommend/{st.session_state.selected_idx}")
        similar_indices = recommend_res.json()["recommend_indices"]
    except:
        st.warning("Hệ thống gợi ý đang bảo trì.")
    # Nếu có các phim tương đồng thì hiển thị
    if similar_indices:
        display_grid(similar_indices, key_prefix="sim")
else:
    st.title("Phim mới đề xuất")
    trending_indices = []
    try:
        trending_res = requests.get(f"{API_URL}/movies/trending")
        trending_indices = trending_res.json().get("trending_indices", [])
    except:
        st.warning("Hệ thống đề xuất đang bảo trì.")
    # Nếu có các phim trending thì hiển thị
    if trending_indices:
        display_grid(trending_indices, key_prefix="trend")
st.markdown('<div class="footer"><p>Nguyễn Kim An - Nguyễn Tiến Đạt - Trần Đức Lâm - PTIT © 2026</p></div>', unsafe_allow_html=True)