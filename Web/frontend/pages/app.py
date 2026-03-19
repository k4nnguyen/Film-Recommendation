import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# KIỂM TRA ĐĂNG NHẬP
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("Vui lòng đăng nhập để sử dụng hệ thống!")
    st.stop()

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
    header {height: 0px !important; opacity: 0 !important;} 

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

# --- 2. LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../../crawl_data/data/movies_with_posters.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('../../crawl_data/data/movies_metadata_encoded.csv', encoding='utf-8-sig')
    
    try:
        u_info = pd.read_csv('../../crawl_data/data/u.info', sep='\t', names=['user_id', 'username', 'password'], encoding='utf-8-sig')
    except:
        u_info = pd.DataFrame(columns=['user_id', 'username', 'password'])

    try:
        u_data = pd.read_csv('../../crawl_data/data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating'], encoding='utf-8-sig')
    except:
        u_data = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

    try:
        genres_df = pd.read_csv('../../crawl_data/data/u.genre', sep='|', names=['genre', 'id'], encoding='utf-8-sig')
        list_genres = genres_df['genre'].tolist()
        if 'unknown' in list_genres: list_genres.remove('unknown')
    except:
        list_genres = []

    features = df.select_dtypes(include=['number'])
    sim_matrix = cosine_similarity(features)
    return df, sim_matrix, u_info, u_data, list_genres

df, sim_matrix, u_info, u_data, all_genres = load_data()

# Xác định User ID hiện tại
username = st.session_state.get('username', 'Người dùng')
user_row = u_info[u_info['username'] == username]
current_user_id = user_row.iloc[0]['user_id'] if not user_row.empty else None

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
    data_path = '../../crawl_data/data/u.data'
    temp_u_data = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating'], encoding='utf-8-sig')
    mask = (temp_u_data['user_id'] == u_id) & (temp_u_data['movie_id'] == m_id)
    if temp_u_data[mask].empty:
        new_row = pd.DataFrame([[u_id, m_id, new_rate]], columns=['user_id', 'movie_id', 'rating'])
        temp_u_data = pd.concat([temp_u_data, new_row], ignore_index=True)
    else:
        temp_u_data.loc[mask, 'rating'] = new_rate
    temp_u_data.to_csv(data_path, sep='\t', index=False, header=False, encoding='utf-8-sig')
    st.cache_data.clear()

def display_grid(indices, cols=5):
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
                    if st.button("Chi tiết", key=f"grid_btn_{idx}"):
                        st.session_state.selected_idx = idx
                        st.session_state.page = "Chi tiết"
                        st.rerun()

# --- 5. LOGIC COLD START ---
is_new_user = (current_user_id is None) or (current_user_id not in u_data['user_id'].values)

if is_new_user:
    st.title("Chào mừng bạn đến với hệ thống")
    st.write("Vui lòng chọn sở thích ban đầu:")
    selected_genres = st.multiselect("Bước 1: Chọn 3 thể loại yêu thích", all_genres, max_selections=3)
    
    if len(selected_genres) == 3:
        st.divider()
        st.markdown("#### Bước 2: Chọn 3 bộ phim bạn thấy ấn tượng nhất:")
        mask = df[selected_genres].any(axis=1)
        filtered_movies = df[mask].head(20)
        num_cols = 5
        if 'cs_selected_titles' not in st.session_state: st.session_state.cs_selected_titles = []
        
        for i in range(0, len(filtered_movies), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                m_idx = i + j
                if m_idx < len(filtered_movies):
                    movie = filtered_movies.iloc[m_idx]
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
                new_entries = []
                for m_title in st.session_state.cs_selected_titles:
                    real_id = df[df['title'] == m_title].index[0] + 1
                    new_entries.append([current_user_id, real_id, 5])
                pd.DataFrame(new_entries).to_csv('../../crawl_data/data/u.data', mode='a', sep='\t', index=False, header=False, encoding='utf-8-sig')
                st.session_state.cs_selected_titles = []
                st.cache_data.clear()
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
    if st.button("Trang chủ"): 
        st.session_state.page = "Trang chủ"
        st.session_state.selected_idx = None
        st.rerun()
with nav_c3:
    if st.button("Danh sách"): 
        st.session_state.page = "Danh sách"
        st.session_state.selected_idx = None
        st.rerun()
with nav_c4: search = st.text_input("", placeholder="Tìm kiếm phim...", label_visibility="collapsed")

st.divider()

if search:
    res = df[df['title'].str.contains(search, case=False)].index.tolist()
    st.title(f"Kết quả cho: '{search}'")
    display_grid(res)
elif st.session_state.page == "Danh sách":
    scroll_to_top()
    st.title("Tất cả phim")
    display_grid(list(range(len(df))))
elif st.session_state.page == "Chi tiết" and st.session_state.selected_idx is not None:
    scroll_to_top()
    movie = df.iloc[st.session_state.selected_idx]
    movie_id = st.session_state.selected_idx + 1
    
    if st.button("Quay lại"): 
        st.session_state.page = "Trang chủ"
        st.rerun()

    c1, c2 = st.columns([1, 2])
    with c1: st.image(movie['poster_url'], use_container_width=True)
    with c2:
        st.title(movie['title'])
        st.write(f"**Thể loại:** {movie['genre']}")
        st.write(f"**Quốc gia:** {movie['country'] if pd.notna(movie['country']) else 'N/A'}")
        st.write(f"**Ngày chiếu:** {movie['release_date']}")
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
    st.subheader("Gợi ý phim tương tự")
    sim_scores = list(enumerate(sim_matrix[st.session_state.selected_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    display_grid([x[0] for x in sim_scores])
else:
    st.title("Phim mới đề xuất")
    display_grid(list(range(len(df)))[:15])

st.markdown('<div class="footer"><p>Nguyễn Kim An - Nguyễn Tiến Đạt - Trần Đức Lâm - PTIT © 2026</p></div>', unsafe_allow_html=True)