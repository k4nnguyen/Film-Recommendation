import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

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

# --- KHỞI TẠO STATE CHO COLD START ---
if 'cs_selected_titles' not in st.session_state:
    st.session_state.cs_selected_titles = []

# --- 2. THANH SIDEBAR: PROFILE & ĐĂNG XUẤT ---
with st.sidebar:
    st.markdown("### Thông Tin Người Dùng")
    username = st.session_state.get('username', 'User')
    st.write(f"Chào ngày mới {username}!")
    
    st.markdown("---")
    if st.button("Đăng xuất"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.session_state.cs_selected_titles = []
        st.switch_page("login.py")

# --- 3. LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    # Load metadata phim
    try:
        df = pd.read_csv('../../crawl_data/data/movies_with_posters.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('../../crawl_data/data/movies_metadata_encoded.csv', encoding='utf-8-sig')
    
    # Load u.info để lấy ánh xạ Username -> UserID
    try:
        u_info = pd.read_csv('../../crawl_data/data/u.info', sep='\t', names=['user_id', 'username', 'password'], encoding='utf-8-sig')
    except:
        u_info = pd.DataFrame(columns=['user_id', 'username', 'password'])

    # Load u.data để kiểm tra lịch sử đánh giá
    try:
        u_data = pd.read_csv('../../crawl_data/data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating'], encoding='utf-8-sig')
    except:
        u_data = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])

    # Load u.genre
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

# --- 4. QUẢN LÝ TRẠNG THÁI TRANG CHÍNH ---
if 'page' not in st.session_state:
    st.session_state.page = "Trang chủ"
if 'selected_idx' not in st.session_state:
    st.session_state.selected_idx = None

def nav_to(page):
    st.session_state.page = page
    st.session_state.selected_idx = None
    st.rerun()

# --- 5. HÀM HIỂN THỊ LƯỚI PHIM ---
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
                    st.image(p_url, use_container_width=True)
                    st.markdown(f'<div class="movie-title">{movie["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-genre">{movie["genre"]}</div>', unsafe_allow_html=True)
                    
                    if st.button("Chi tiết", key=f"btn_{idx}"):
                        st.session_state.selected_idx = idx
                        st.session_state.page = "Chi tiết"
                        st.rerun()

# --- 6. LOGIC COLD START (KIỂM TRA THEO USER_ID) ---
# 1. Tìm user_id từ u.info dựa trên username đang đăng nhập
user_row = u_info[u_info['username'] == username]
if not user_row.empty:
    current_user_id = user_row.iloc[0]['user_id']
    # 2. Kiểm tra xem user_id này đã có bản ghi nào trong u.data chưa
    is_new_user = current_user_id not in u_data['user_id'].values
else:
    # Trường hợp hy hữu không thấy trong u.info (có thể do lỗi ghi file)
    is_new_user = True
    current_user_id = None

if is_new_user:
    st.title("Chào mừng bạn đến với hệ thống")
    st.write("Vui lòng chọn sở thích ban đầu để chúng tôi gợi ý phim tốt hơn cho bạn.")
    
    selected_genres = st.multiselect("Chọn 3 thể loại bạn yêu thích nhất", all_genres, max_selections=3)
    
    if len(selected_genres) == 3:
        st.divider()
        st.markdown("#### Hãy chọn ra 3 bộ phim bạn thấy ấn tượng nhất:")
        
        mask = df[selected_genres].any(axis=1)
        filtered_movies = df[mask].head(20)
        
        num_cols = 5
        selected_titles = st.session_state.cs_selected_titles

        for i in range(0, len(filtered_movies), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                movie_idx = i + j
                if movie_idx < len(filtered_movies):
                    movie = filtered_movies.iloc[movie_idx]
                    title = movie['title']
                    p_url = movie['poster_url'] if pd.notna(movie['poster_url']) else "https://via.placeholder.com/300x450"
                    
                    with cols[j]:
                        st.image(p_url, use_container_width=True)
                        st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
                        cb_key = f"cb_cs_{movie.name}"
                        disabled = len(selected_titles) >= 3 and title not in selected_titles
                        is_selected = st.checkbox("Chọn", key=cb_key, value=title in selected_titles, disabled=disabled)
                        
                        if is_selected and title not in selected_titles:
                            st.session_state.cs_selected_titles.append(title)
                        elif not is_selected and title in selected_titles:
                            st.session_state.cs_selected_titles.remove(title)
                        selected_titles = st.session_state.cs_selected_titles

        count_selected = len(st.session_state.cs_selected_titles)
        if count_selected < 3:
            st.warning(f"Bạn đang chọn {count_selected}/3 bộ phim.")
        elif count_selected == 3:
            st.success("Bấm nút dưới đây để hoàn tất.")
            if st.button("Hoàn tất thiết lập"):
                new_entries = []
                for m_title in st.session_state.cs_selected_titles:
                    # m_idx + 1 để khớp với logic movie_id thường bắt đầu từ 1
                    m_idx = df[df['title'] == m_title].index[0] + 1
                    # Lưu bằng current_user_id (số) thay vì username (chuỗi)
                    new_entries.append([current_user_id, m_idx, 5])
                
                new_data_df = pd.DataFrame(new_entries)
                new_data_df.to_csv('../../crawl_data/data/u.data', mode='a', sep='\t', index=False, header=False, encoding='utf-8-sig')
                
                st.session_state.cs_selected_titles = []
                st.cache_data.clear()
                st.rerun()
    st.stop()

# --- 7. THANH NAVBAR ---
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

# --- 8. LOGIC ĐIỀU HƯỚNG VÀ HIỂN THỊ CHÍNH ---
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
        if st.button("Quay lại"): nav_to("Trang chủ")
        c1, c2 = st.columns([1, 2])
        with c1: st.image(movie['poster_url'], use_container_width=True)
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

# --- 9. FOOTER ---
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