import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CẤU HÌNH & LOAD DỮ LIỆU ---
st.set_page_config(page_title="Movie Recommender System", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('../../crawl_data/movies_metadata_encoded.csv')
    # Trích xuất ma trận đặc trưng (từ cột index 5 trở đi là các genre binary)
    genre_features = df.iloc[:, 5:]
    # Tính toán ma trận tương đồng Cosine
    sim_matrix = cosine_similarity(genre_features)
    return df, sim_matrix

df, sim_matrix = load_data()

# Khởi tạo session_state để quản lý việc "đang xem phim nào"
if 'selected_movie_index' not in st.session_state:
    st.session_state.selected_movie_index = None

# --- 2. HÀM HỖ TRỢ ---
def get_recommendations(movie_idx, top_n=10):
    # Lấy điểm tương đồng của phim này với tất cả phim khác
    sim_scores = list(enumerate(sim_matrix[movie_idx]))
    # Sắp xếp giảm dần theo điểm tương đồng
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Lấy top_n phim (bỏ qua chính nó ở vị trí index 0)
    return sim_scores[1:top_n+1]

def display_movie_grid(movie_indices, columns=5):
    """Hiển thị danh sách phim dưới dạng lưới"""
    for i in range(0, len(movie_indices), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(movie_indices):
                idx = movie_indices[i+j]
                movie = df.iloc[idx]
                with cols[j]:
                    # Sử dụng placeholder vì dữ liệu hiện tại chưa có link ảnh poster thực tế
                    st.image("https://via.placeholder.com/300x450?text=No+Image", use_container_width=True)
                    st.subheader(movie['title'])
                    st.caption(f"📅 {movie['release_date']} | 🌍 {movie['country'] if pd.notna(movie['country']) else 'N/A'}")
                    st.write(f"🎭 {movie['genre']}")
                    
                    # Nút bấm để "Xem chi tiết & Gợi ý"
                    if st.button(f"Xem phim này", key=f"btn_{idx}"):
                        st.session_state.selected_movie_index = idx
                        st.rerun()

# --- 3. GIAO DIỆN CHÍNH ---

# Nút quay lại trang chủ nếu đang ở trang gợi ý
if st.session_state.selected_movie_index is not None:
    if st.button("⬅️ Quay lại Trang chủ"):
        st.session_state.selected_movie_index = None
        st.rerun()

# LOGIC HIỂN THỊ
if st.session_state.selected_movie_index is None:
    # --- TRANG CHỦ ---
    st.title("🎬 Trang chủ: Phim Mới & Thịnh Hành")
    st.write("Dưới đây là những bộ phim nổi bật nhất dựa trên nội dung bạn quan tâm.")
    
    # Hiển thị mặc định 15 phim đầu tiên (hoặc có thể lấy ngẫu nhiên)
    all_indices = list(range(len(df)))
    display_movie_grid(all_indices[:15])

else:
    # --- TRANG CHI TIẾT & GỢI Ý ---
    selected_idx = st.session_state.selected_movie_index
    current_movie = df.iloc[selected_idx]
    
    st.title(f"🎥 Bạn đang xem: {current_movie['title']}")
    
    # Hiển thị thông tin chi tiết phim đang chọn
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://via.placeholder.com/400x600?text=Poster", use_container_width=True)
    with col2:
        st.header("Thông tin chi tiết")
        st.write(f"**Thể loại:** {current_movie['genre']}")
        st.write(f"**Ngày phát hành:** {current_movie['release_date']}")
        st.write(f"**Quốc gia:** {current_movie['country']}")
        if pd.notna(current_movie['url']):
            st.link_button("Xem Review trên Momo", current_movie['url'])
    
    st.divider()
    
    # GỢI Ý PHIM TƯƠNG TỰ
    st.subheader("✨ Vì bạn quan tâm đến phim này, có thể bạn sẽ thích:")
    rec_results = get_recommendations(selected_idx)
    rec_indices = [x[0] for x in rec_results]
    
    display_movie_grid(rec_indices)

# --- 4. CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        background-color: #ff4b4b;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #ff3333;
        border: 1px solid white;
    }
    </style>
""", unsafe_allow_html=True)