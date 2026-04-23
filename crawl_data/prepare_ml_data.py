import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Fix Vietnamese output trên Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DATA_DIR = 'data'
METADATA_FILE = os.path.join(DATA_DIR, 'movies_metadata_encoded.csv')

# Các file đầu ra
U_DATA_FILE = os.path.join(DATA_DIR, 'u.data')
U_INFO_FILE = os.path.join(DATA_DIR, 'u.info')
U_GENRE_FILE = os.path.join(DATA_DIR, 'u.genre')
TRAIN_FILE = os.path.join(DATA_DIR, 'ua_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'ua_test.csv')

# ===========================================================================
# Định nghĩa các PERSONA (nhóm sở thích thể loại)
# Mỗi persona là dict: tên_thể_loại -> weight (0.0 – 1.0)
# User sẽ được gán ngẫu nhiên vào 1 persona.
# Thể loại không được đề cập sẽ có weight baseline thấp (0.2).
# ===========================================================================
PERSONAS = [
    # Nhóm 1: Mê hành động & phiêu lưu
    {
        "Hành động": 1.0, "Phiêu lưu": 0.9, "Gay cấn": 0.8,
        "Tội phạm": 0.6, "Giật gân": 0.6, "Hình sự": 0.5,
    },
    # Nhóm 2: Mê tình cảm & lãng mạn
    {
        "Tình cảm": 1.0, "Lãng mạn": 0.9, "Gia đình": 0.7,
        "Hài": 0.6, "Chính kịch": 0.5, "Drama": 0.5,
    },
    # Nhóm 3: Mê kinh dị & bí ẩn
    {
        "Kinh dị": 1.0, "Ma": 0.9, "Bí ẩn": 0.8,
        "Tâm Linh": 0.7, "Giật gân": 0.7, "Tội phạm": 0.4,
    },
    # Nhóm 4: Mê khoa học viễn tưởng & siêu anh hùng
    {
        "Khoa học - Viễn tưởng": 1.0, "Siêu anh hùng": 0.9,
        "Giả tưởng": 0.8, "Phiêu lưu": 0.6, "Hành động": 0.5,
    },
    # Nhóm 5: Mê hài & gia đình
    {
        "Hài": 1.0, "Gia đình": 0.9, "Hoạt hình": 0.8,
        "Tình cảm": 0.5, "Lãng mạn": 0.4,
    },
    # Nhóm 6: Mê tâm lý & drama nghệ thuật
    {
        "Tâm lý": 1.0, "Drama": 0.9, "Chính kịch": 0.8,
        "Trinh thám": 0.7, "Hình sự": 0.6, "Bí ẩn": 0.5,
    },
    # Nhóm 7: Mê lịch sử & chiến tranh
    {
        "Lịch sử": 1.0, "Chiến tranh": 0.9, "Tài liệu": 0.7,
        "Chính kịch": 0.6, "Cổ Trang": 0.6,
    },
    # Nhóm 8: Người xem đa dạng (xem nhiều thể loại, ít thiên lệch)
    {},  # empty -> tất cả genre có weight baseline bằng nhau
]

# Tỉ lệ phân bổ user vào từng persona
# (để một số nhóm phổ biến hơn, ví dụ tình cảm/hành động chiếm nhiều hơn)
PERSONA_WEIGHTS = [0.18, 0.20, 0.14, 0.12, 0.12, 0.10, 0.08, 0.06]


def generate_mock_data():
    print("=" * 60)
    print("=== BẮT ĐẦU TẠO DỮ LIỆU HUẤN LUYỆN (MOCK DATA) ===")
    print("=== Chế độ: Persona-Based (Sở thích theo thể loại)  ===")
    print("=" * 60)

    if not os.path.exists(METADATA_FILE):
        print(f"Lỗi: Không tìm thấy file {METADATA_FILE}.")
        print("Vui lòng chạy 'Thu thập phim' trong data_pipeline.py trước.")
        return

    movies_df = pd.read_csv(METADATA_FILE, encoding='utf-8-sig')
    num_movies = len(movies_df)
    print(f"[1/5] Số lượng phim tìm thấy: {num_movies}")

    # Xác định cột thể loại (loại bỏ các cột metadata cơ bản)
    base_cols = {'title', 'genre', 'release_date', 'country', 'url', 'poster_url'}
    genre_columns = [col for col in movies_df.columns if col not in base_cols]
    genre_matrix = movies_df[genre_columns].fillna(0).values.astype(float)
    # Shape: (num_movies, num_genres)

    # --- BƯỚC 1: TẠO USER RATINGS (u.data) theo persona ---
    print(f"[2/5] Đang tạo dữ liệu đánh giá giả lập theo sở thích thể loại (u.data)...")
    num_users = 500
    min_ratings = 8
    max_ratings = 20
    np.random.seed(42)

    baseline_weight = 0.2  # weight mặc định cho thể loại không nằm trong persona

    def build_user_weights(persona: dict) -> np.ndarray:
        """Chuyển persona dict thành vector weight theo thứ tự genre_columns."""
        weights = np.full(len(genre_columns), baseline_weight)
        for i, g in enumerate(genre_columns):
            if g in persona:
                weights[i] = persona[g]
        return weights

    def movie_preference_score(movie_idx: int, user_weights: np.ndarray) -> float:
        """
        Tính điểm ưa thích của user với phim theo genre overlap.
        Trả về giá trị trong [0, 1].
        """
        genre_vec = genre_matrix[movie_idx]
        total_weight = genre_vec.sum()
        if total_weight == 0:
            return 0.4  # phim không có thể loại -> trung tính thấp
        return float(np.dot(genre_vec, user_weights) / total_weight)

    # Gán persona cho từng user (theo tỉ lệ PERSONA_WEIGHTS)
    persona_ids = np.random.choice(
        len(PERSONAS), size=num_users, p=PERSONA_WEIGHTS
    )

    data = []
    for user_id in range(1, num_users + 1):
        persona = PERSONAS[persona_ids[user_id - 1]]
        user_weights = build_user_weights(persona)

        # Tính preference score cho tất cả phim
        scores = np.array([movie_preference_score(i, user_weights) for i in range(num_movies)])

        # Chọn phim để rate: xác suất tỉ lệ với preference score
        # (user thích thể loại nào thì xem nhiều thể loại đó hơn)
        prob = scores / scores.sum()
        num_rated = np.random.randint(min_ratings, max_ratings + 1)
        num_rated = min(num_rated, num_movies)
        rated_indices = np.random.choice(num_movies, size=num_rated, replace=False, p=prob)

        for idx in rated_indices:
            item_id = idx + 1  # item_id bắt đầu từ 1
            preference = scores[idx]  # trong [0, 1]

            # Map preference [0, 1] -> base rating [1, 5] + nhiễu Gaussian
            # preference cao -> rating cao; nhiễu ±0.6 sao tạo tính tự nhiên
            base_rating = 1.0 + preference * 4.0
            noise = np.random.normal(0, 0.65)
            rating = int(np.clip(round(base_rating + noise), 1, 5))
            data.append([user_id, item_id, rating])

    generated_u_data = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    generated_u_data.to_csv(U_DATA_FILE, sep='\t', index=False, header=False)
    print(f"      -> Đã tạo {len(generated_u_data)} đánh giá cho {num_users} users (persona-based).")

    # Thống kê phân bổ persona để kiểm tra
    unique, counts = np.unique(persona_ids, return_counts=True)
    persona_names = [
        "Hành động/Phiêu lưu", "Tình cảm/Lãng mạn", "Kinh dị/Bí ẩn",
        "KH Viễn tưởng", "Hài/Gia đình", "Tâm lý/Drama", "Lịch sử/CT", "Đa dạng"
    ]
    print("      Phân bổ persona:")
    for pid, cnt in zip(unique, counts):
        print(f"        - {persona_names[pid]}: {cnt} users")

    # --- BƯỚC 2: TẠO USER INFO (u.info) ---
    print(f"[3/5] Đang tạo thông tin tài khoản (u.info)...")
    user_info_data = []
    for i in range(1, num_users + 1):
        user_info_data.append([i, f"user{i}", f"user{i}"])
    pd.DataFrame(user_info_data).to_csv(U_INFO_FILE, sep='\t', index=False, header=False)

    # --- BƯỚC 3: TRÍCH XUẤT THỂ LOẠI (u.genre) ---
    print(f"[4/5] Đang trích xuất danh sách thể loại (u.genre)...")
    genre_data = [['unknown', 0]]
    for i, genre_name in enumerate(genre_columns):
        genre_data.append([genre_name, i + 1])
    pd.DataFrame(genre_data).to_csv(U_GENRE_FILE, sep='|', index=False, header=False)

    # --- BƯỚC 4: CHIA TRAIN/TEST (ua_train, ua_test) ---
    print(f"[5/5] Đang chia tập dữ liệu Train/Test (70/30)...")
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(U_DATA_FILE, sep=r'\s+', names=names, usecols=[0, 1, 2], engine='python')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    train_df.to_csv(TRAIN_FILE, index=False, sep=',')
    test_df.to_csv(TEST_FILE, index=False, sep=',')

    print("\n" + "=" * 60)
    print("=== HOÀN TẤT TẠO DỮ LIỆU ===")
    print(f"- Dữ liệu gốc       : {U_DATA_FILE}")
    print(f"- Dữ liệu tài khoản : {U_INFO_FILE}")
    print(f"- Danh sách thể loại: {U_GENRE_FILE}")
    print(f"- Tập Train (70%)   : {len(train_df)} dòng -> {TRAIN_FILE}")
    print(f"- Tập Test  (30%)   : {len(test_df)} dòng -> {TEST_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_mock_data()
