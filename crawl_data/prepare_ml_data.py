import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DATA_DIR = 'data'
METADATA_FILE = os.path.join(DATA_DIR, 'movies_metadata_encoded.csv')

# Các file đầu ra
U_DATA_FILE = os.path.join(DATA_DIR, 'u.data')
U_INFO_FILE = os.path.join(DATA_DIR, 'u.info')
U_GENRE_FILE = os.path.join(DATA_DIR, 'u.genre')
TRAIN_FILE = os.path.join(DATA_DIR, 'ua_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'ua_test.csv')

def generate_mock_data():
    print("=" * 50)
    print("=== BẮT ĐẦU TẠO DỮ LIỆU HUẤN LUYỆN (MOCK DATA) ===")
    print("=" * 50)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Lỗi: Không tìm thấy file {METADATA_FILE}.")
        print("Vui lòng chạy 'Thu thập phim' trong data_pipeline.py trước.")
        return
        
    movies_df = pd.read_csv(METADATA_FILE)
    num_movies = len(movies_df)
    print(f"[1/4] Số lượng phim tìm thấy: {num_movies}")
    
    # --- BƯỚC 1: TẠO USER RATINGS (u.data) ---
    print(f"[2/4] Đang tạo dữ liệu đánh giá giả lập (u.data)...")
    num_users = 500
    min_ratings = 5
    max_ratings = 15
    np.random.seed(42)
    
    item_ids = list(range(1, num_movies + 1))
    data = []
    
    for user_id in range(1, num_users + 1):
        num_rated = np.random.randint(min_ratings, max_ratings + 1)
        rated_movies = np.random.choice(item_ids, size=num_rated, replace=False)
        for item_id in rated_movies:
            rating = np.random.randint(1, 6)
            data.append([user_id, item_id, rating])
            
    generated_u_data = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    generated_u_data.to_csv(U_DATA_FILE, sep='\t', index=False, header=False)
    print(f"      -> Đã tạo {len(generated_u_data)} đánh giá cho {num_users} users.")
    
    # --- BƯỚC 2: TẠO USER INFO (u.info) ---
    print(f"[3/4] Đang tạo thông tin tài khoản (u.info)...")
    user_info_data = []
    for i in range(1, num_users + 1):
        user_info_data.append([i, f"user{i}", f"user{i}"])
    pd.DataFrame(user_info_data).to_csv(U_INFO_FILE, sep='\t', index=False, header=False)
    
    # --- BƯỚC 3: TRÍCH XUẤT THỂ LOẠI (u.genre) ---
    print(f"[4/4] Đang trích xuất danh sách thể loại (u.genre)...")
    # Các cột từ vị trí index 5 trở đi là thể loại
    # (Nếu cột metadata thay đổi, có thể cần điều chỉnh lại index này)
    genre_columns = [col for col in movies_df.columns[5:] if col not in ['url', 'poster_url']]
    
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
    
    print("\n" + "=" * 50)
    print("=== HOÀN TẤT TẠO DỮ LIỆU ===")
    print(f"- Dữ liệu gốc       : {U_DATA_FILE}")
    print(f"- Dữ liệu tài khoản : {U_INFO_FILE}")
    print(f"- Danh sách thể loại: {U_GENRE_FILE}")
    print(f"- Tập Train (70%)   : {len(train_df)} dòng -> {TRAIN_FILE}")
    print(f"- Tập Test (30%)    : {len(test_df)} dòng -> {TEST_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    # Đảm bảo thư mục data tồn tại
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_mock_data()
