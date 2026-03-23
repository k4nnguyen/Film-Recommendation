import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# --- 1. TẢI DỮ LIỆU ---
def load_data(file_path):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(file_path, sep='\s+', names=names, usecols=[0, 1, 2], engine='python')
    return df

# --- 2. HÀM DỰ ĐOÁN (ĐIỀN DẤU "?") ---
def predict(ratings, similarity):
    # Với Cosine, chúng ta dùng trực tiếp ratings thay vì trừ đi trung bình (vì Cosine không tự chuẩn hóa)
    ratings_filled = ratings.fillna(0).values
    pred = ratings_filled.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    df = load_data('u.data')
    
    # Tạo ma trận User-Item (Hàng: User, Cột: Item)
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

    # --- ĐO THỜI GIAN TRAIN (Tính Cosine Similarity) ---
    start_train = time.time()
    
    # Bước cực kỳ quan trọng: Cosine không xử lý được NaN, nên phải fill bằng 0
    # .T để tính tương quan giữa các Item (Cột)
    item_sim_matrix = cosine_similarity(user_item_matrix.fillna(0).T)
    
    # Chuyển về DataFrame để dễ xử lý
    item_similarity = pd.DataFrame(item_sim_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    end_train = time.time()
    
    # --- ĐIỀN DẤU "?" (Inference) ---
    start_pred = time.time()
    predicted_ratings = predict(user_item_matrix, item_similarity)
    
    # Tạo DataFrame kết quả
    full_matrix = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    # Đảo ngược ma trận theo yêu cầu của anh/chị (Hàng: Item, Cột: User)
    final_matrix = full_matrix.T
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]
    end_pred = time.time()

    # --- TÍNH TOÁN RMSE ---
    y_true = []
    y_pred = []
    for row in df.itertuples():
        u_label = f"User {row.user_id}"
        i_label = f"Item {row.item_id}"
        if u_label in final_matrix.columns and i_label in final_matrix.index:
            y_true.append(row.rating)
            y_pred.append(final_matrix.loc[i_label, u_label])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"--- KẾT QUẢ DÙNG COSINE SIMILARITY ---")
    print(f"Thời gian tính toán (Train): {end_train - start_train:.4f} giây")
    print(f"Chỉ số lỗi RMSE: {rmse:.4f}")