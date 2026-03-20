import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error

# --- 1. TẢI DỮ LIỆU ---
def load_data(file_path):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(file_path, sep='\s+', names=names, usecols=[0, 1, 2], engine='python')
    return df

# --- 2. HÀM DỰ ĐOÁN USER-USER ---
def predict_user_user(ratings, similarity):
    
    mean_user_rating = ratings.mean(axis=1)
    
    ratings_diff = ratings.sub(mean_user_rating, axis=0).fillna(0)

    dot_product = similarity.dot(ratings_diff)
    
    # 4. Tính mẫu số: Tổng độ tương đồng của mỗi User (Trị tuyệt đối)
    sum_of_sim = np.abs(similarity).sum(axis=1)
    
    # 5. Phép chia có trọng số: [504x83] / [504x1]
    # Dùng .div(axis=0) để tránh lỗi "operands could not be broadcast"
    weighted_diff = dot_product.div(sum_of_sim, axis=0).fillna(0)
    
    # 6. Cộng lại điểm trung bình để ra kết quả cuối
    pred = weighted_diff.add(mean_user_rating, axis=0)
    
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    df = load_data('u.data')
    
    # Ma trận User-Item (Hàng: User, Cột: Item)
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

    # --- ĐO THỜI GIAN TRAIN (Tính User-User Similarity) ---
    start_train = time.time()
    
    # Để tính User-User, chúng ta cần tính tương quan giữa các HÀNG.
    # Vì .corr() tính theo CỘT, nên ta phải chuyển vị (.T) trước khi tính.
    user_similarity = user_item_matrix.T.corr(method='pearson').fillna(0)
    
    end_train = time.time()
    
    # --- ĐIỀN DẤU "?" (Inference) ---
    start_pred = time.time()
    predicted_ratings = predict_user_user(user_item_matrix, user_similarity)
    
    # Tạo DataFrame kết quả (Vẫn là User x Item)
    full_matrix = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    # Đổi vai trò theo ý anh/chị (Hàng là Item, Cột là User)
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
        y_true.append(row.rating)
        y_pred.append(final_matrix.loc[i_label, u_label])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"--- KẾT QUẢ USER-USER COLLABORATIVE FILTERING ---")
    print(f"Kích thước ma trận tương đồng: {user_similarity.shape}")
    print(f"Thời gian tính tương đồng (Train): {end_train - start_train:.4f} giây")
    print(f"Chỉ số lỗi RMSE: {rmse:.4f}")
    
