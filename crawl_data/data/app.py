import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error

# --- 1. TẢI DỮ LIỆU ---
def load_data(file_path):
    # u.data có cấu trúc: user_id | item_id | rating
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(file_path, sep='\s+', names=names, usecols=[0, 1, 2], engine='python')
    return df

# --- 2. HÀM DỰ ĐOÁN (ĐIỀN DẤU "?") ---
def predict(ratings, similarity):
    # Tính toán dự đoán dựa trên trọng số tương đồng
    mean_user_rating = ratings.mean(axis=1).values.reshape(-1, 1)
    ratings_diff = (ratings - mean_user_rating).fillna(0)
    pred = mean_user_rating + ratings_diff.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    df = load_data('u.data')
    
    # Tạo ma trận User-Item ban đầu (Hàng: User, Cột: Item)
    # Chúng ta làm vậy để dùng hàm .corr() tính Item-Item dễ dàng nhất
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

    # --- ĐO THỜI GIAN TRAIN (Tính Similarity) ---
    start_train = time.time()
    # corr() tính toán sự tương quan giữa các Cột (Item)
    item_similarity = user_item_matrix.corr(method='pearson').fillna(0)
    end_train = time.time()
    
    # --- ĐIỀN DẤU "?" (Inference) ---
    start_pred = time.time()
    # Dự đoán toàn bộ các ô trống
    predicted_ratings = predict(user_item_matrix, item_similarity)
    
    # Tạo DataFrame kết quả (Vẫn đang là User x Item)
    full_matrix = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
    
    # --- ĐỔI VAI TRÒ: HÀNG LÀ ITEM, CỘT LÀ USER ---
    # Sử dụng phép chuyển vị .T
    final_matrix = full_matrix.T
    
    # Thêm nhãn "Item" và "User" cho chuyên nghiệp
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]
    end_pred = time.time()

    # --- TÍNH TOÁN METRICS (RMSE) ---
    # So sánh các ô đã có dữ liệu trong u.data với giá trị dự đoán
    y_true = []
    y_pred = []
    for row in df.itertuples():
        u_label = f"User {row.user_id}"
        i_label = f"Item {row.item_id}"
        if u_label in final_matrix.columns and i_label in final_matrix.index:
            y_true.append(row.rating)
            y_pred.append(final_matrix.loc[i_label, u_label])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # --- XUẤT KẾT QUẢ ---
    print(f"--- BÁO CÁO HUẤN LUYỆN ---")
    print(f"Thời gian tính toán độ tương đồng (Train): {end_train - start_train:.4f} giây")
    print(f"Thời gian điền đầy ma trận (Inference): {end_pred - start_pred:.4f} giây")
    print(f"Chỉ số lỗi RMSE: {rmse:.4f}")
    
    item_similarity.to_csv('item_item_matrix.csv')
    final_matrix.to_csv('item_user_final.csv')
    print("\nĐã lưu ma trận (Hàng: Item, Cột: User) vào file: item_user_final.csv")