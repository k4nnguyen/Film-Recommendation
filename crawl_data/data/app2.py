import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error

# --- 1. HÀM TẢI DỮ LIỆU ---
def load_csv_data(file_path):
    # Đọc file CSV đã chia (dấu phẩy)
    return pd.read_csv(file_path, sep=',')

# --- 2. HÀM DỰ ĐOÁN USER-USER KNN ---
def predict_user_user_knn(ratings, similarity, k=40):
    # k=40 là con số phổ biến cho tập MovieLens
    pred = np.zeros(ratings.shape)
    
    # Tính điểm trung bình của mỗi User trong tập Train
    mean_user_rating = ratings.mean(axis=1).values.reshape(-1, 1)
    
    # Độ lệch điểm (chuẩn hóa dựa trên tập Train)
    ratings_diff = (ratings - mean_user_rating).fillna(0).values
    
    for i in range(len(similarity)):
        # Tìm chỉ số của Top K người dùng tương đồng nhất với User i
        top_k_users = np.argsort(similarity.iloc[i, :])[:-k-1:-1]
        
        # Lấy giá trị tương đồng của Top K láng giềng
        sim_top_k = similarity.iloc[i, top_k_users].values
        
        # Phép tính trung bình có trọng số chỉ trên Top K
        sum_sim = np.abs(sim_top_k).sum()
        if sum_sim != 0:
            pred[i, :] = mean_user_rating[i] + sim_top_k.dot(ratings_diff[top_k_users, :]) / sum_sim
        else:
            # Nếu không có ai tương đồng, dùng điểm trung bình của chính User đó
            pred[i, :] = mean_user_rating[i].flatten()
            
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Nạp dữ liệu từ 2 file riêng biệt
    train_df = load_csv_data('ua_train.csv')
    test_df = load_csv_data('ua_test.csv')
    
    print(f"Dữ liệu huấn luyện: {len(train_df)} dòng")
    print(f"Dữ liệu kiểm thử: {len(test_df)} dòng")

    # Tạo ma trận User-Item từ tập TRAIN (Hàng: User, Cột: Item)
    train_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')

    # --- TRAIN: Tính độ tương quan (Chỉ dùng tập Train) ---
    start_train = time.time()
    # Chuyển vị để tính tương quan giữa các User
    user_similarity = train_matrix.T.corr(method='pearson').fillna(0)
    end_train = time.time()
    
    # --- TÌM K TỐI ƯU ---
    k_values = [10, 20, 30, 40, 50, 60, 70]
    rmse_results = []
    best_pred_matrix = None
    min_rmse = float('inf')
    best_k = k_values[0]

    print(f"--- BẮT ĐẦU TÌM K TỐI ƯU (User-User KNN) ---")
    
    for k in k_values:
        # Dự đoán với giá trị k hiện tại dựa trên tri thức từ tập Train
        current_pred = predict_user_user_knn(train_matrix, user_similarity, k=k)
        
        # Tạo DataFrame tạm thời để tra cứu kết quả
        temp_full_matrix = pd.DataFrame(current_pred, index=train_matrix.index, columns=train_matrix.columns)
        
        # --- TÍNH TOÁN RMSE TRÊN TẬP TEST ---
        y_true = []
        y_pred = []
        for row in test_df.itertuples():
            # Chỉ tính RMSE trên những cặp (User, Item) thực tế có trong tập TEST
            if row.user_id in temp_full_matrix.index and row.item_id in temp_full_matrix.columns:
                y_true.append(row.rating)
                y_pred.append(temp_full_matrix.loc[row.user_id, row.item_id])
        
        if len(y_true) > 0:
            current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rmse_results.append(current_rmse)
            print(f"K = {k:2d}: RMSE = {current_rmse:.4f}")

            # Lưu lại kết quả tốt nhất
            if current_rmse < min_rmse:
                min_rmse = current_rmse
                best_k = k
                best_pred_matrix = temp_full_matrix

    # --- XỬ LÝ KẾT QUẢ TỐT NHẤT ---
    # Đảo ngược ma trận (Hàng là Item, Cột là User)
    final_matrix = best_pred_matrix.T
    
    # Đổi tên nhãn sang định dạng "User X", "Item Y"
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]

    print(f"\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Giá trị K tốt nhất: {best_k}")
    print(f"RMSE thấp nhất trên tập Test: {min_rmse:.4f}")
    print(f"Thời gian tính tương đồng (Train): {end_train - start_train:.4f} giây")
    
    # Lưu file kết quả tối ưu
    final_matrix.to_csv('user_user_optimized_results.csv')
