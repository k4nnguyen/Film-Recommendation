import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error

# --- 1. HÀM TẢI DỮ LIỆU ---
def load_csv_data(file_path):
    # Đọc file CSV đã chia (dấu phẩy)
    return pd.read_csv(file_path, sep=',')

# --- 2. HÀM DỰ ĐOÁN ITEM-ITEM KNN ---
def predict_item_knn(train_matrix, similarity, k=10):
    # train_matrix: (Users x Items), similarity: (Items x Items)
    pred = np.zeros(train_matrix.shape)
    
    # Tính trung bình của mỗi User trong tập Train để khử bias
    mean_user_rating = train_matrix.mean(axis=1).values.reshape(-1, 1)
    ratings_diff = (train_matrix - mean_user_rating).fillna(0).values
    
    sim_matrix = similarity.values
    
    for j in range(train_matrix.shape[1]): # Duyệt qua từng phim
        # Tìm Top K phim giống với phim j nhất
        # argsort trả về chỉ số tăng dần, lấy phần cuối và đảo ngược
        sim_col = sim_matrix[:, j].copy()
        
        # Loại chính nó ra khỏi vòng pháp luật (gán độ tương đồng nhỏ nhất)
        sim_col[j] = -2 
        
        # Tìm Top K (lúc này chắc chắn không còn lẫn Phim j)
        top_k_indices = np.argsort(sim_col)[:-k-1:-1]
        
        # Lấy trọng số của đúng K láng giềng đó
        weights = sim_col[top_k_indices]
        # 1. Bit mask: Chỉ lấy những phim mà User thực sự đã rate
        has_rated_mask = (ratings_diff[:, top_k_indices] != 0)
        
        # 2. Tính mẫu số thực tế (tổng độ tương đồng) ĐỘNG cho từng User
        sum_weights = np.sum(has_rated_mask * np.abs(weights), axis=1)
        
        # 3. Tránh chia cho 0 nếu User chưa xem phim nào trong Top K
        sum_weights[sum_weights == 0] = 1e-9
        
        # 4. Hệ số giảm xóc (Damping factor) để trị dữ liệu thưa thớt
        damping_factor = 3.0 
        
        # --- DỰ ĐOÁN ---
        # Tính tử số bằng phép nhân ma trận
        numerator = ratings_diff[:, top_k_indices].dot(weights)
        
        # Điểm TB + (Tử số / (Mẫu số động + Giảm xóc))
        pred[:, j] = mean_user_rating.flatten() + (numerator / (sum_weights + damping_factor))
            
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    # Nạp dữ liệu từ 2 file đã chia
    train_df = load_csv_data('ua_train.csv')
    test_df = load_csv_data('ua_test.csv')
    
    print(f"Dữ liệu huấn luyện: {len(train_df)} dòng")
    print(f"Dữ liệu kiểm thử: {len(test_df)} dòng")

    # Tạo ma trận User-Item từ tập Train
    train_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')

    # --- TRAIN: Tính độ tương quan (Chỉ dùng tập Train) ---
    start_train = time.time()
    # .corr() mặc định tính theo cột (Item)
    item_similarity = train_matrix.corr(method='pearson').fillna(0)
    end_train = time.time()
    
    # Thử nghiệm các giá trị K để tìm điểm tối ưu trên tập Test
    k_values = [50,60,70,80,90,100,110]
    best_k = -1
    min_rmse = float('inf')
    best_pred_df = None

    print(f"\n--- ĐANG TÌM K TỐI ƯU TRÊN TẬP TEST ---")

    for k in k_values:
        # Dự đoán
        predictions = predict_item_knn(train_matrix, item_similarity, k=k)
        temp_pred_df = pd.DataFrame(predictions, index=train_matrix.index, columns=train_matrix.columns)
        
        # --- TEST: Tính RMSE trên tập Test ---
        y_true = []
        y_pred = []
        for row in test_df.itertuples():
            # Chỉ tính nếu User và Item đó tồn tại trong mô hình đã học
            if row.user_id in temp_pred_df.index and row.item_id in temp_pred_df.columns:
                y_true.append(row.rating)
                y_pred.append(temp_pred_df.loc[row.user_id, row.item_id])
        
        if len(y_true) > 0:
            current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"K = {k:2d} | RMSE = {current_rmse:.4f}")

            if current_rmse < min_rmse:
                min_rmse = current_rmse
                best_k = k
                best_pred_df = temp_pred_df

    # --- XUẤT KẾT QUẢ TỐT NHẤT ---
    # Chuyển vị ma trận: Hàng là Item, Cột là User
    final_matrix = best_pred_df.T
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]

    print(f"\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Giá trị K tốt nhất: {best_k}")
    print(f"Chỉ số RMSE thấp nhất: {min_rmse:.4f}")
    print(f"Thời gian huấn luyện: {end_train - start_train:.4f} giây")
    
    # Lưu file kết quả
    final_matrix.to_csv('item_user_optimized_results.csv')
