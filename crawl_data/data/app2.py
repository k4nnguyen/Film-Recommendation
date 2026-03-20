import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error

# --- 1. TẢI DỮ LIỆU ---
def load_data(file_path):
    names = ['user_id', 'item_id', 'rating']
    df = pd.read_csv(file_path, sep='\s+', names=names, usecols=[0, 1, 2], engine='python')
    return df

def predict_user_user_knn(ratings, similarity, k=40):
    # k=40 là con số phổ biến cho tập MovieLens
    pred = np.zeros(ratings.shape)
    
    # Tính điểm trung bình của mỗi User
    mean_user_rating = ratings.mean(axis=1).values.reshape(-1, 1)
    
    # Độ lệch điểm (chuẩn hóa)
    ratings_diff = (ratings - mean_user_rating).fillna(0).values
    
    for i in range(len(similarity)):
        # Tìm chỉ số của Top K người dùng tương đồng nhất với User i
        # np.argsort trả về chỉ số tăng dần, ta lấy phần cuối và đảo ngược
        top_k_users = np.argsort(similarity.iloc[i, :])[:-k-1:-1]
        
        # Lấy giá trị tương đồng của Top K láng giềng
        sim_top_k = similarity.iloc[i, top_k_users].values
        
        # Phép tính trung bình có trọng số chỉ trên Top K
        sum_sim = np.abs(sim_top_k).sum()
        if sum_sim != 0:
            # [1xK] dot [Kx83] = [1x83]
            pred[i, :] = mean_user_rating[i] + sim_top_k.dot(ratings_diff[top_k_users, :]) / sum_sim
        else:
            # Nếu không có ai tương đồng, dùng điểm trung bình của chính User đó
            pred[i, :] = mean_user_rating[i].flatten()
            
    return pred

# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    df = load_data('u.data')
    
    # Ma trận User-Item (Hàng: User, Cột: Item)
    user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

    # --- ĐO THỜI GIAN TRAIN (Tính User-User Similarity) ---
    start_train = time.time()
    user_similarity = user_item_matrix.T.corr(method='pearson').fillna(0)
    end_train = time.time()
    
    # --- TÌM K TỐI ƯU ---
    k_values = [10, 20, 30, 40, 50, 60, 70]
    rmse_results = []
    best_pred_matrix = None
    min_rmse = float('inf')
    best_k = k_values[0]

    print(f"--- BẮT ĐẦU TÌM K TỐI ƯU (User-User) ---")
    
    for k in k_values:
        # Dự đoán với giá trị k hiện tại
        current_pred = predict_user_user_knn(user_item_matrix, user_similarity, k=k)
        
        # Tạo DataFrame tạm thời (vẫn dùng ID số để tính toán cho nhanh)
        temp_full_matrix = pd.DataFrame(current_pred, index=user_item_matrix.index, columns=user_item_matrix.columns)
        
        # --- TÍNH TOÁN RMSE NHANH ---
        y_true = []
        y_pred = []
        for row in df.itertuples():
            y_true.append(row.rating)
            # Truy xuất trực tiếp bằng ID số (row.user_id, row.item_id)
            y_pred.append(temp_full_matrix.loc[row.user_id, row.item_id])
        
        current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_results.append(current_rmse)
        print(f"K = {k}: RMSE = {current_rmse:.4f}")

        # Lưu lại kết quả tốt nhất
        if current_rmse < min_rmse:
            min_rmse = current_rmse
            best_k = k
            best_pred_matrix = temp_full_matrix

    # --- KẾT THÚC VÒNG LẶP - XỬ LÝ KẾT QUẢ TỐT NHẤT ---
    
    # Sử dụng ma trận của giá trị K tốt nhất để tạo file cuối cùng
    # Đảo ngược ma trận (Hàng là Item, Cột là User) theo yêu cầu của anh/chị
    final_matrix = best_pred_matrix.T
    
    # Đổi tên nhãn sang định dạng "User X", "Item Y" cho file báo cáo
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]

    print(f"\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Giá trị K tốt nhất: {best_k}")
    print(f"RMSE thấp nhất: {min_rmse:.4f}")
    print(f"Thời gian tính tương đồng (Train): {end_train - start_train:.4f} giây")
    
    
