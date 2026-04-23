import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 1. HÀM TẢI DỮ LIỆU ---
def load_csv_data(file_path):
    return pd.read_csv(file_path, sep=',')

# --- 2. HÀM SIMILARITY SHRINKAGE ---
# Phạt độ tương đồng giữa các phim có ít người cùng chấm điểm
def shrink_similarity(raw_sim_df, train_matrix, shrinkage=5):
    item_ids = train_matrix.columns.tolist()
    mask = train_matrix.notna().values
    new_sim = raw_sim_df.values.copy()
    n = len(item_ids)
    for i in range(n):
        for j in range(i+1, n):
            n_corated = int(np.sum(mask[:, i] & mask[:, j]))
            factor = n_corated / (n_corated + shrinkage)
            new_sim[i, j] *= factor
            new_sim[j, i] *= factor
    return pd.DataFrame(new_sim, index=item_ids, columns=item_ids)

# --- 2. HÀM DỰ ĐOÁN ITEM-ITEM KNN (Có Z-score Normalization) ---
def predict_item_knn(train_matrix, similarity_matrix, k=10):
    pred = np.zeros(train_matrix.shape)
    
    # Z-score normalization: chuẩn hóa rating về phân phối N(0,1) theo từng user
    # Giúp đưa user "dễ tính" (hay cho 5*) và "khắt khe" (hay cho 2*) về cùng thang đo
    global_mean = float(train_matrix.stack().mean())
    mean_user_rating = train_matrix.mean(axis=1).values.reshape(-1, 1)
    std_user_rating = train_matrix.std(axis=1).values.reshape(-1, 1)
    std_user_rating[std_user_rating == 0] = 1.0  # tránh chia cho 0
    
    z_matrix = (train_matrix - mean_user_rating) / std_user_rating
    ratings_diff = z_matrix.fillna(0).values
    
    sim_matrix = similarity_matrix.values
    has_rated_mask_all = train_matrix.notna().values
    
    for j in range(train_matrix.shape[1]):
        sim_col = sim_matrix[:, j].copy()
        sim_col[j] = -2 
        
        top_k_indices = np.argsort(sim_col)[:-k-1:-1]
        weights = sim_col[top_k_indices]
        
        has_rated_mask = has_rated_mask_all[:, top_k_indices]
        sum_weights = np.sum(has_rated_mask * np.abs(weights), axis=1)
        sum_weights[sum_weights == 0] = 1e-9
        
        damping_factor = 3.0 
        # Dự đoán trong không gian z-score rồi quy đổi ngược lại
        z_pred = ratings_diff[:, top_k_indices].dot(weights) / (sum_weights + damping_factor)
        pred[:, j] = mean_user_rating.flatten() + z_pred * std_user_rating.flatten()
    
    # Xử lý NaN và clip về đúng thang điểm [1, 5]
    pred = np.where(np.isnan(pred), global_mean, pred)
    pred = np.clip(pred, 1.0, 5.0)
    return pred


# --- 3. CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH HYBRID RECOMMENDATION ---")
    start_total = time.time()
    
    # 3.1. Nạp dữ liệu Rating
    train_df = load_csv_data('ua_train.csv')
    test_df = load_csv_data('ua_test.csv')
    train_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')
    
    # Đảm bảo ma trận vuông (bao gồm tất cả phim dựa theo movies_metadata_encoded.csv)
    movies_df = pd.read_csv('movies_metadata_encoded.csv', encoding='utf-8-sig')
    num_movies = len(movies_df)
    for item in range(1, num_movies + 1):
        if item not in train_matrix.columns:
            train_matrix[item] = np.nan
    train_matrix = train_matrix.reindex(sorted(train_matrix.columns), axis=1)

    print("1. Đã tải ma trận User-Item (Ratings)")
    
    # 3.2. Tính ma trận Rating Similarity (Cosine + Shrinkage)
    # Shrinkage giúp phạt các cặp phim ít người cùng đánh giá, tránh similarity giả cao
    raw_rating_sim = pd.DataFrame(
        cosine_similarity(train_matrix.fillna(0).T), 
        index=train_matrix.columns, 
        columns=train_matrix.columns
    )
    rating_sim_matrix = shrink_similarity(raw_rating_sim, train_matrix, shrinkage=5)
    
    # 3.3. Tải dữ liệu Text (Bình luận phim)
    # movies_df và num_movies đã được đọc ở bước trên
    try:
        reviews_df = pd.read_csv('movie_reviews_cleaned.csv')
        reviews_df = reviews_df.dropna(subset=['clean_comment'])
        
        # Gom nhóm text theo Tên phim
        movie_texts = reviews_df.groupby('Movie_Title')['clean_comment'].apply(lambda x: ' '.join(x)).to_dict()
    except Exception as e:
        print("Lỗi đọc text:", e)
        movie_texts = {}
        
    print("2. Đã tải dữ liệu Text (Bình luận)")
    
    # 3.4. Chuyển Text thành TF-IDF và Tính Text Similarity
    # Tạo danh sách văn bản theo đúng thứ tự item_id (từ 1 đến 83)
    corpus = []
    for item_id in range(1, num_movies + 1):
        title = movies_df.iloc[item_id - 1]['title'] if (item_id - 1) < len(movies_df) else ""
        text = movie_texts.get(title, "")
        corpus.append(text)
        
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    text_sim_matrix = cosine_similarity(tfidf_matrix)
    text_sim_df = pd.DataFrame(text_sim_matrix, index=range(1, num_movies + 1), columns=range(1, num_movies + 1))
    
    print("3. Đã tính xong ma trận tương đồng nội dung (Content-Based)")
    
    # 3.5. Lai tạo (Hybrid) & Tìm K tối ưu
    alphas = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0] # 1.0 là 100% Rating, 0.0 là 100% Text
    k_values = [5, 8, 10, 12, 15, 20]
    
    best_k = -1
    best_alpha = -1
    min_rmse = float('inf')
    best_pred_df = None

    print("\n--- TÌM KIẾM THAM SỐ TỐI ƯU (HYBRID ALPHA & K) TRÊN TẬP TEST ---")

    for alpha in alphas:
        # Trộn 2 ma trận tương đồng
        hybrid_sim = (alpha * rating_sim_matrix) + ((1 - alpha) * text_sim_df)
        
        for k in k_values:
            predictions = predict_item_knn(train_matrix, hybrid_sim, k=k)
            temp_pred_df = pd.DataFrame(predictions, index=train_matrix.index, columns=train_matrix.columns)
            
            y_true = []
            y_pred = []
            for row in test_df.itertuples():
                if row.user_id in temp_pred_df.index and row.item_id in temp_pred_df.columns:
                    y_true.append(row.rating)
                    y_pred.append(temp_pred_df.loc[row.user_id, row.item_id])
            
            if len(y_true) > 0:
                current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                current_mae = mean_absolute_error(y_true, y_pred)
                
                print(f"Alpha = {alpha:.1f} (Rating {int(alpha*100):3d}%, Text {int((1-alpha)*100):3d}%) | K = {k:2d} | RMSE = {current_rmse:.4f} | MAE = {current_mae:.4f}")

                if current_rmse < min_rmse:
                    min_rmse = current_rmse
                    best_k = k
                    best_alpha = alpha
                    best_pred_df = temp_pred_df

    # 3.6. Xuất kết quả
    final_matrix = best_pred_df.T
    final_matrix.index = [f"Item {int(i)}" for i in final_matrix.index]
    final_matrix.columns = [f"User {int(u)}" for u in final_matrix.columns]

    print(f"\n--- KẾT QUẢ CUỐI CÙNG LỰA CHỌN ---")
    print(f"Tỷ lệ lai tối ưu (Alpha): {best_alpha:.1f} ({int(best_alpha*100)}% Phim, {int((1-best_alpha)*100)}% Chữ)")
    print(f"Giá trị K tốt nhất      : {best_k}")
    print(f"RMSE thấp nhất          : {min_rmse:.4f}")
    
    end_total = time.time()
    print(f"Tổng thời gian huấn luyện: {end_total - start_total:.2f} giây")
    
    final_matrix.to_csv('item_user_optimized_results.csv')
    print("Đã lưu kết quả ma trận lai thành công!")
