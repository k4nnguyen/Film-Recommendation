from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import asyncio
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import os 
from typing import List
import sqlite3
import bcrypt
import datetime 
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Dùng backend không cần GUI/Tkinter, an toàn cho multi-thread
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = FastAPI()
executor = ProcessPoolExecutor(max_workers=1)
df = pd.read_csv('../../crawl_data/data/movies_metadata_encoded.csv')
features = df.select_dtypes(include=['number'])
sim_matrix = cosine_similarity(features)

# Load reviews dataset
try:
    raw_reviews_df = pd.read_csv('../../crawl_data/data/movie_reviews.csv')
    reviews_df = pd.read_csv('../../crawl_data/data/movie_reviews_cleaned.csv')
    # Thêm cột bình luận gốc vào reviews_df
    reviews_df['original_comment'] = raw_reviews_df['comment']
except Exception as e:
    print("Lỗi đọc file reviews:", e)
    reviews_df = pd.DataFrame(columns=['rating', 'clean_comment', 'Movie_Title', 'original_comment'])


last_update = "Đang khởi tạo..."
try:
    cf_path = '../../crawl_data/data/item_user_optimized_results.csv'
    if not os.path.exists(cf_path):
        cf_path = 'item_user_optimized_results.csv'
        
    # Đọc file CF, lấy cột "Item X" làm index để dữ liệu chỉ còn lại các con số
    cf_df = pd.read_csv(cf_path, index_col=0)
    
    # Tính ma trận tương đồng giữa các PHIM dựa trên hành vi người dùng (Item-Item CF)
    cf_sim_matrix = cosine_similarity(cf_df)
except Exception as e:
    print("Lỗi đọc file CF:", e)
    cf_sim_matrix = []
    
# =============== 
#     login.py
# ===============

class UserAuth(BaseModel):
    username: str
    password: str

@app.post("/register") # Cho phần đăng ký 
def register_user(user: UserAuth):
    # Dùng mật khẩu đã mã hóa 
    hashed_pw = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') 
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')

    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (user.username, hashed_pw))
        conn.commit()
        # Tạo ID và lưu vào u.info
        info_path = '../../crawl_data/data/u.info'
        next_id = 1
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if lines:
                    last_id = int(lines[-1].split('\t')[0])
                    next_id = last_id + 1
        with open(info_path, 'a', encoding='utf-8') as f:
            f.write(f"{next_id}\t{user.username}\t{user.password}\n")
        return {"status": "success", "message": "Đăng ký thành công"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")
    finally:
        conn.close()

@app.post("/login") 
def login_user(user: UserAuth):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    c.execute('SELECT password FROM users WHERE username = ?', (user.username,))
    data = c.fetchone()
    conn.close()
    
    if data and bcrypt.checkpw(user.password.encode('utf-8'), data[0].encode('utf-8')):
        info_path = "../../crawl_data/data/u.info"
        try:
            u_info = pd.read_csv(info_path, sep='\t', names=['user_id', 'username', 'password'], encoding='utf-8-sig')
            user_row = u_info[u_info['username'] == user.username]
            user_id = int(user_row.iloc[0]['user_id'])
            return {"status": "success", "user_id": user_id, "username": user.username}
        except Exception:
            raise HTTPException(status_code=500, detail="Lỗi hệ thống data")
    raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu")
# =============== 
#     main.py
# ===============

# ======= Vong lap de auto update real time =======
DATA_DIR = "../../crawl_data/data/"
movies_df = None
predictions_df = None # Lưu ma trận dự đoán (Users x Items)
u_info = None

def predict_item_knn(train_matrix, similarity_matrix, k=10):
    pred = np.zeros(train_matrix.shape)
    mean_user_rating = train_matrix.mean(axis=1).values.reshape(-1, 1)
    ratings_diff = (train_matrix - mean_user_rating).fillna(0).values
    
    sim_matrix_vals = similarity_matrix.values
    # Sửa lỗi: dùng notna() để xác định user đã xem, không dùng != 0
    has_rated_mask_all = train_matrix.notna().values
    
    for j in range(train_matrix.shape[1]):
        sim_col = sim_matrix_vals[:, j].copy()
        sim_col[j] = -2 
        top_k_indices = np.argsort(sim_col)[:-k-1:-1]
        weights = sim_col[top_k_indices]
        
        has_rated_mask = has_rated_mask_all[:, top_k_indices]
        sum_weights = np.sum(has_rated_mask * np.abs(weights), axis=1)
        sum_weights[sum_weights == 0] = 1e-9
        
        damping_factor = 3.0 
        numerator = ratings_diff[:, top_k_indices].dot(weights)
        pred[:, j] = mean_user_rating.flatten() + (numerator / (sum_weights + damping_factor))
    return pred

def update_recommender_system():
    global movies_df, predictions_df, u_info, last_update
    print("  Đang huấn luyện lại mô hình Item-Item KNN...")

    try:
        # Load dữ liệu phim để hiển thị tên
        movies_df = pd.read_csv(DATA_DIR + 'movies_metadata_encoded.csv', encoding='utf-8-sig')
        u_info = pd.read_csv(DATA_DIR + 'u.info', sep='\t', names=['user_id', 'username', 'password'])
        
        # Load u.data 
        raw_data = pd.read_csv(DATA_DIR + 'u.data', sep='\t', names=['user_id', 'movie_id', 'rating'])
        raw_data = raw_data.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        
        # Tạo ma trận User-Item
        train_matrix = raw_data.pivot(index='user_id', columns='movie_id', values='rating')
        
        # Tính độ tương quan Pearson giữa các Item
        item_similarity = train_matrix.corr(method='pearson').fillna(0)
        
        # Tìm K tối ưu (Sử dụng danh sách K của bạn)
        best_k = 80
        
        # Tính toán ma trận dự đoán cuối cùng
        final_predictions = predict_item_knn(train_matrix, item_similarity, k=best_k)
        
        # Chuyển về DataFrame để dễ truy vấn theo UserID
        predictions_df = pd.DataFrame(final_predictions, index=train_matrix.index, columns=train_matrix.columns)
        
        final_matrix_export = predictions_df.T
        final_matrix_export.index = [f"Item {int(i)}" for i in final_matrix_export.index]
        final_matrix_export.columns = [f"User {int(u)}" for u in final_matrix_export.columns]
        # Ghi đè trực tiếp vào file kết quả
        final_matrix_export.to_csv('../../crawl_data/data/item_user_optimized_results.csv')
        print(f"    Đã cập nhật xong ma trận dự đoán. Best K dùng: {best_k}")
        last_update = datetime.datetime.now().strftime("%H:%M:%S")
        return predictions_df
    except Exception as e:
        print(f"    Lỗi trong quá trình cập nhật: {e}")

# --- 3. VÒNG LẶP CHẠY NGẦM ---
async def update_periodically():
    global predictions_df, last_update
    loop = asyncio.get_event_loop()
    while True:
        try:
            new_predictions = await loop.run_in_executor(executor, update_recommender_system)
            if new_predictions is not None:
                predictions_df = new_predictions
                last_update = datetime.datetime.now().strftime("%H:%M:%S")
                # 2. In ra lúc hoàn thành thành công
                print(f"[{last_update}] Cập nhật ma trận thành công.")    
        except Exception as e:
            print(f"Lỗi vòng lặp: {e}")
        await asyncio.sleep(30)

@app.get("/get-update-status")
def get_update_status():
    return {"last_update": last_update}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_periodically())

@app.get("/init-data")
def get_init_data():
    return {
        "movies": movies_df.fillna("").to_dict(orient="records") if movies_df is not None else [],
        "users": u_info.to_dict(orient="records") if u_info is not None else []
    }

@app.get("/recommend-for-user/{user_id}")
def get_personal_recommendations(user_id: int):
    if predictions_df is None or user_id not in predictions_df.index:
        return []
    
    # Lấy các điểm dự đoán của User này
    user_preds = predictions_df.loc[user_id]
    
    # Lọc bỏ các phim mà người dùng ĐÃ đánh giá rồi 
    # Lấy dữ liệu rating hiện tại từ file hoặc biến toàn cục
    u_data_current = pd.read_csv(DATA_DIR + 'u.data', sep='\t', names=['user_id', 'movie_id', 'rating'])
    user_rated_ids = u_data_current[u_data_current['user_id'] == user_id]['movie_id'].tolist()
    
    # Bỏ qua các phim đã rate
    user_preds = user_preds.drop(labels=user_rated_ids, errors='ignore')

    # 3. Sắp xếp lấy Top 15 movie_id có điểm dự đoán cao nhất
    top_movie_ids = user_preds.sort_values(ascending=False).head(15).index.tolist()
    
    recommend_indices = [int(mid) - 1 for mid in top_movie_ids]
    
    return recommend_indices

@app.get("/user-status/{username}")
def get_user_status(username: str):
    if u_info is None: return {"exists": False}
    row = u_info[u_info['username'] == username]
    if row.empty: return {"exists": False}
    
    uid = int(row.iloc[0]['user_id'])
    # Kiểm tra xem đã có trong ma trận dự đoán chưa
    is_new = uid not in predictions_df.index if predictions_df is not None else True
    return {"exists": True, "user_id": uid, "is_new": is_new}

# ===============================================================

# Khai báo để tạo class Rating (Gồm id ng dùng, id phim, số rate)
class RatingData(BaseModel):
    user_id: int
    movie_id: int
    rating: int

# Khai báo cấu trúc dữ liệu gửi lên cho Cold Start
class ColdStartData(BaseModel):
    user_id: int
    selected_movie_ids: List[int] # Nhận vào một danh sách các ID phim

class GenreList(BaseModel):
    genres: List[str]
    
@app.get("/all-users")
def get_all_users(): # Lấy ra tất cả người dùng
    u_info = pd.read_csv('crawl_data/data/u.info', sep='\t', names=['user_id', 'username', 'password'])
    safe_data = u_info[['user_id', 'username']] 
    return safe_data.to_dict(orient='records')

@app.get("/movies") # Trả về các phim
def get_movies():
    df_clean = df.where(pd.notna(df), None)
    
    return df_clean.to_dict(orient='records')

@app.get("/genres") # Trả về các thể loại cho phần Cold Start
def get_genres():
    genre_path = "../../crawl_data/data/u.genre"
    try:
        genres_df = pd.read_csv(genre_path, sep='|', names=['genre', 'id'], encoding='utf-8-sig')
        list_genres = genres_df['genre'].tolist()
        if 'unknown' in list_genres: 
            list_genres.remove('unknown')
        return list_genres
    except Exception as e:
        # Nếu không tìm thấy file, trả về list rỗng tránh sập server
        print("Lỗi đọc file genre:", e)
        return []

@app.get("/search") # Tìm kiếm phim theo tên
def search_movies(query: str):
    try:
        res = df[df['title'].str.contains(query, case=False, na=False)].index.tolist()
        return {"result_indices": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/recommend/{idx}") # Hàm trả về các phim đc đề xuất
def get_recommendations(idx: int):
    try:
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse=True)[1:11]
        indices = [x[0] for x in sim_scores]
        return {"recommend_indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-ratings/{user_id}") # Lấy ra các phim mà user đã đánh giá
def get_user_ratings(user_id: int):
    data_path = "../../crawl_data/data/u.data"
    try:
        u_data = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating'], encoding='utf-8-sig')
        
        # Lọc chỉ lấy những dòng của user_id được yêu cầu
        user_history = u_data[u_data['user_id'] == user_id]
        
        # Trả về dạng JSON
        return user_history.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies/trending")
def get_trending_movies():
    try:
        # Lấy vị trí của 15 phim đầu tiên 
        trending_indices = list(range(min(15, len(df))))
        return {"trending_indices": trending_indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cf-recommend/{idx}")
def get_cf_recommendations(idx: int):
    try:
        # Dùng trực tiếp biến predictions_df (dữ liệu live)
        if predictions_df is None or idx not in predictions_df.index:
            return {"recommend_indices": []}
        
        # Lấy điểm dự đoán của User {idx} từ RAM
        user_preds = predictions_df.loc[idx]
        
        # Lấy danh sách phim user này đã rate để lọc bỏ (để thấy gợi ý mới)
        u_data_current = pd.read_csv(DATA_DIR + 'u.data', sep='\t', names=['user_id', 'movie_id', 'rating'])
        rated_ids = u_data_current[u_data_current['user_id'] == idx]['movie_id'].tolist()
        user_preds = user_preds.drop(labels=rated_ids, errors='ignore')
        
        # Sắp xếp lấy Top 10 movie_id cao nhất và đổi sang index (mid - 1)
        top_movie_ids = user_preds.sort_values(ascending=False).head(10).index.tolist()
        top_indices = [int(mid) - 1 for mid in top_movie_ids]
        
        return {"recommend_indices": top_indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
@app.post("/cold-start") # Hàm Cold Start cho người dùng mới
def cold_start(data: ColdStartData):
    data_path = '../../crawl_data/data/u.data'
    try:
        # Đọc file hiện tại để kiểm tra trùng lặp
        if os.path.exists(data_path):
            existing = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating'], encoding='utf-8-sig')
        else:
            existing = pd.DataFrame(columns=['user_id', 'movie_id', 'rating'])
        
        # Lọc bỏ những phim mà user này đã có trong u.data (tránh duplicate)
        already_rated = existing[existing['user_id'] == data.user_id]['movie_id'].tolist()
        new_movie_ids = [mid for mid in data.selected_movie_ids if mid not in already_rated]
        
        if not new_movie_ids:
            return {"status": "success", "message": "Tất cả phim đã được đánh giá trước đó"}
        
        # Ghi các phim mới vào file (4 sao = yêu thích ban đầu, hợp lý hơn 5 sao tùy tiện)
        new_rows = [[data.user_id, mid, 4] for mid in new_movie_ids]
        with open(data_path, 'a', encoding='utf-8-sig') as f:
            for row in new_rows:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
        
        return {
            "status": "success", 
            "message": f"Đã lưu {len(new_rows)} phim yêu thích ban đầu",
            "saved_count": len(new_rows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/movies/by-genres")
def get_movies_by_genres(data: GenreList):
    try:
        mask = df[data.genres].any(axis=1)
        res_indices = df[mask].head(20).index.tolist()
        return {"result_indices": res_indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/update-rating") # Hàm update phần rating trong u.data
def update_rating(data: RatingData):
    data_path = '../../crawl_data/data/u.data'
    try:
        # Đọc file u.data
        temp_u_data = pd.read_csv(data_path,sep='\t',names=['user_id', 'movie_id', 'rating'],encoding='utf-8-sig')
        
        # Kiểm tra đã đánh giá phim chưa
        check = (temp_u_data['user_id'] == data.user_id) & (temp_u_data['movie_id'] == data.movie_id)
        if(temp_u_data[check].empty):
            new_row = pd.DataFrame([[data.user_id, data.movie_id, data.rating]], columns=['user_id', 'movie_id', 'rating'])
            temp_u_data = pd.concat([temp_u_data, new_row], ignore_index=True)
        else:
            temp_u_data.loc[check, 'rating'] = data.rating
        
        temp_u_data.to_csv(data_path, sep='\t', index=False, header=False, encoding='utf-8-sig')
        return {"status": "success", "message": "Đã cập nhật đánh giá"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movie-reviews/{title}")
def get_movie_reviews(title: str):
    if reviews_df.empty:
        return {"reviews": [], "wordcloud_base64": None, "pos_ratio": 0, "neg_ratio": 0, "total_reviews": 0}
    
    # Lọc comments theo title
    movie_reviews = reviews_df[reviews_df['Movie_Title'].str.lower() == title.lower()]
    
    # Lọc bỏ các bình luận "không_bình_luận" hoặc trống
    movie_reviews = movie_reviews[
        ~movie_reviews['clean_comment'].astype(str).str.contains('không_bình_luận', case=False, na=False) &
        (movie_reviews['clean_comment'].astype(str).str.strip() != '') &
        ~movie_reviews['original_comment'].astype(str).str.contains('không_bình_luận', case=False, na=False)
    ]

    if movie_reviews.empty:
        return {"reviews": [], "wordcloud_base64": None, "pos_ratio": 0, "neg_ratio": 0, "total_reviews": 0}
    
    # Tính tỉ lệ khen chê (>= 8 là khen, <= 5 là chê)
    total_reviews = len(movie_reviews)
    pos_reviews = len(movie_reviews[movie_reviews['rating'] >= 8])
    neg_reviews = len(movie_reviews[movie_reviews['rating'] <= 5])
    
    pos_ratio = int((pos_reviews / total_reviews) * 100) if total_reviews > 0 else 0
    neg_ratio = int((neg_reviews / total_reviews) * 100) if total_reviews > 0 else 0
    
    # Lấy 5 comments nổi bật nhất (rating cao nhất)
    top_reviews = movie_reviews.sort_values(by='rating', ascending=False).head(5)
    
    # Đổi sang sử dụng 'original_comment' thay vì 'clean_comment'
    reviews_list = []
    for _, row in top_reviews.iterrows():
        comment_text = row['original_comment'] if pd.notna(row['original_comment']) else row['clean_comment']
        reviews_list.append({
            'rating': row['rating'],
            'clean_comment': comment_text # Frontend vẫn dùng key 'clean_comment' nhưng dữ liệu là comment gốc
        })
    
    # Nối tất cả text (dùng clean_comment cho wordcloud để tối ưu)
    all_text = " ".join(movie_reviews['clean_comment'].dropna().astype(str).tolist())
    
    # Tạo wordcloud
    wordcloud_base64 = None
    if all_text.strip():
        try:
            # Dùng colormap đẹp mắt
            wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(all_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            wordcloud_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print("Error generating wordcloud:", e)
    
    return {
        "reviews": reviews_list, 
        "wordcloud_base64": wordcloud_base64,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "total_reviews": total_reviews
    }
