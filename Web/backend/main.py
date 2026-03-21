from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import os 
from typing import List
import sqlite3
import bcrypt

app = FastAPI()
df = pd.read_csv('../../crawl_data/data/movies_with_posters.csv')
features = df.select_dtypes(include=['number'])
sim_matrix = cosine_similarity(features)

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
        # Tự động tìm đường dẫn file csv
        cf_path = '../../crawl_data/data/item_user_optimized_results.csv'
        if not os.path.exists(cf_path):
            cf_path = 'item_user_optimized_results.csv'
            
        cf_matrix = pd.read_csv(cf_path, index_col=0)
        
        # (Lấy cột User tương ứng với số thứ tự của phim)
        col_name = f"User {idx}"
        
        if col_name in cf_matrix.columns:
            # Cách ban đầu: sắp xếp và bóc tách chuỗi "Item X"
            top_indices_str = cf_matrix[col_name].sort_values(ascending=False).head(10).index.tolist()
            
            top_indices = []
            for idx_str in top_indices_str:
                try:
                    item_num = int(idx_str.split(' ')[1])
                    top_indices.append(item_num - 1)
                except:
                    continue
            
            return {"recommend_indices": top_indices}
        else:
            return {"recommend_indices": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
@app.post("/cold-start") # Hàm Cold Start cho người dùng mới
def cold_start(data: ColdStartData):
    data_path = '../../crawl_data/data/u.data'
    try:
        # Tạo danh sách các dòng mới (user_id, movie_id, 5 sao)
        new_entries = [[data.user_id, mid, 5] for mid in data.selected_movie_ids]
        pd.DataFrame(new_entries).to_csv(data_path, mode='a', sep='\t', index=False, header=False, encoding='utf-8-sig')
        return {"status": "success", "message": "Đã lưu thiết lập ban đầu"}
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