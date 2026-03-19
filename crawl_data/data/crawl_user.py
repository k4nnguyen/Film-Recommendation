import pandas as pd
import numpy as np

movies_df = pd.read_csv('movies_metadata.csv')
num_movies = len(movies_df)
print(f"Số lượng phim tìm thấy trong metadata: {num_movies}")

item_ids = list(range(1, num_movies + 1))


num_users = 500        # Số lượng người dùng ảo 
min_ratings = 5        # Số phim TỐI THIỂU mỗi người dùng đánh giá
max_ratings = 15       # Số phim TỐI ĐA mỗi người dùng đánh giá

np.random.seed(42) # Đặt seed để nếu chạy lại code nhiều lần thì ra cùng 1 kết quả
data = []

for user_id in range(1, num_users + 1):
    # Chọn ngẫu nhiên số lượng phim user này sẽ đánh giá (từ min đến max)
    num_rated = np.random.randint(min_ratings, max_ratings + 1)
    
    # Chọn ngẫu nhiên ID các bộ phim mà user này đã xem (không trùng lặp)
    rated_movies = np.random.choice(item_ids, size=num_rated, replace=False)
    
    for item_id in rated_movies:
        # Random số sao từ 1 đến 5
        rating = np.random.randint(1, 6)
        
        data.append([user_id, item_id, rating])

generated_u_data = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])

# Ghi ra file .data, sử dụng tab (sep='\t') làm dấu phân cách, bỏ đi header và index
output_filename = 'u.data'
generated_u_data.to_csv(output_filename, sep='\t', index=False, header=False)

print(f"Đã tạo thành công file {output_filename} với {len(generated_u_data)} dòng đánh giá.")