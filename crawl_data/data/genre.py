import pandas as pd

movies_df = pd.read_csv('movies_metadata_encoded.csv')

# 2. Lấy danh sách các thể loại
# Các cột từ vị trí số 5 (index 5) trở đi là các thể loại phim
genre_columns = movies_df.columns[5:].tolist()

genre_data = []
genre_data.append(['unknown', 0])
for i, genre_name in enumerate(genre_columns):
    # i sẽ chạy từ 1, 2... tương ứng với ID
    genre_data.append([genre_name, i+1])

genre_df = pd.DataFrame(genre_data, columns=['genre_name', 'genre_id'])

# 5. Lưu ra file với định dạng ngăn cách bởi dấu gạch đứng "|" 
output_filename = 'u.genre'
genre_df.to_csv(output_filename, sep='|', index=False, header=False)

print(f"Đã tạo thành công file {output_filename} chứa {len(genre_columns)} thể loại.")
print("\nXem thử 10 dòng đầu tiên:")
print(genre_df.head(10))