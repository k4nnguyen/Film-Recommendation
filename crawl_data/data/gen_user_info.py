import pandas as pd

num_users = 500

data = []

# Tạo dữ liệu cho từng người dùng
for i in range(1, num_users + 1):
    user_id = i
    username = f"user{i}"
    password = f"user{i}"

    data.append([user_id, username, password])

generated_user_info = pd.DataFrame(data, columns=['user_id', 'username', 'password'])

# Lưu ra file u.info, dùng dấu Tab (sep='\t') để phân cách các cột
output_filename = 'u.info'
generated_user_info.to_csv(output_filename, sep='\t', index=False, header=False)

print(f"Đã tạo thành công file {output_filename} với {len(generated_user_info)} tài khoản.")
print("\nXem thử 5 dòng đầu tiên:")
print(generated_user_info.head())