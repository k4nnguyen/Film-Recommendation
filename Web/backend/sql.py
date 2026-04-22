import sqlite3

# Kết nối database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Chạy lệnh lấy dữ liệu
c.execute('SELECT * FROM users')

# Gom tất cả các dòng kết quả lại
rows = c.fetchall()

# In ra từng dòng để xem
print("Danh sách tài khoản trong Database:")
print("-" * 40)
for row in rows:
    print(f"Username: {row[0]} | Password (đã mã hóa): {row[1]}")

# Nhớ đóng kết nối cho sạch sẽ
conn.close()

