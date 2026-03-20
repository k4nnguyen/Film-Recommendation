import sqlite3
import bcrypt
import os

def migrate_data():
    # Đường dẫn tới file u.info và users.db
    # Lấy tọa độ gốc tuyệt đối để không bao giờ sai đường dẫn
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.join(BASE_DIR, 'data/u.info')
    db_path = os.path.join(BASE_DIR, '../Web/backend/users.db')

    if not os.path.exists(info_path):
        print(f"Không tìm thấy file {info_path}")
        return

    # Kết nối DB và tạo bảng nếu chưa có
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')

    # Đọc file u.info
    with open(info_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    total_users = len(lines)
    print(f"Bắt đầu đồng bộ {total_users} users vào Database...\n")

    success_count = 0
    skip_count = 0

    for i, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) >= 3:
            username = parts[1]
            password = parts[2]

            # Mã hóa mật khẩu
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            try:
                # Lưu vào SQLite
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
                success_count += 1
            except sqlite3.IntegrityError:
                # Nếu user đã tồn tại thì bỏ qua
                skip_count += 1
        
        # In tiến độ
        if (i + 1) % 50 == 0 or (i + 1) == total_users:
            print(f"Đang xử lý: {i + 1}/{total_users}...")

    # Lưu thay đổi và đóng kết nối
    conn.commit()
    conn.close()

    print("\nHOÀN TẤT ĐỒNG BỘ!")
    print(f"Thêm thành công: {success_count} users")
    print(f"Bỏ qua (đã tồn tại): {skip_count} users")

if __name__ == "__main__":
    migrate_data()