# Hệ thống Gợi ý Phim

## Giới thiệu

Dự án này là một ứng dụng web cung cấp các tính năng gợi ý phim cá nhân hóa cho người dùng. Hệ thống kết hợp giữa phương pháp gợi ý dựa trên nội dung (Content-Based) và lọc cộng tác (Collaborative Filtering - Item-Item KNN) để đưa ra kết quả đề xuất chính xác nhất.

## Công nghệ sử dụng

- Backend: FastAPI, Scikit-learn, Pandas, SQLite.
- Frontend: Streamlit.
- Thuật toán: Item-Item KNN, Cosine Similarity, Pearson Correlation.
- Bảo mật: Bcrypt để mã hóa mật khẩu người dùng.

## Cấu trúc dự án

- main.py: Máy chủ API xử lý dữ liệu, xác thực và chạy vòng lặp huấn luyện lại mô hình mỗi 30 giây.
- login.py: Giao diện đăng nhập và đăng ký (Điểm bắt đầu của ứng dụng Frontend).
- pages/app.py: Giao diện chính của ứng dụng sau khi đăng nhập, bao gồm danh sách phim và chi tiết gợi ý.
- sql.py: Script hỗ trợ kiểm tra dữ liệu tài khoản trong cơ sở dữ liệu SQLite.

## Cài đặt

1. Cài đặt các thư viện cần thiết từ file requirements.txt:
   pip install -r requirements.txt

2. Đảm bảo cấu trúc thư mục dữ liệu đầu vào khớp với đường dẫn khai báo trong mã nguồn (mặc định là ../../crawl_data/data/).

## Hướng dẫn sử dụng

Để vận hành hệ thống, bạn cần khởi chạy cả Backend và Frontend theo trình tự sau:

### Bước 1: Khởi chạy Backend

Mở terminal tại thư mục ./backend và chạy lệnh:
fastapi dev main.py

Backend sẽ nạp dữ liệu phim và bắt đầu vòng lặp tính toán ma trận gợi ý ngầm.

### Bước 2: Khởi chạy Frontend

Mở một terminal mới và chạy lệnh:
streamlit run login.py

Ứng dụng sẽ mở trên trình duyệt, yêu cầu bạn đăng nhập hoặc đăng ký tài khoản mới để bắt đầu sử dụng.

## Các tính năng chính

- Đăng ký/Đăng nhập: Quản lý người dùng và lịch sử đánh giá phim.
- Cold Start: Thu thập sở thích của người dùng mới thông qua việc chọn thể loại và phim ấn tượng ban đầu.
- Gợi ý tương tự: Đề xuất các phim dựa trên đặc trưng nội dung (Thể loại, quốc gia...).
- Gợi ý cá nhân hóa: Sử dụng ma trận dự đoán KNN để tìm phim phù hợp với gu riêng của từng người dùng.
- Cập nhật thời gian thực: Hệ thống tự động cập nhật ma trận gợi ý mỗi 30 giây để ghi nhận các hành vi mới nhất của người dùng.

## Nhóm tác giả

- Nguyễn Kim An
- Nguyễn Tiến Đạt
- Trần Đức Lâm
