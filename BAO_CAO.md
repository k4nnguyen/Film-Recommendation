# BÁO CÁO BÀI TẬP LỚN
## Hệ Thống Gợi Ý Phim Ứng Dụng Học Máy

---

| Thông tin | Chi tiết |
|-----------|---------|
| **Trường** | Học viện Công nghệ Bưu chính Viễn thông (PTIT) |
| **Nhóm** | Nguyễn Kim An · Nguyễn Tiến Đạt · Trần Đức Lâm |
| **Năm học** | 2025 – 2026 |
| **Công nghệ** | Python · FastAPI · Streamlit · Scikit-learn · Selenium |

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Thu thập dữ liệu](#3-thu-thập-dữ-liệu)
4. [Tiền xử lý dữ liệu](#4-tiền-xử-lý-dữ-liệu)
5. [Phân tích khám phá dữ liệu (EDA)](#5-phân-tích-khám-phá-dữ-liệu-eda)
6. [Mô hình học máy](#6-mô-hình-học-máy)
7. [Giao diện Web](#7-giao-diện-web)
8. [Kết quả & Đánh giá](#8-kết-quả--đánh-giá)
9. [Hướng phát triển](#9-hướng-phát-triển)

---

## 1. Giới thiệu

### 1.1 Đặt vấn đề

Trong thời đại số, người dùng đứng trước hàng nghìn lựa chọn phim mỗi ngày. Các nền tảng như Netflix, YouTube đã chứng minh rằng hệ thống gợi ý phim cá nhân hóa là yếu tố then chốt giúp tăng trải nghiệm người dùng và thời gian sử dụng dịch vụ.

Bài tập lớn này xây dựng một **hệ thống gợi ý phim hoàn chỉnh** — từ việc tự động thu thập dữ liệu từ mạng xã hội phim Momo, làm sạch văn bản tiếng Việt, đến huấn luyện mô hình học máy và triển khai thành ứng dụng web thực tế.

### 1.2 Mục tiêu

- Thu thập dữ liệu phim và bình luận từ nền tảng Momo bằng kỹ thuật Web Scraping.
- Xử lý ngôn ngữ tự nhiên tiếng Việt (Vietnamese NLP) để làm sạch văn bản bình luận.
- Xây dựng hệ thống gợi ý phim kết hợp hai phương pháp: **Content-Based** và **Collaborative Filtering**.
- Triển khai ứng dụng web đa tầng với backend API và frontend Streamlit.

### 1.3 Phạm vi

| Hạng mục | Số lượng |
|----------|---------|
| Phim thu thập | ~83 bộ phim |
| Bình luận thu thập | ~10,000+ bình luận |
| Người dùng mô phỏng | 500 users |
| Tương tác đánh giá | ~5,000 lượt |

---

## 2. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────┐
│                    TẦNG DỮ LIỆU                         │
│  Momo.vn → Selenium Scraper → CSV Files                 │
│       ├── movies_metadata_encoded.csv                   │
│       ├── movie_reviews.csv                             │
│       └── movie_reviews_cleaned.csv                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    TẦNG MÔ HÌNH AI                      │
│  app.py: Hybrid KNN (Content-Based + Collaborative)     │
│  ├── Rating Similarity (Cosine + Shrinkage)             │
│  ├── Text Similarity (TF-IDF)                           │
│  └── Output: item_user_optimized_results.csv            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    TẦNG BACKEND (FastAPI)                │
│  main.py : Port 8000                                    │
│  ├── /movies, /genres, /search                          │
│  ├── /recommend/{idx}     (Content-Based)               │
│  ├── /recommend-for-user  (Collaborative Filtering)     │
│  ├── /movie-reviews       (NLP + WordCloud)             │
│  └── /login, /register, /update-rating, /cold-start    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    TẦNG FRONTEND (Streamlit)             │
│  login.py → pages/app.py                                │
│  ├── Trang chủ: Phim đề xuất cá nhân hóa               │
│  ├── Danh sách: Toàn bộ phim                            │
│  ├── Chi tiết: Phân tích bình luận + WordCloud          │
│  └── Cold Start: Khởi tạo sở thích người dùng mới      │
└─────────────────────────────────────────────────────────┘
```

### 2.1 Cấu trúc thư mục dự án

```
Film_Recommendation/
├── crawl_data/
│   ├── data_pipeline.py        # Menu CLI điều phối toàn bộ pipeline
│   ├── prepare_ml_data.py      # Tạo dữ liệu huấn luyện (mock data)
│   ├── preprocess_code.py      # Hàm NLP xử lý văn bản
│   ├── eda_plots/              # Biểu đồ EDA
│   └── data/
│       ├── movies_metadata_encoded.csv   # Metadata phim (one-hot genre)
│       ├── movie_reviews.csv             # Bình luận thô
│       ├── movie_reviews_cleaned.csv     # Bình luận đã xử lý NLP
│       ├── app.py                        # Huấn luyện mô hình Hybrid
│       ├── u.data / u.info / u.genre     # Dữ liệu chuẩn MovieLens
│       └── ua_train.csv / ua_test.csv    # Tập train/test
├── Web/
│   ├── backend/main.py         # FastAPI server
│   └── frontend/
│       ├── login.py            # Trang đăng nhập/đăng ký
│       └── pages/app.py        # Giao diện chính
└── requirements.txt
```

---

## 3. Thu thập dữ liệu

### 3.1 Công nghệ sử dụng

- **Selenium + ChromeDriver**: Tự động hóa trình duyệt, giả lập hành vi cuộn trang.
- **BeautifulSoup**: Phân tích và trích xuất nội dung HTML.

### 3.2 Thu thập siêu dữ liệu phim

Từ trang danh sách phim của Momo, pipeline thu thập các trường:

| Trường | Mô tả |
|--------|-------|
| `title` | Tên phim |
| `genre` | Thể loại (văn bản gốc) |
| `country` | Quốc gia sản xuất |
| `release_date` | Ngày công chiếu |
| `url` | Liên kết trang phim trên Momo |
| `poster_url` | Đường dẫn ảnh poster |
| `genre_*` | One-hot encoding theo từng thể loại |

**Điểm kỹ thuật nổi bật — Dynamic One-Hot Encoding:**

Thay vì xử lý thể loại hậu kỳ, pipeline tự động phát hiện thể loại mới xuất hiện trong quá trình crawl và mở rộng cột CSV theo thời gian thực. Đảm bảo tính nhất quán dữ liệu dù thứ tự crawl thay đổi.

### 3.3 Thu thập bình luận

Pipeline liên kết với file metadata, tự động lấy URL từng phim và crawl bình luận. Cơ chế **auto-scroll** giả lập việc người dùng kéo xuống để tải thêm bình luận ẩn dưới nút "Xem thêm".

> **Tính idempotent**: Hệ thống so sánh danh sách phim đã có với file CSV, chỉ thu thập bình luận của những phim còn thiếu — tránh trùng lặp, tiết kiệm thời gian.

---

## 4. Tiền xử lý dữ liệu

### 4.1 Quy trình xử lý văn bản tiếng Việt (12 bước)

```
Bình luận thô
  │
  ▼  1. Chuyển chữ thường
  ▼  2. Xóa URL, link mạng xã hội
  ▼  3. Chuyển Emoji → văn bản (thư viện emoji)
  ▼  4. Dịch Teencode  ("ko"→"không", "mk"→"mình")
  ▼  5. Chuẩn hóa lặp âm ("hayyy"→"hay", "koooo"→"không")
  ▼  6. Xóa ký tự đặc biệt, số thừa
  ▼  7. Tách từ tiếng Việt (PyVi ViTokenizer)
  ▼  8. Loại Stopwords (vietnamese-stopwords từ GitHub)
  ▼  9. Bảo vệ từ khóa cảm xúc quan trọng
  ▼ 10. Lọc bình luận quá ngắn (< 3 từ)
  ▼ 11. Đánh dấu bình luận rỗng ("không_bình_luận")
  ▼ 12. Lưu vào cột clean_comment
  │
  ▼
Văn bản sạch
```

### 4.2 Bảo vệ từ khóa cảm xúc

Khi lọc stopwords, hệ thống **giữ lại có chủ đích** các từ mang giá trị phân tích cảm xúc:

```python
important_sentiment_words = {
    "hay", "quá", "tốt", "đỉnh", "tuyệt_vời", "xuất_sắc",
    "dở", "tệ", "chán", "buồn", "vui", "thích", "ghét",
    "không", "chưa", "cảm_động", "hấp_dẫn", ...
}
```

### 4.3 Tách từ tiếng Việt

Dùng **PyVi ViTokenizer** ghép các từ ghép lại đúng ngữ nghĩa:

> `"bộ phim hay"` → `"bộ_phim hay"`

Điều này giúp thuật toán TF-IDF hiểu đúng nghĩa của cụm từ thay vì tách rời từng chữ.

---

## 5. Phân tích khám phá dữ liệu (EDA)

### 5.1 Phân phối độ dài bình luận

So sánh trước/sau xử lý cho thấy: độ dài trung bình giảm ~40% sau khi loại nhiễu, phân phối tập trung hơn.

### 5.2 Top 10 phim được bình luận nhiều nhất

Biểu đồ cột thể hiện các phim thu hút lượng tương tác cao nhất (sau khi lọc bỏ bình luận rỗng).

### 5.3 WordCloud

| Trước xử lý | Sau xử lý |
|------------|----------|
| Nhiều emoji, ký tự lạ, teencode | Chỉ còn từ khóa đánh giá chất lượng |
| Khó khai thác ý nghĩa | Rõ ràng cảm xúc tích cực/tiêu cực |

---

## 6. Mô hình học máy

### 6.1 Chuẩn bị dữ liệu huấn luyện

Nhóm sinh **Mock Data** theo chuẩn MovieLens 100k (`prepare_ml_data.py`):

- **500 người dùng** ảo (`u.info`)
- Mỗi user đánh giá ngẫu nhiên **5–15 phim**, điểm 1–5 sao (`u.data`)
- Chia tập **Train/Test = 70%/30%**

### 6.2 Hybrid Item-Item KNN

Kết hợp hai nguồn thông tin để tính ma trận độ tương đồng giữa các phim:

```
Hybrid_Similarity = α × Rating_Similarity + (1-α) × Text_Similarity
```

**Rating Similarity (Collaborative Filtering)**
- Cosine Similarity trên ma trận User-Item
- **Shrinkage Penalty**: Phạt cặp phim ít người cùng đánh giá, tránh độ tương đồng ảo

**Text Similarity (Content-Based)**
- Gom toàn bộ bình luận của mỗi phim thành một tài liệu
- Biểu diễn bằng vector **TF-IDF** (5,000 features, bigrams)
- Cosine Similarity giữa các vector phim

**Z-score Normalization:**

Chuẩn hóa điểm đánh giá theo từng user trước khi dự đoán:

```
z(u,i) = (r(u,i) - mean_u) / std_u
```

Giúp "san bằng" sự chênh lệch giữa user khắt khe và user dễ tính.

### 6.3 Grid Search tìm tham số tối ưu

| Tham số | Giá trị thử nghiệm |
|---------|-------------------|
| **α (Alpha)** | 0.0, 0.3, 0.5, 0.7, 0.8, 1.0 |
| **K (số láng giềng)** | 5, 8, 10, 12, 15, 20 |

Với mỗi cặp (α, K): tính **RMSE** và **MAE** trên tập Test, chọn cấu hình RMSE thấp nhất.

**Công thức dự đoán:**

```
pred(u,j) = mean_u + z_pred × std_u

Trong đó: z_pred = Σ[w(i,j) × z(u,i)] / (Σ|w(i,j)| + damping)
```

### 6.4 Cập nhật mô hình thời gian thực

Backend chạy **asyncio loop**, tự động cập nhật ma trận dự đoán mỗi **30 giây**. Khi người dùng thực tế đánh giá phim, kết quả gợi ý cải thiện dần mà không cần restart server.

---

## 7. Giao diện Web

### 7.1 Backend — FastAPI

| Endpoint | Phương thức | Chức năng |
|----------|------------|-----------|
| `/movies` | GET | Danh sách toàn bộ phim |
| `/genres` | GET | Danh sách thể loại |
| `/search?query=...` | GET | Tìm kiếm phim theo tên |
| `/recommend/{idx}` | GET | Gợi ý tương tự (Content-Based) |
| `/recommend-for-user/{id}` | GET | Gợi ý cá nhân hóa (CF) |
| `/movie-reviews/{title}` | GET | Bình luận + WordCloud + tỉ lệ khen/chê |
| `/register` | POST | Đăng ký tài khoản |
| `/login` | POST | Đăng nhập |
| `/update-rating` | POST | Cập nhật đánh giá phim |
| `/cold-start` | POST | Khởi tạo sở thích user mới |

> Mật khẩu được mã hóa bằng **bcrypt** và lưu vào **SQLite**.

### 7.2 Frontend — Streamlit

**Màn hình Cold Start** (user mới):
1. Chọn tối đa **3 thể loại** yêu thích.
2. Hệ thống hiển thị phim theo thể loại.
3. User chọn **đúng 3 bộ phim** → gửi backend làm điểm khởi đầu CF.

**Trang chi tiết phim:**
- Thông tin phim, link tới Momo
- Hệ thống **đánh giá sao** (1–5) tương tác trực tiếp
- Phân tích bình luận: tổng lượt, tỉ lệ khen/chê, WordCloud
- Mục **"Có thể bạn cũng thích"** (CF-based)
- Mục **"Gợi ý phim tương tự"** (Content-Based)

---

## 8. Kết quả & Đánh giá

| Tiêu chí | Đánh giá |
|----------|---------|
| Thu thập dữ liệu tự động | ✅ Idempotent, xử lý lazy-loading |
| NLP tiếng Việt | ✅ 12 bước, teencode + emoji + lặp âm |
| Giao diện người dùng | ✅ Cold Start + cập nhật mô hình tự động |
| Độ chính xác gợi ý | ⚠️ Phụ thuộc vào số user thực đánh giá |
| Hiệu năng pipeline | ⚠️ CSV chưa tối ưu cho dữ liệu lớn |

---

## 9. Hướng phát triển

1. **Cơ sở dữ liệu**: Chuyển từ CSV sang **PostgreSQL** — tăng hiệu năng và đảm bảo ACID.
2. **Crawl song song**: Dùng **multithreading** để giảm thời gian thu thập từ hàng giờ xuống vài phút.
3. **Mô hình nâng cao**: Thử nghiệm **SVD (Matrix Factorization)** hoặc **Neural CF**.
4. **Phân tích cảm xúc**: Xây dựng classifier Positive/Negative trên bình luận thay ngưỡng điểm cứng.
5. **Triển khai**: Đóng gói bằng **Docker**, deploy lên cloud (Railway, Render, VPS).

---

## Phụ lục — Hướng dẫn chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt
pip install selenium pyvi emoji wordcloud

# 2. Chạy Data Pipeline
cd crawl_data
python data_pipeline.py
# Chọn 1: Thu thập phim | 2: Thu thập bình luận | 3: Tiền xử lý | 4: EDA

# 3. Tạo dữ liệu huấn luyện
python prepare_ml_data.py

# 4. Huấn luyện mô hình Hybrid
cd crawl_data/data
python -X utf8 app.py

# 5. Khởi động Backend
cd Web/backend
uvicorn main:app --reload --port 8000

# 6. Khởi động Frontend
cd Web/frontend
streamlit run login.py
```

---

*Báo cáo được tạo ngày 23/04/2026 · PTIT © 2026*
