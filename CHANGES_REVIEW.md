# Báo Cáo Thay Đổi: Nâng Cấp Hệ Thống Gợi Ý Phim

> **Phiên làm việc:** 18/05/2026  
> **Mục tiêu:** Cải thiện độ chính xác gợi ý bằng trích chọn đặc trưng tự động (bình luận) và Deep Learning (NCF)

---

## Tổng Quan Thay Đổi

Hệ thống được nâng cấp theo **2 hướng chính** và kết hợp thành **1 pipeline hoàn chỉnh**:

| Hướng | Mô tả | Áp dụng cho |
|---|---|---|
| **Trích chọn đặc trưng tự động** | TF-IDF + Autoencoder Neural Network | Bình luận phim |
| **Deep Learning cho gợi ý** | NCF v2 (NeuMF + Genre Side Features) | Rating người dùng |

---

## Phần 1 — Trích Chọn Đặc Trưng Tự Động Cho Bình Luận

### Vấn đề của TF-IDF cũ

```python
# Cũ: 5000 chiều, thủ công, không học
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(corpus)
text_sim_matrix = cosine_similarity(tfidf_matrix)
```

TF-IDF chỉ đếm tần suất từ — **không học được pattern ẩn**, nhiều chiều nhiễu.

### Giải pháp: TF-IDF + Autoencoder

```
Binh luan (text tho)
    -> TF-IDF: 5000 chieu (nhieu nhieu)
    -> Encoder: 5000 -> 512 -> 128 -> 64  (TU DONG hoc dac trung)
    -> Decoder: 64 -> 128 -> 512 -> 5000  (kiem tra chat luong)
    -> 64 chieu dac trung tinh gon
    -> Cosine Similarity -> text_sim_matrix (98x98)
```

### Kiến trúc Autoencoder

```python
Encoder: Linear(5000->512)+ReLU+Drop -> Linear(512->128)+ReLU+Drop -> Linear(128->64)+ReLU
Decoder: Linear(64->128)+ReLU -> Linear(128->512)+ReLU -> Linear(512->5000)+Sigmoid
Loss: MSELoss | Adam | Epochs: 30 | LR: 0.001
```

### Hội tụ Training

| Epoch | Loss |
|---|---|
| 1 | 0.23437 |
| 10 | 0.00016 |
| 20 | 0.00016 |
| 30 | 0.00016 |

> Model hội tụ từ epoch 10 — **30 epochs là đủ**.

### Các phương pháp đã thử nghiệm

| Phương pháp | RMSE | So với TF-IDF |
|---|---|---|
| TF-IDF (baseline) | 1.1216 | — |
| TF-IDF + SVD/LSA | 1.1216 | 0% |
| TF-IDF + NMF | 1.1362 | tệ hơn 1.3% |
| **TF-IDF + Autoencoder** | **1.1057** | **tốt hơn 1.41%** |
| Sentence Transformer (SBERT) | 1.1249 | kém hơn 0.3% |

> **Kết luận:** Autoencoder thắng nhờ mapping phi tuyến 5000 → 64 chiều.  
> SBERT kém do corpus tiếng Việt nhỏ, chưa đủ tận dụng pretrained model đa ngôn ngữ.

---

## Phần 2 — Neural Collaborative Filtering (NCF v2)

### Vấn đề của Item-Item KNN cũ

```python
# Cũ: Pearson correlation co dinh, khong hoc
item_similarity = train_matrix.corr(method='pearson').fillna(0)
```

KNN tính similarity **tuyến tính cố định** — không xử lý tốt ma trận thưa, không hiểu tương tác phi tuyến.

### Giải pháp: NCF v2 (NeuMF + Genre Side Features)

**Kiến trúc NeuMF** kết hợp 2 nhánh:

```
User_ID -> Embed -+-> GMF: u x i (element-wise) ----------+
                  |                                        +--> FC --> Rating
Item_ID -> Embed -+-> MLP: concat(u, i+genre) -> 64->32 --+
Genre (32-dim)  ---> Linear projection -> embed_dim
                      (cong vao item embed trong MLP)
```

**Điểm mới — Genre Side Features:**
- 32 thể loại phim (one-hot) → Linear → embed_dim → cộng vào Item Embedding
- Giúp model hiểu đặc tính nội dung phim từ đầu

### Cấu hình tối ưu

| Tham số | Giá trị | Lý do |
|---|---|---|
| `embed_dim` | 16 | Dataset nhỏ (98 items), tránh overfit |
| `layers` | [64, 32] | Đủ sâu, không quá phức tạp |
| `epochs` | 80 | Loss hội tụ tốt |
| `optimizer` | SGD + momentum | Tốt hơn Adam trên sparse data (~9.6%) |
| `batch_size` | 64 | Nhiều update hơn / epoch |

### Các phiên bản NCF đã thử

| Phiên bản | Mô tả | RMSE |
|---|---|---|
| NCF v1 | MLP đơn thuần (lỗi normalize) | 1.9687 |
| NCF v1 (fixed) | Sigmoid + BCELoss | 1.1248 |
| NCF v2 | NeuMF + Genre Features | **1.1133** |

---

## Phần 3 — Pipeline Hoàn Chỉnh

### Kiến trúc tổng thể

```
[Binh luan] -> TF-IDF + AE (30ep) -> text_sim ----+
                                                    |
[Rating]    -> Cosine + Shrinkage ----------------> +--> KNN (alpha=0.2) -> knn_ae_pred
                                                    |
[Rating +   -> NCF v2 NeuMF + Genre (80ep) --------+--> ncf_pred
 Genre]                                             |
                                                    |
                                    0.5 x NCF + 0.5 x KNN_AE
                                                    |
                                                    v
                                    item_user_optimized_results.csv
```

### Tìm alpha tối ưu — Bước 1: KNN + AE

| Alpha (rating) | Text% | RMSE |
|---|---|---|
| 0.2 | 80% | **1.1035** (best) |
| 0.3 | 70% | 1.1068 |
| 0.5 | 50% | 1.1119 |

### Tìm alpha tối ưu — Bước 2: NCF + KNN_AE

| Alpha (NCF) | KNN_AE% | RMSE |
|---|---|---|
| 0.0 | 100% | 1.1035 |
| 0.3 | 70% | 1.0767 |
| **0.5** | **50%** | **1.0729** |
| 0.7 | 30% | 1.0807 |
| 1.0 | 0% | 1.1133 |

> Tỷ lệ **50% NCF + 50% KNN_AE** cho kết quả tốt nhất.

---

## Bảng So Sánh Tổng Hợp

| Bước | Phương pháp | RMSE | MAE | Cải thiện |
|---|---|---|---|---|
| **Baseline** | Item-Item KNN (gốc) | 1.1363 | 0.9247 | — |
| +Text (cũ) | Hybrid KNN + TF-IDF | 1.1216 | 0.9121 | -1.30% |
| +Text (mới) | KNN + TF-IDF Autoencoder | 1.1035 | 0.8976 | -2.88% |
| +DL Rating | NCF v2 (NeuMF + Genre) | 1.1133 | 0.9355 | -2.02% |
| **Pipeline** | **NCF v2 + KNN_AE Hybrid** | **1.0729** | **0.8850** | **-5.57%** |

```
RMSE: 1.1363 -> 1.0729  (-0.0634 = -5.57%)
MAE:  0.9247 -> 0.8850  (-0.0397 = -4.29%)

Y nghia: Sai lech du doan trung binh: 0.885 sao (thay vi 0.925 sao) tren thang 1-5
```

---

## Hướng Dẫn Sử Dụng

### Chạy lại pipeline

```bash
cd crawl_data/data
python ml_pipeline.py
# Thoi gian: ~30 giay
# Output: item_user_optimized_results.csv (ghi de)
```

---

## Ghi Chú Kỹ Thuật

| Câu hỏi | Giải thích |
|---|---|
| Tại sao 30 epochs cho AE? | Loss hội tụ từ epoch 10, 30 đủ tốt và nhanh |
| Tại sao SVD không cải thiện? | 98 phim → SVD 100 components = 100% phương sai → không lọc nhiễu |
| Tại sao SGD thay Adam? | Ma trận thưa 9.6% → Adam bị local minimum → SGD+momentum tổng quát hơn |
| Tại sao 50/50 là tốt nhất? | NCF giỏi rating cực đoan, KNN_AE ổn định vùng trung bình |
