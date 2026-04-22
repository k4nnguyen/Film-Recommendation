import os
import time
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

import emoji
from pyvi import ViTokenizer

# --- CẤU HÌNH ĐƯỜNG DẪN ---
DATA_DIR = 'data'
EDA_DIR = 'eda_plots'
REVIEWS_FILE = os.path.join(DATA_DIR, 'movie_reviews.csv')
CLEANED_REVIEWS_FILE = os.path.join(DATA_DIR, 'movie_reviews_cleaned.csv')
ENCODED_METADATA_FILE = os.path.join(DATA_DIR, 'movies_metadata_encoded.csv')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# --- CẤU HÌNH TIỀN XỬ LÝ ---
try:
    response = requests.get("https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt")
    vietnamese_stopwords = set([line.strip().replace(" ", "_") for line in response.text.split('\n') if line])
except:
    vietnamese_stopwords = set()

important_sentiment_words = {
    "hay", "quá", "tốt", "đỉnh", "tuyệt", "tuyệt_vời", "xuất_sắc", "ổn", 
    "dở", "tệ", "chê", "chán", "buồn", "vui", "thích", "ghét", "sợ",
    "không", "chưa", "chẳng", "đẹp", "mới", "cảm_động", "hấp_dẫn"
}
vietnamese_stopwords = vietnamese_stopwords - important_sentiment_words   

VOWELS = r'[aeiouyáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]'

TEENCODE_DICT = {
    "ah" : "à" ,"ko": "không", "k": "không", "kh": "không", "khong": "không", "khum": "không", "hông": "không",
    "dc": "được", "đc": "được", "đx": "được", "dk": "được",
    "vs": "với", "mng": "mọi người", "mn": "mọi người",
    "mik": "mình", "m": "mình", "t": "tôi", "tui": "tôi",
    "nma": "nhưng mà", "ròi": "rồi", "rùi": "rồi", "r": "rồi",
    "típ": "tiếp", "cx": "cũng", "bt": "bình thường", "nx": "nữa",
    "lun": "luôn", "thui": "thôi", "ksao": "không sao",
    "okela": "ok", "oke": "ok", "oki": "ok",
    "nhìu": "nhiều", "flim": "phim", "film": "phim",
    "ngta": "người ta", "mqh": "mối quan hệ", "ny": "người yêu",
    "nvat": "nhân vật", "vde": "vấn đề", "qcao": "quảng cáo" , "cũm": "cũng"
}

def clean_vietnamese_text_final(text):
    if not isinstance(text, str):
        return "không_bình_luận"
    
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.replace('thu gọn', '')
    
    # Chuẩn hóa các chữ kéo dài
    text = re.sub(r'(á|à|ả|ã|ạ|â|ấ|ầ|ẩ|ẫ|ậ|ă|ắ|ằ|ẳ|ẵ|ặ)a+', r'\1', text)
    text = re.sub(r'(ó|ò|ỏ|õ|ọ|ô|ố|ồ|ổ|ỗ|ộ|ơ|ớ|ờ|ở|ỡ|ợ)o+', r'\1', text)
    text = re.sub(r'(é|è|ẻ|ẽ|ẹ|ê|ế|ề|ể|ễ|ệ)e+', r'\1', text)
    text = re.sub(r'(í|ì|ỉ|ĩ|ị)i+', r'\1', text)
    text = re.sub(r'(ú|ù|ủ|ũ|ụ|ư|ứ|ừ|ử|ữ|ự)u+', r'\1', text)
    text = re.sub(r'(ý|ỳ|ỷ|ỹ|ỵ)y+', r'\1', text)
    text = re.sub(r'([a-z\u00C0-\u1EF9])\1+', r'\1', text)
    
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    words = text.split()
    valid_words = []
    for word in words:
        if word in TEENCODE_DICT:
            word = TEENCODE_DICT[word]
            
        if " " not in word and len(word) > 10:
            continue
        if not re.search(VOWELS, word):
            continue
        valid_words.append(word)
        
    filtered_text = ' '.join(valid_words)
    tokenized_text = ViTokenizer.tokenize(filtered_text)
    
    final_words = tokenized_text.split()
    clean_words = [w for w in final_words if w not in vietnamese_stopwords]
    
    result = ' '.join(clean_words)
    return result if result.strip() and len(result.split()) >= 2 else "không_bình_luận"
# --- HÀM CRAWL DATA ---
def get_existing_urls():
    urls = set()
    if os.path.exists(ENCODED_METADATA_FILE):
        try:
            df = pd.read_csv(ENCODED_METADATA_FILE)
            if 'url' in df.columns:
                urls.update(df['url'].dropna().tolist())
        except Exception as e:
            print(f"Lỗi khi đọc {ENCODED_METADATA_FILE}: {e}")
    return urls

def get_crawled_review_urls():
    urls = set()
    if os.path.exists(REVIEWS_FILE):
        try:
            df = pd.read_csv(REVIEWS_FILE)
            if 'Movie_URL' in df.columns:
                urls.update(df['Movie_URL'].dropna().tolist())
        except Exception as e:
            pass
    return urls

def get_momo_poster(url, movie_title):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            img_by_alt = soup.find('img', alt=movie_title)
            if img_by_alt and img_by_alt.get('src'):
                return img_by_alt['src']
            
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if 'cinema.momocdn.net' in src or 'image.tmdb.org' in src:
                    if 'size=M' in src or '/w500/' in src or '/original/' in src:
                        return src
    except:
        pass
    return "https://via.placeholder.com/300x450?text=No+Image"

def extract_movie_metadata(driver):
    metadata = {'title': 'N/A', 'genre': 'N/A', 'release_date': 'N/A', 'country': 'N/A'}
    try:
        metadata['title'] = driver.find_element(By.CSS_SELECTOR, "div.font-bold").text.strip()
        items = driver.find_elements(By.TAG_NAME, "li")
        for li in items:
            text = li.text
            if "Thể loại" in text:
                metadata['genre'] = text.split(":")[-1].strip()
            elif "Ngày chiếu" in text:
                metadata['release_date'] = text.split(":")[-1].strip()
            elif "Quốc gia" in text:
                metadata['country'] = text.split(":")[-1].strip()
    except Exception as e:
        print(f"Lỗi khi trích xuất metadata: {e}")
    return metadata

def crawl_movies():
    print("\n--- Bắt đầu tiến trình Thu thập phim ---")
    existing_urls = get_existing_urls()
    print(f"Đã có {len(existing_urls)} phim trong database. Tiến hành kiểm tra phim mới...")

    driver = webdriver.Chrome()
    base_url = "https://www.momo.vn/cinema/review"
    driver.get(base_url)
    wait = WebDriverWait(driver, 15)

    print("Đang tải danh sách phim...")
    loops = 50
    for i in range(loops):
        try:
            more_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button.border-pink-600")))
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", more_btn)
            time.sleep(1.5) 
            driver.execute_script("arguments[0].click();", more_btn)
            print(f"--- Đã bấm 'Xem thêm' danh sách phim lần {i+1}/{loops} ---")
            time.sleep(2) 
        except Exception:
            print(f"Đã load hết danh sách phim (hoặc không thấy nút 'Xem thêm' ở lần {i+1}).")
            break

    review_elements = driver.find_elements(By.XPATH, "//a[contains(., 'Xem thêm')]")
    all_links = []
    for el in review_elements:
        link = el.get_attribute("href")
        if link and link not in all_links:
            all_links.append(link)
    
    new_links = [link for link in all_links if link not in existing_urls]
    print(f"Tìm thấy tổng cộng {len(all_links)} phim. Trong đó có {len(new_links)} phim mới cần thu thập.")

    if not new_links:
        print("Không có phim mới để thu thập.")
        driver.quit()
        return

    total_new = len(new_links)
    for idx, url in enumerate(new_links):
        print(f"\n[{idx+1}/{total_new}] Đang thu thập thông tin phim: {url}")
        driver.get(url)
        time.sleep(2)

        movie_info = extract_movie_metadata(driver)
        movie_info['url'] = url
        movie_info['poster_url'] = get_momo_poster(url, movie_info['title'])

        # --- Lưu dữ liệu Metadata (Thực hiện One-hot Encoding trực tiếp) ---
        new_row = movie_info.copy()
        
        if os.path.exists(ENCODED_METADATA_FILE):
            df_existing = pd.read_csv(ENCODED_METADATA_FILE)
            existing_cols = list(df_existing.columns)
        else:
            df_existing = pd.DataFrame()
            existing_cols = list(new_row.keys())
            
        original_num_cols = len(existing_cols)
        
        if new_row['genre'] != 'N/A':
            for g in [g.strip() for g in new_row['genre'].split(',')]:
                new_row[g] = 1
                if g not in existing_cols:
                    existing_cols.append(g)
                        
        for k in new_row.keys():
            if k not in existing_cols:
                existing_cols.append(k)

        for col in existing_cols:
            if col not in new_row:
                new_row[col] = 0
                
        if not df_existing.empty and len(existing_cols) > original_num_cols:
            df_new = pd.DataFrame([new_row], columns=existing_cols)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True).fillna(0)
            df_combined.to_csv(ENCODED_METADATA_FILE, index=False, encoding='utf-8-sig')
        else:
            df_new = pd.DataFrame([new_row], columns=existing_cols)
            df_new.to_csv(ENCODED_METADATA_FILE, mode='a', index=False, header=not os.path.exists(ENCODED_METADATA_FILE), encoding='utf-8-sig')

        print(f"Đã lưu thông tin phim: {movie_info['title']}")

    driver.quit()
    print("Hoàn tất thu thập danh sách phim.")

def crawl_reviews():
    print("\n--- Bắt đầu tiến trình Thu thập bình luận ---")
    if not os.path.exists(ENCODED_METADATA_FILE):
        print("Chưa có danh sách phim! Vui lòng chạy chức năng 'Thu thập phim' trước.")
        return
        
    df_meta = pd.read_csv(ENCODED_METADATA_FILE)
    if 'url' not in df_meta.columns or 'title' not in df_meta.columns:
        print("Cấu trúc file phim không hợp lệ.")
        return
        
    all_movies = df_meta[['title', 'url']].dropna().to_dict('records')
    crawled_urls = get_crawled_review_urls()
    
    movies_to_crawl = [m for m in all_movies if m['url'] not in crawled_urls]
    print(f"Tổng cộng {len(all_movies)} phim. Có {len(movies_to_crawl)} phim chưa thu thập bình luận.")
    
    if not movies_to_crawl:
        print("Tất cả phim đã được thu thập bình luận.")
        return
        
    driver = webdriver.Chrome()
    total_movies = len(movies_to_crawl)
    
    for idx, movie in enumerate(movies_to_crawl):
        url = movie['url']
        title = movie['title']
        print(f"\n[{idx+1}/{total_movies}] Đang thu thập bình luận phim: {title}")
        driver.get(url)
        time.sleep(2)
        
        # Lấy review
        review_loops = 10
        for i in range(review_loops):
            try:
                more_btns = driver.find_elements(By.XPATH, "//button[contains(., 'Xem tiếp nhé!')]")
                if not more_btns: 
                    print(f"  -> Đã mở hết bình luận (hoàn thành ở lần cuộn thứ {i}).")
                    break
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", more_btns[0])
                time.sleep(1.5)
                driver.execute_script("arguments[0].click();", more_btns[0])
                print(f"  -> Đang tải thêm bình luận: Lần cuộn {i+1}/{review_loops}...")
                time.sleep(1.25)
            except:
                break
        
        read_more_btns = driver.find_elements(By.XPATH, "//span[contains(text(), '...Xem thêm')]")
        for btn in read_more_btns:
            try: driver.execute_script("arguments[0].click();", btn)
            except: continue

        reviews_data = []
        review_items = driver.find_elements(By.CSS_SELECTOR, "div.relative.mt-3")
        for item in review_items:
            try:
                comment_text = item.find_element(By.CSS_SELECTOR, "div.leading-relaxed").text.strip()
                rating_raw = item.find_element(By.XPATH, ".//span[contains(@class, 'pl-0.5')]").text.strip()
                reviews_data.append({
                    "rating": rating_raw.split("/")[0],
                    "comment": comment_text,
                    "Movie_Title": title,
                    "Movie_URL": url
                })
            except:
                continue

        if not reviews_data:
            reviews_data.append({
                "rating": "",
                "comment": "không_bình_luận",
                "Movie_Title": title,
                "Movie_URL": url
            })

        df_reviews = pd.DataFrame(reviews_data)
        df_reviews.to_csv(REVIEWS_FILE, mode='a', index=False, header=not os.path.exists(REVIEWS_FILE), encoding='utf-8-sig')
        
        valid_reviews_count = len(reviews_data) if reviews_data[0]['rating'] != "" else 0
        print(f"Đã lưu xong {title} - {valid_reviews_count} reviews")

    driver.quit()
    print("Hoàn tất thu thập bình luận.")


# --- HÀM EDA ---
def plot_eda(df, prefix="before"):
    if df.empty: return
    
    df_plot = df.copy()
    df_plot = df.copy()
    
    # Lọc bỏ các đánh giá rỗng (không có bình luận thực sự) khỏi tất cả các biểu đồ
    text_col = 'comment' if 'comment' in df_plot.columns else 'clean_comment'
    if text_col in df_plot.columns:
        df_plot = df_plot[df_plot[text_col] != "không_bình_luận"].copy()
        
    if df_plot.empty:
        print(f"Không có bình luận hợp lệ nào để vẽ biểu đồ {prefix}.")
        return
    
    # 1. Biểu đồ Đám mây từ vựng (WordCloud)
    try:
        from wordcloud import WordCloud
        if text_col in df_plot.columns:
            text_data = " ".join(df_plot[text_col].dropna().astype(str).tolist())
            if text_data.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Đám mây từ vựng (WordCloud) - {prefix.upper()}')
                plt.savefig(os.path.join(EDA_DIR, f'wordcloud_{prefix}.png'))
                plt.close()
    except ImportError:
        print("Thư viện 'wordcloud' chưa được cài đặt. Vui lòng chạy 'pip install wordcloud' để xem biểu đồ này.")

    # 2. Phân phối độ dài bình luận
    if text_col in df_plot.columns:
        valid_comments = df_plot.copy()
        
        if not valid_comments.empty:
            valid_comments['word_count'] = valid_comments[text_col].apply(lambda x: len(str(x).split()))
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_comments['word_count'], bins=50, kde=True, color='blue')
            plt.title(f'Phân phối số lượng từ - {prefix.upper()}')
            plt.xlabel('Số lượng từ')
            plt.ylabel('Tần suất')
            plt.xlim(0, max(10, valid_comments['word_count'].quantile(0.95)))
            plt.savefig(os.path.join(EDA_DIR, f'word_count_distribution_{prefix}.png'))
            plt.close()

    # 3. Top 10 phim
    if 'Movie_Title' in df_plot.columns:
        plt.figure(figsize=(12, 6))
        top_movies = df_plot['Movie_Title'].value_counts().head(10)
        sns.barplot(y=top_movies.index, x=top_movies.values, hue=top_movies.index, palette='magma', legend=False)
        plt.title(f'Top 10 phim có nhiều bình luận nhất - {prefix.upper()}')
        plt.xlabel('Số lượng bình luận')
        plt.ylabel('Tên phim')
        plt.savefig(os.path.join(EDA_DIR, f'top_10_movies_{prefix}.png'))
        plt.close()

# --- HÀM PREPROCESS ---
def preprocess_data():
    print("\n--- Bắt đầu tiến trình Tiền xử lý ---")
    if os.path.exists(REVIEWS_FILE):
        df_reviews = pd.read_csv(REVIEWS_FILE)
        df_reviews['comment'] = df_reviews['comment'].fillna("không_bình_luận")
        
        print("Đang thực hiện làm sạch bình luận (có thể tốn chút thời gian)...")
        df_reviews['clean_comment'] = df_reviews['comment'].apply(clean_vietnamese_text_final)
        
        final_df = df_reviews[['rating', 'clean_comment', 'Movie_Title', 'Movie_URL']]
        final_df.to_csv(CLEANED_REVIEWS_FILE, index=False, encoding='utf-8-sig')
        print(f"Đã lưu bình luận sạch vào {CLEANED_REVIEWS_FILE}")
    else:
        print(f"Không tìm thấy file {REVIEWS_FILE} để tiền xử lý.")

# --- HÀM EDA DATA ---
def eda_data():
    print("\n--- Bắt đầu tiến trình EDA (Vẽ biểu đồ) ---")
    if os.path.exists(REVIEWS_FILE):
        print("Đang tạo EDA trước khi tiền xử lý...")
        df_reviews = pd.read_csv(REVIEWS_FILE)
        df_reviews['comment'] = df_reviews['comment'].fillna("không_bình_luận")
        plot_eda(df_reviews, "before")
    else:
        print(f"Không tìm thấy file {REVIEWS_FILE} để vẽ EDA trước.")
        
    if os.path.exists(CLEANED_REVIEWS_FILE):
        print("Đang tạo EDA sau khi tiền xử lý...")
        final_df = pd.read_csv(CLEANED_REVIEWS_FILE)
        plot_eda(final_df, "after")
    else:
        print(f"Không tìm thấy file {CLEANED_REVIEWS_FILE} để vẽ EDA sau.")

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("=== MENU CHƯƠNG TRÌNH ===")
        print("1. Thu thập phim (Lấy thông tin và Poster)")
        print("2. Thu thập bình luận (Từ danh sách phim đã thu thập)")
        print("3. Tiền xử lý dữ liệu (Làm sạch bình luận)")
        print("4. EDA (Vẽ biểu đồ phân phối, top phim)")
        print("5. Full Pipeline (Chạy toàn bộ từ 1 -> 4)")
        print("0. Thoát chương trình")
        print("="*40)
        
        choice = input("Nhập lựa chọn của bạn (0-5): ").strip()
        
        if choice == '1':
            crawl_movies()
        elif choice == '2':
            crawl_reviews()
        elif choice == '3':
            preprocess_data()
        elif choice == '4':
            eda_data()
        elif choice == '5':
            print("Đang khởi động chuỗi Full Pipeline...")
            crawl_movies()
            crawl_reviews()
            preprocess_data()
            eda_data()
            print("\n=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")
        elif choice == '0':
            print("Đã thoát chương trình. Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập lại!")
