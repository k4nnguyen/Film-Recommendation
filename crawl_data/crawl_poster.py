import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def get_momo_poster(url, movie_title):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # ƯU TIÊN 1: Tìm chính xác thẻ img có alt là tên phim
            # Ví dụ: <img alt="Thỏ Ơi!!" ...>
            img_by_alt = soup.find('img', alt=movie_title)
            if img_by_alt and img_by_alt.get('src'):
                return img_by_alt['src']
            
            # ƯU TIÊN 2: Nếu không tìm thấy theo alt, quét tất cả img 
            # để tìm link từ các nguồn poster quen thuộc (Momo hoặc TMDB)
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if 'cinema.momocdn.net' in src or 'image.tmdb.org' in src:
                    # Kiểm tra xem có phải là ảnh poster (kích thước lớn) không
                    if 'size=M' in src or '/w500/' in src or '/original/' in src:
                        return src
                        
    except Exception as e:
        print(f"Lỗi khi xử lý phim '{movie_title}': {e}")
    return None

# Đọc file dữ liệu
df = pd.read_csv('data/movies_metadata_encoded.csv')

print("🚀 Đang bắt đầu quét poster toàn diện...")
posters = []
for index, row in df.iterrows():
    print(f"[{index+1}/{len(df)}] Đang lấy ảnh cho: {row['title']}...")
    # Truyền thêm cả title vào hàm để tìm theo alt
    poster_url = get_momo_poster(row['url'], row['title'])
    posters.append(poster_url if poster_url else "https://via.placeholder.com/300x450?text=No+Image")
    
    # Nghỉ 1-2 giây để không bị máy chủ chặn
    time.sleep(1.5)

df['poster_url'] = posters
df.to_csv('data/movies_with_posters.csv', index=False, encoding='utf-8-sig')
print("✅ Hoàn tất! Bạn đã có đầy đủ link từ cả Momo và TMDB.")