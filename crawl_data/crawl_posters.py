import sys
sys.stdout.reconfigure(encoding='utf-8')

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
            img_by_alt = soup.find('img', alt=movie_title)
            if img_by_alt and img_by_alt.get('src'):
                return img_by_alt['src']
            
            # ƯU TIÊN 2: Quét tất cả img tìm link poster quen thuộc (Momo hoặc TMDB)
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if 'cinema.momocdn.net' in src or 'image.tmdb.org' in src:
                    if 'size=M' in src or '/w500/' in src or '/original/' in src:
                        return src
                        
    except Exception as e:
        print(f"  Lỗi khi xử lý phim '{movie_title}': {e}")
    return None

# Đọc file dữ liệu
df = pd.read_csv('data/movies_metadata_encoded.csv', encoding='utf-8-sig')

print(f"Tìm thấy {len(df)} phim. Bắt đầu crawl poster...\n")
posters = []
failed = []

for index, row in df.iterrows():
    print(f"[{index+1}/{len(df)}] {row['title']}...", end=" ", flush=True)
    poster_url = get_momo_poster(row['url'], row['title'])
    
    if poster_url:
        print(f"OK")
        posters.append(poster_url)
    else:
        print(f"KHÔNG TÌM THẤY -> dùng placeholder")
        posters.append("https://placehold.co/300x450?text=No+Image")
        failed.append(row['title'])
    
    time.sleep(1.5)

df['poster_url'] = posters
df.to_csv('data/movies_with_posters.csv', index=False, encoding='utf-8-sig')

print(f"\n{'='*50}")
print(f"HOÀN TẤT!")
print(f"- Thành công : {len(df) - len(failed)}/{len(df)} phim")
print(f"- Thất bại   : {len(failed)} phim")
if failed:
    print(f"- Danh sách thất bại:")
    for t in failed:
        print(f"    * {t}")
print(f"\nĐã lưu vào: data/movies_with_posters.csv")
