from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

movies = []
review_links = []
base_url = "https://www.momo.vn/cinema"
url = base_url + "/review"

print(f"Đang bắt đầu crawl dữ liệu từ: {url}")
driver = webdriver.Chrome()
driver.get(url)
wait = WebDriverWait(driver, 15)
more_btn = driver.find_element(By.CSS_SELECTOR,"button.border-pink-600")
print(more_btn)

loops = 120

# Xử lý nút xem thêm để lấy list các film
for i in range(loops):
    try:
        more_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "button.border-pink-600")))
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", more_btn)
        time.sleep(1.5) 
        driver.execute_script("arguments[0].click();", more_btn)
        print(f"--- Đã bấm 'Xem thêm' lần {i+1} ---")
        time.sleep(2) 
        
    except Exception:
        print("Đã load hết dữ liệu hoặc không tìm thấy nút 'Xem thêm' nữa.")
        break

# Xử lý chữ xem thêm để lấy link
review_elements = driver.find_elements(By.XPATH, "//a[contains(., 'Xem thêm')]")
for i in review_elements:
    link = i.get_attribute("href")
    if link not in review_links: # Tránh lấy trùng
        review_links.append(link)
print(f"Đã tìm thấy tổng cộng {len(review_links)} đường dẫn review.")
for i, link in enumerate(review_links[:5]): # In thử 5 link đầu
    print(f"{i+1}. {link}")

import csv 
# Chuyển link vào file csv
file_name = "link_list_momo.csv"
with open(file_name,mode='w',newline='',encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['Review_link'])
    for link in review_links:
        writer.writerow([link])
print(f"Đã xuất thành công {len(review_links)} link vào file {file_name}")

import pandas as pd
import os
def extract_movie_metadata(container_element):
    metadata = {}
    try:
        # Tên phim
        metadata['title'] = container_element.find_element(By.CSS_SELECTOR, "div.font-bold").text.strip()
        
        # Thể loại, ngày xuất bản, quốc gia
        items = container_element.find_elements(By.TAG_NAME, "li")
        metadata['genre'] = "N/A"
        metadata['release_date'] = "N/A"
        metadata['country'] = "N/A"
        
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

def save_to_csv(all_movies_data, reviews_data, url):
    if not all_movies_data: return
    movie_title = all_movies_data[0]['title']
    
    # Lưu Metadata
    df_meta = pd.DataFrame(all_movies_data)
    df_meta.to_csv('movies_metadata.csv', mode='a', index=False, 
                   header=not os.path.exists('movies_metadata.csv'), encoding='utf-8-sig')

    # Lưu Reviews
    # for r in reviews_data:
    #     r['Movie_Title'] = movie_title
    #     r['Movie_URL'] = url
    # df_reviews = pd.DataFrame(reviews_data)
    # df_reviews.to_csv('movie_reviews.csv', mode='a', index=False, 
    #                   header=not os.path.exists('movie_reviews.csv'), encoding='utf-8-sig')
    # print(f"Đã xong: {movie_title} ({len(reviews_data)} reviews)")

df_links = pd.read_csv('link_list_momo.csv')
urls = df_links['Review_link'].tolist()
print(urls)
for current_url in urls:
    print(f"\nĐang truy cập: {current_url}")
    all_movies_data = []
    reviews_data = []
    
    try:
        driver.get(current_url)
        time.sleep(2) # Đợi trang ổn định
        
        # 2. Lấy Metadata 
        try:
            movie_info = extract_movie_metadata(driver)
            movie_info['url'] = current_url
            all_movies_data.append(movie_info)
        except:
            print("Không lấy được metadata, bỏ qua phim này.")
            continue

        # 3. Vòng lặp "Xem tiếp nhé!"
        loops = 50 # Bạn có thể điều chỉnh số lần bấm tùy ý
        for i in range(loops):
            try:
                more_btns = driver.find_elements(By.XPATH, "//button[contains(., 'Xem tiếp nhé!')]")
                if len(more_btns) == 0:
                    print(f"--- Đã hết bình luận tại lần scroll thứ {i}, chuyển sang trích xuất... ---")
                    break
                more_btn = more_btns[0]
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", more_btn)
                print(f"Đang scroll lần thứ {i+1}")
                time.sleep(1.5)
                driver.execute_script("arguments[0].click();", more_btn)
                time.sleep(1.25)
            except:
                break

        # 4. Mở rộng "...Xem thêm" bình luận
        read_more_btns = driver.find_elements(By.XPATH, "//span[contains(text(), '...Xem thêm')]")
        for btn in read_more_btns:
            try: driver.execute_script("arguments[0].click();", btn)
            except: continue
        
        # 5. Trích xuất Review
        review_items = driver.find_elements(By.CSS_SELECTOR, "div.relative.mt-3")
        for item in review_items:
            try:
                comment_text = item.find_element(By.CSS_SELECTOR, "div.leading-relaxed").text.strip()
                rating_element = item.find_element(By.XPATH, ".//span[contains(@class, 'pl-0.5')]")
                rating_raw = rating_element.text.strip()
                
                reviews_data.append({
                    "rating": rating_raw.split("/")[0],
                    "comment": comment_text
                })
            except:
                continue

        # 6. Lưu vào CSV ngay lập tức sau mỗi phim để tránh mất dữ liệu
        save_to_csv(all_movies_data, reviews_data, current_url)

    except Exception as e:
        print(f"Lỗi tại {current_url}: {e}")

def save_to_csv(all_movies_data):
    if not all_movies_data: 
        return
    
    # 1. Chuyển list metadata thành DataFrame
    df_meta = pd.DataFrame(all_movies_data)
    
    # 2. Đường dẫn file lưu
    file_name = 'movies_metadata.csv'
    
    # 3. Ghi vào CSV (mode='a' để ghi tiếp, header chỉ ghi lần đầu)
    df_meta.to_csv(file_name, mode='a', index=False, 
                   header=not os.path.exists(file_name), 
                   encoding='utf-8-sig')
    
    print(f"Đã lưu metadata phim: {all_movies_data[0].get('title', 'N/A')}")

import pandas as pd
df = pd.read_csv('link_list_momo.csv')
urls = df['Review_link'].tolist()
for url in urls:
    try:
        driver.get(url)
        time.sleep(1)
        movie_info = extract_movie_metadata(driver)
        movie_info['url'] = url
        
        # Tạo list chứa 1 dict phim hiện tại để truyền vào hàm
        current_movie_list = [movie_info] 
        
        # Chỉ truyền list metadata vào hàm
        save_to_csv(current_movie_list)
        
    except Exception as e:
        print(f"Lỗi: {e}")
    
    

import pandas as pd

# 1. Đọc file kết quả metadata mà bạn vừa lưu
try:
    df_result = pd.read_csv('movies_metadata.csv')

    # 2. Khởi tạo một set trống
    unique_genres = set()

    # 3. Lặp qua cột 'genre'
    # .dropna() để tránh lỗi nếu có hàng nào đó bị trống dữ liệu
    for genres_str in df_result['genre'].dropna():
        # Kiểm tra nếu dữ liệu là N/A (do hàm extract_metadata của bạn gán) thì bỏ qua
        if genres_str == "N/A":
            continue
            
        # Tách chuỗi "Chính kịch, Hài" thành list ["Chính kịch", "Hài"]
        genres_list = [g.strip() for g in genres_str.split(',')]
        
        # Thêm list này vào set (update sẽ tự động lọc trùng)
        unique_genres.update(genres_list)

    # 4. In kết quả để kiểm tra
    print(f"Đã trích xuất thành công {len(unique_genres)} thể loại duy nhất:")
    print(unique_genres)

except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'movies_metadata.csv'. Hãy kiểm tra lại quá trình lưu file.")