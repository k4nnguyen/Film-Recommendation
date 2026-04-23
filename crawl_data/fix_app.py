import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: replace hardcoded range(1, 84) in corpus loop
content = content.replace(
    "    for item_id in range(1, 84):\n        # item_id 1 tương ứng index 0 trong movies_with_posters\n        title = movies_df.iloc[item_id - 1]['title'] if (item_id - 1) < len(movies_df) else \"\"\n        text = movie_texts.get(title, \"\")\n        corpus.append(text)",
    "    for item_id in range(1, num_movies + 1):\n        title = movies_df.iloc[item_id - 1]['title'] if (item_id - 1) < len(movies_df) else \"\"\n        text = movie_texts.get(title, \"\")\n        corpus.append(text)"
)

# Fix: replace hardcoded index in text_sim_df
content = content.replace(
    "    text_sim_df = pd.DataFrame(text_sim_matrix, index=range(1, 84), columns=range(1, 84))",
    "    text_sim_df = pd.DataFrame(text_sim_matrix, index=range(1, num_movies + 1), columns=range(1, num_movies + 1))"
)

# Fix: remove duplicate movies_df read (already loaded earlier)
content = content.replace(
    "    movies_df = pd.read_csv('movies_metadata_encoded.csv')\n    try:\n        reviews_df",
    "    try:\n        reviews_df"
)

# Add comment for section 3.3
content = content.replace(
    "    # 3.3. Tải dữ liệu Text (Bình luận phim)\n    try:",
    "    # 3.3. Tải dữ liệu Text (Bình luận phim)\n    # movies_df và num_movies đã được đọc ở bước trên\n    try:"
)

with open('data/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Hoàn tất! Kiểm tra kết quả:")
# Verify
with open('data/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if 'range(1,' in line or 'num_movies' in line:
        print(f"  Line {i}: {line.rstrip()}")
