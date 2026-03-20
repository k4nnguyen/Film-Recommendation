import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. TẢI DỮ LIỆU GỐC ---
names = ['user_id', 'item_id', 'rating']
# Giả sử file u.data nằm cùng thư mục
df = pd.read_csv('u.data', sep='\s+', names=names, usecols=[0, 1, 2], engine='python')

# --- 2. CHIA THEO TỶ LỆ 70/30 ---
# random_state=42 giúp kết quả chia luôn giống nhau ở mọi máy tính
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# --- 3. XUẤT RA 2 FILE RIÊNG BIỆT ---
train_df.to_csv('ua_train.csv', index=False, sep='\t')
test_df.to_csv('ua_test.csv', index=False, sep='\t')

print(f"--- ĐÃ TẠO FILE THÀNH CÔNG ---")
print(f"Tổng số mẫu: {len(df)}")
print(f"Số mẫu tập Train (70%): {len(train_df)}")
print(f"Số mẫu tập Test (30%): {len(test_df)}")