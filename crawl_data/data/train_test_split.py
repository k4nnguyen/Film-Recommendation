import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. TẢI DỮ LIỆU GỐC ---
# u.data dùng khoảng trắng hoặc tab (\s+) làm phân cách 
names = ['user_id', 'item_id', 'rating']
df = pd.read_csv('u.data', sep='\s+', names=names, usecols=[0, 1, 2], engine='python')

# --- 2. CHIA THEO TỶ LỆ 70/30 ---
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# --- 3. XUẤT RA FILE CSV CHUẨN (Dùng dấu phẩy) ---
# Dùng sep=',' để Excel tự động chia cột chính xác
train_df.to_csv('ua_train.csv', index=False, sep=',')
test_df.to_csv('ua_test.csv', index=False, sep=',')

print("--- ĐÃ TẠO FILE THÀNH CÔNG ---")
print(f"File Train: {len(train_df)} dòng")
print(f"File Test: {len(test_df)} dòng")
