import pandas as pd

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
users = pd.read_csv("./ml-100k/u.user", sep="|", names= u_cols)

rating_base = pd.read_csv("./ml-100k/ua.base",sep="\t",names = r_cols)
rating_test = pd.read_csv("./ml-100k/ua.test",sep="\t",names = r_cols)
rate_base = rating_base.to_numpy()
rate_test = rating_test.to_numpy()

print(rating_base.shape[0])
print(rating_base.head())
print(rate_base)
print(rate_test)

print(rate_base.shape[0])
print(rate_test.shape[0])