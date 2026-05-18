# -*- coding: utf-8 -*-
"""
============================================================
  ML PIPELINE HOAN CHINH - He Thong Goi Y Phim
============================================================
Pipeline ket hop 2 ky thuat chinh:

  [A] Trich chon dac trung tu dong cho binh luan:
      TF-IDF (5000 features) + Autoencoder (64 chieu, 30 epochs)
      -> text_sim_matrix (ngu nghia bình luan)

  [B] Neural Collaborative Filtering:
      NCF v2 (NeuMF + Genre Side Features) + Item-Item KNN
      -> Hybrid NCF+KNN prediction matrix

  [C] Ket hop cuoi:
      alpha x NCF_pred + (1-alpha) x KNN_with_AE_text_sim
      -> Ma tran du doan cuoi cung -> item_user_optimized_results.csv

Chay: python ml_pipeline.py
============================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# CAU HINH PIPELINE
# ============================================================
AE_LATENT_DIM   = 64      # chieu embedding autoencoder
AE_EPOCHS       = 30      # so epoch train autoencoder
AE_LR           = 0.001
AE_BATCH_SIZE   = 16

NCF_EMBED_DIM   = 16      # chieu embedding NCF
NCF_LAYERS      = [64, 32]
NCF_EPOCHS      = 80
NCF_BATCH_SIZE  = 64
NCF_LR          = 0.005
NCF_DROPOUT     = 0.1

KNN_K           = 10
KNN_SHRINKAGE   = 5
ALPHAS          = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]

RATING_MIN, RATING_MAX = 1.0, 5.0

# ============================================================
# PHAN 1: LOAD DU LIEU
# ============================================================
def load_all_data():
    print("[1/6] Loading du lieu...")
    train_df  = pd.read_csv(os.path.join(BASE_DIR, 'ua_train.csv'))
    test_df   = pd.read_csv(os.path.join(BASE_DIR, 'ua_test.csv'))
    movies_df = pd.read_csv(os.path.join(BASE_DIR, 'movies_metadata_encoded.csv'),
                            encoding='utf-8-sig')

    # Doc binh luan
    try:
        rev_df      = pd.read_csv(os.path.join(BASE_DIR, 'movie_reviews_cleaned.csv'))
        rev_df      = rev_df.dropna(subset=['clean_comment'])
        movie_texts = (rev_df.groupby('Movie_Title')['clean_comment']
                       .apply(lambda x: ' '.join(x)).to_dict())
    except Exception as e:
        print(f"  Loi doc reviews: {e}"); movie_texts = {}

    n_items = len(movies_df)
    corpus  = [movie_texts.get(movies_df.iloc[i]['title'], "phim khong co binh luan")
               for i in range(n_items)]

    # Genre matrix (1-indexed: hang 0 la padding)
    genre_cols   = movies_df.select_dtypes(include=['number']).columns.tolist()
    genre_vals   = movies_df[genre_cols].values.astype(np.float32)
    n_genres     = len(genre_cols)
    genre_matrix = np.vstack([np.zeros((1, n_genres), dtype=np.float32), genre_vals])

    # Rating matrix
    n_users = max(train_df['user_id'].max(), test_df['user_id'].max())
    train_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')
    for i in range(1, n_items + 1):
        if i not in train_matrix.columns:
            train_matrix[i] = np.nan
    train_matrix = train_matrix.reindex(sorted(train_matrix.columns), axis=1)

    has_bl = sum(1 for c in corpus if c != "phim khong co binh luan")
    print(f"  Users: {n_users} | Items: {n_items} | Genres: {n_genres}")
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Co binh luan: {has_bl}/{n_items} phim")
    return (train_df, test_df, corpus, train_matrix,
            genre_matrix, n_users, n_items, n_genres)

# ============================================================
# PHAN 2A: TF-IDF + AUTOENCODER
# ============================================================
class TextAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim),nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),nn.ReLU(),
            nn.Linear(128, 512),       nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def build_ae_text_sim(corpus, n_items):
    print(f"\n[2/6] Trich chon dac trung binh luan: TF-IDF + Autoencoder ({AE_EPOCHS} epochs)...")

    # TF-IDF
    vec         = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_mat   = vec.fit_transform(corpus)
    X_dense     = tfidf_mat.toarray().astype(np.float32)
    input_dim   = X_dense.shape[1]
    print(f"  TF-IDF matrix: {X_dense.shape}")

    # Autoencoder
    X_tensor    = torch.FloatTensor(X_dense)
    loader      = DataLoader(TensorDataset(X_tensor, X_tensor),
                             batch_size=AE_BATCH_SIZE, shuffle=True)
    model       = TextAutoencoder(input_dim, AE_LATENT_DIM)
    opt         = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    crit        = nn.MSELoss()
    sched       = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=AE_EPOCHS)
    best_loss, best_state = float('inf'), None

    print(f"  {'Epoch':>6} | {'Loss':>10}")
    print("  " + "-" * 20)
    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, _ in loader:
            opt.zero_grad()
            out, _ = model(xb)
            loss   = crit(out, xb)
            loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(loader)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6} | {avg:>10.5f}")

    # Lay embedding tu encoder
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, z = model(X_tensor)
    z_np = z.numpy()
    print(f"  AE embedding: {z_np.shape} | Best loss: {best_loss:.5f}")

    # Tinh sim matrix
    sim    = cosine_similarity(normalize(z_np))
    sim_df = pd.DataFrame(sim, index=range(1, n_items+1), columns=range(1, n_items+1))
    return sim_df

# ============================================================
# PHAN 2B: RATING SIMILARITY (Item-Item, Cosine + Shrinkage)
# ============================================================
def build_rating_sim(train_matrix):
    raw = pd.DataFrame(cosine_similarity(train_matrix.fillna(0).T),
                       index=train_matrix.columns, columns=train_matrix.columns)
    mask = train_matrix.notna().values
    S    = raw.values.copy()
    n    = len(S)
    for i in range(n):
        for j in range(i+1, n):
            nc = int(np.sum(mask[:,i] & mask[:,j]))
            f  = nc / (nc + KNN_SHRINKAGE)
            S[i,j] *= f; S[j,i] *= f
    return pd.DataFrame(S, index=raw.index, columns=raw.columns)

# ============================================================
# PHAN 3: ITEM-ITEM KNN (Z-score)
# ============================================================
def knn_predict(train_matrix, sim_df):
    gmean = float(train_matrix.stack().mean())
    mu    = train_matrix.mean(axis=1).values.reshape(-1,1)
    std   = train_matrix.std(axis=1).values.reshape(-1,1); std[std==0]=1.0
    zd    = ((train_matrix-mu)/std).fillna(0).values
    S     = sim_df.values; mask = train_matrix.notna().values
    pred  = np.zeros(train_matrix.shape)
    for j in range(train_matrix.shape[1]):
        col     = S[:,j].copy(); col[j] = -2
        top     = np.argsort(col)[:-KNN_K-1:-1]; w = col[top]
        sw      = np.sum(mask[:,top]*np.abs(w),axis=1); sw[sw==0]=1e-9
        pred[:,j] = mu.flatten() + zd[:,top].dot(w)/(sw+3.0)*std.flatten()
    pred = np.where(np.isnan(pred), gmean, pred)
    return pd.DataFrame(np.clip(pred,1.,5.),
                        index=train_matrix.index, columns=train_matrix.columns)

# ============================================================
# PHAN 4: NCF v2 (NeuMF + Genre Side Features)
# ============================================================
def norm(r): return (r - RATING_MIN) / (RATING_MAX - RATING_MIN)
def denorm(r): return r * (RATING_MAX - RATING_MIN) + RATING_MIN

class RatingDS(torch.utils.data.Dataset):
    def __init__(self, df, gm):
        self.u = torch.LongTensor(df['user_id'].values)
        self.i = torch.LongTensor(df['item_id'].values)
        self.r = torch.FloatTensor(norm(df['rating'].values.astype(float)))
        self.gm = torch.FloatTensor(gm)
    def __len__(self): return len(self.r)
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.gm[self.i[idx]], self.r[idx]

class NCFv2(nn.Module):
    def __init__(self, n_users, n_items, n_genres, embed_dim, layers, dropout):
        super().__init__()
        self.gmf_u = nn.Embedding(n_users+1, embed_dim)
        self.gmf_i = nn.Embedding(n_items+1, embed_dim)
        self.mlp_u = nn.Embedding(n_users+1, embed_dim)
        self.mlp_i = nn.Embedding(n_items+1, embed_dim)
        self.gproj = nn.Sequential(nn.Linear(n_genres, embed_dim), nn.ReLU())
        seq=[]; ins=embed_dim*2
        for outs in layers:
            seq += [nn.Linear(ins,outs), nn.ReLU(), nn.Dropout(dropout)]; ins=outs
        self.mlp = nn.Sequential(*seq)
        self.out  = nn.Linear(embed_dim+layers[-1], 1)
        for emb in [self.gmf_u,self.gmf_i,self.mlp_u,self.mlp_i]:
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, u, i, g):
        gmf = self.gmf_u(u) * self.gmf_i(i)
        gp  = self.gproj(g)
        mlp = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)+gp], dim=1))
        return torch.sigmoid(self.out(torch.cat([gmf,mlp],dim=1)).squeeze())

def train_ncf(train_df, test_df, genre_matrix, n_users, n_items, n_genres):
    print(f"\n[3/6] Train NCF v2 (NeuMF + Genre, {NCF_EPOCHS} epochs)...")
    loader_tr = DataLoader(RatingDS(train_df, genre_matrix),
                           batch_size=NCF_BATCH_SIZE, shuffle=True)
    loader_te = DataLoader(RatingDS(test_df,  genre_matrix),
                           batch_size=NCF_BATCH_SIZE, shuffle=False)
    model   = NCFv2(n_users, n_items, n_genres,
                    NCF_EMBED_DIM, NCF_LAYERS, NCF_DROPOUT)
    opt     = torch.optim.SGD(model.parameters(), lr=NCF_LR,
                               momentum=0.9, weight_decay=1e-4)
    crit    = nn.BCELoss()
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NCF_EPOCHS)
    best_rmse, best_state = float('inf'), None

    print(f"  {'Epoch':>6} | {'Loss':>10} | {'RMSE':>8}")
    print("  " + "-" * 32)
    for epoch in range(1, NCF_EPOCHS+1):
        model.train()
        total = 0.0
        for u,i,g,r in loader_tr:
            opt.zero_grad()
            loss = crit(model(u,i,g), r)
            loss.backward(); opt.step(); total += loss.item()
        sched.step()
        model.eval()
        py, ty = [], []
        with torch.no_grad():
            for u,i,g,r in loader_te:
                p = model(u,i,g).numpy()
                py.extend(np.clip(denorm(p),1.,5.))
                ty.extend(denorm(r.numpy()))
        rmse = np.sqrt(mean_squared_error(ty, py))
        if rmse < best_rmse:
            best_rmse  = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            avg = total / len(loader_tr)
            print(f"  {epoch:>6} | {avg:>10.4f} | {rmse:>8.4f}")

    model.load_state_dict(best_state)
    print(f"  NCF v2 best RMSE: {best_rmse:.4f}")
    return model, best_rmse

def ncf_pred_matrix(model, genre_matrix, n_users, n_items):
    model.eval()
    gm      = torch.FloatTensor(genre_matrix)
    all_u   = torch.arange(1, n_users+1).repeat_interleave(n_items)
    all_i   = torch.arange(1, n_items+1).repeat(n_users)
    all_g   = gm[all_i]
    with torch.no_grad():
        p   = model(all_u, all_i, all_g).numpy()
    preds = np.clip(denorm(p), 1., 5.).reshape(n_users, n_items)
    return pd.DataFrame(preds,
                        index=range(1, n_users+1),
                        columns=range(1, n_items+1))

# ============================================================
# PHAN 5: DANH GIA
# ============================================================
def evaluate(pred_df, test_df):
    yt, yp = [], []
    for r in test_df.itertuples():
        if r.user_id in pred_df.index and r.item_id in pred_df.columns:
            yt.append(r.rating); yp.append(pred_df.loc[r.user_id, r.item_id])
    return (np.sqrt(mean_squared_error(yt,yp)), mean_absolute_error(yt,yp))

# ============================================================
# PHAN 6: TIM ALPHA TOI UU VA XUAT KET QUA
# ============================================================
def find_best_final(ncf_df, knn_df, test_df, alphas):
    print(f"\n[5/6] Tim alpha toi uu: alpha*NCF + (1-alpha)*KNN...")
    print(f"  {'Alpha':>6} | {'NCF%':>5} | {'KNN%':>5} | {'RMSE':>8} | {'MAE':>8}")
    print("  " + "-" * 42)
    best = {'rmse': float('inf'), 'mae':None, 'alpha':None, 'pred_df':None}
    for a in alphas:
        # Chỉ blend các user/item có trong cả 2
        common_users = ncf_df.index.intersection(knn_df.index)
        common_items = ncf_df.columns.intersection(knn_df.columns)
        hybrid = a * ncf_df.loc[common_users, common_items] + \
                 (1-a) * knn_df.loc[common_users, common_items]
        rmse, mae = evaluate(hybrid, test_df)
        mark = " <--" if rmse < best['rmse'] else ""
        print(f"  {a:>6.1f} | {int(a*100):>4d}% | {int((1-a)*100):>4d}% | {rmse:>8.4f} | {mae:>8.4f}{mark}")
        if rmse < best['rmse']:
            best = {'rmse':rmse,'mae':mae,'alpha':a,'pred_df':hybrid}
    return best

# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == '__main__':
    t_total = time.time()

    print("=" * 62)
    print("  ML PIPELINE HOAN CHINH: TF-IDF+AE + NCF v2 + KNN Hybrid")
    print("=" * 62)

    # ---- Load ----
    (train_df, test_df, corpus, train_matrix,
     genre_matrix, n_users, n_items, n_genres) = load_all_data()

    # ---- [BASELINE] KNN goc (khong co gi) ----
    print("\n[*] Tinh baseline KNN (truoc khi cai tien)...")
    rsim     = build_rating_sim(train_matrix)
    knn_base = knn_predict(train_matrix, rsim)
    rmse_base, mae_base = evaluate(knn_base, test_df)
    print(f"  Baseline KNN: RMSE={rmse_base:.4f} | MAE={mae_base:.4f}")

    # ---- [A] TF-IDF + Autoencoder ----
    ae_sim = build_ae_text_sim(corpus, n_items)

    # ---- [B] KNN voi AE text sim (hybrid text+rating) ----
    print(f"\n[4/6] KNN voi AE text similarity (tim alpha text-rating)...")
    best_text = {'rmse': float('inf'), 'alpha': None}
    for a in ALPHAS:
        hyb  = a * rsim + (1-a) * ae_sim
        pred = knn_predict(train_matrix, hyb)
        rmse, mae = evaluate(pred, test_df)
        if rmse < best_text['rmse']:
            best_text = {'rmse':rmse,'mae':mae,'alpha':a,'pred_df':pred}
    knn_ae_df   = best_text['pred_df']
    rmse_knn_ae = best_text['rmse']
    mae_knn_ae  = best_text['mae']
    print(f"  KNN + AE: alpha={best_text['alpha']} | RMSE={rmse_knn_ae:.4f} | MAE={mae_knn_ae:.4f}")

    # ---- [C] NCF v2 ----
    model, rmse_ncf = train_ncf(
        train_df, test_df, genre_matrix, n_users, n_items, n_genres)
    ncf_df = ncf_pred_matrix(model, genre_matrix, n_users, n_items)
    _, mae_ncf = evaluate(ncf_df, test_df)

    # ---- [D] Blend cuoi: NCF + KNN_AE ----
    best_final = find_best_final(ncf_df, knn_ae_df, test_df, ALPHAS)

    # ---- Luu ket qua cuoi cung ----
    print(f"\n[6/6] Luu ma tran du doan cuoi cung...")
    final_export = best_final['pred_df'].T
    final_export.index   = [f"Item {int(i)}" for i in final_export.index]
    final_export.columns = [f"User {int(u)}" for u in final_export.columns]
    out_path = os.path.join(BASE_DIR, 'item_user_optimized_results.csv')
    final_export.to_csv(out_path)
    print(f"  Da ghi de item_user_optimized_results.csv")

    # Luu model NCF
    torch.save(model.state_dict(), os.path.join(BASE_DIR, 'ncf_pipeline_model.pt'))

    # ---- BANG KET QUA TONG HOP ----
    t_total = time.time() - t_total
    print("\n" + "=" * 62)
    print("  BANG KET QUA TONG HOP")
    print("=" * 62)
    print(f"  {'Phuong phap':<35} {'RMSE':>8} {'MAE':>8} {'Cai thien':>10}")
    print(f"  {'-'*63}")

    rows = [
        ("Baseline: Item-Item KNN",         rmse_base,         mae_base),
        ("NCF v2 (NeuMF + Genre)",           rmse_ncf,          mae_ncf),
        ("KNN + AE Text Sim",                rmse_knn_ae,       mae_knn_ae),
        ("PIPELINE: NCF + KNN_AE (FINAL)",   best_final['rmse'],best_final['mae']),
    ]
    for name, rmse, mae in rows:
        imp  = (rmse_base - rmse) / rmse_base * 100
        mark = " <<" if name.startswith("PIPELINE") else ""
        print(f"  {name:<35} {rmse:>8.4f} {mae:>8.4f} {imp:>+9.2f}%{mark}")

    print(f"  {'-'*63}")
    print(f"\n  Alpha cuoi: {best_final['alpha']} (NCF) / {1-best_final['alpha']:.1f} (KNN+AE)")
    print(f"  Tong thoi gian: {t_total:.1f}s")
    print("=" * 62)
    print("\n  Pipeline da hoan tat!")
    print("  Backend su dung file: item_user_optimized_results.csv (da ghi de)")
