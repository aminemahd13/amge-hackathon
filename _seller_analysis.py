import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
print("ROWS:", len(df), "ORDERS:", df['order_id'].nunique())

# ---- NaN check on key seller cols ----
for c in ['seller_id','order_id','price','quantity']:
    print(f"NaN {c}: {df[c].isna().sum()}")

# ---- unique sellers ----
n_sellers = df['seller_id'].nunique()
print("\nUNIQUE SELLERS:", n_sellers)

# CA per line
df['ca'] = df['price'] * df['quantity']
print("NaN ca:", df['ca'].isna().sum())

# ---- per-seller aggregates ----
g = df.groupby('seller_id').agg(
    n_orders=('order_id','nunique'),
    n_lines=('order_id','size'),
    ca=('ca','sum')
).reset_index()

def stats(s, name):
    print(f"\n== {name} ==")
    print(f"  mean={s.mean():.3f} median={s.median():.1f} p90={s.quantile(.90):.1f} p99={s.quantile(.99):.1f} max={s.max():.1f} min={s.min():.1f} sum={s.sum():.1f}")

stats(g['n_orders'], 'ORDERS per seller')
stats(g['n_lines'], 'LINES per seller')
stats(g['ca'], 'CA per seller (BRL)')

# ---- mono-order sellers ----
mono = (g['n_orders']==1).sum()
print(f"\nMONO-ORDER sellers: {mono} ({100*mono/n_sellers:.2f}%)")
mono_line = (g['n_lines']==1).sum()
print(f"MONO-LINE sellers: {mono_line} ({100*mono_line/n_sellers:.2f}%)")

# ---- Gini & Lorenz on CA ----
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2*np.sum(cum)/cum[-1]) / n

gini_ca = gini(g['ca'].values)
gini_orders = gini(g['n_orders'].values)
print(f"\nGINI CA: {gini_ca:.4f}")
print(f"GINI orders: {gini_orders:.4f}")

# ---- top-share of CA by top X% of sellers (ranked by CA) ----
ca_sorted = np.sort(g['ca'].values)[::-1]
total_ca = ca_sorted.sum()
print(f"\nTOTAL CA: {total_ca:.2f} BRL")
for pct in [0.01, 0.05, 0.10, 0.20]:
    k = max(1, int(np.ceil(pct*n_sellers)))
    share = ca_sorted[:k].sum()/total_ca
    print(f"  Top {pct*100:.0f}% ({k} sellers): {share*100:.2f}% of CA")

# ---- Lorenz curve sample points (cumulative seller share -> cumulative CA share) ----
ca_asc = np.sort(g['ca'].values)
cum = np.cumsum(ca_asc)/ca_asc.sum()
seller_frac = np.arange(1,n_sellers+1)/n_sellers
print("\nLORENZ (seller% -> CA%):")
for p in [0.1,0.25,0.5,0.75,0.9]:
    idx = int(p*n_sellers)-1
    print(f"  bottom {p*100:.0f}% sellers hold {cum[idx]*100:.2f}% CA")

# ---- top 10 by orders ----
print("\nTOP10 by ORDERS:")
t = g.sort_values('n_orders', ascending=False).head(10)
for _,r in t.iterrows():
    print(f"  {r['seller_id'][:12]} orders={int(r['n_orders'])} lines={int(r['n_lines'])} ca={r['ca']:.0f}")

# ---- top 10 by CA ----
print("\nTOP10 by CA:")
t = g.sort_values('ca', ascending=False).head(10)
for _,r in t.iterrows():
    print(f"  {r['seller_id'][:12]} ca={r['ca']:.0f} orders={int(r['n_orders'])} lines={int(r['n_lines'])}")

# top1 / top10 absolute share
print(f"\nTop 1 seller CA share: {ca_sorted[0]/total_ca*100:.2f}%")
print(f"Top 10 sellers CA share: {ca_sorted[:10].sum()/total_ca*100:.2f}%")
print(f"Top 100 sellers CA share: {ca_sorted[:100].sum()/total_ca*100:.2f}%")

# distribution buckets of orders per seller
print("\nORDER-COUNT buckets:")
bins=[0,1,2,5,10,50,100,10000]
lab=['1','2','3-5','6-10','11-50','51-100','100+']
gb=pd.cut(g['n_orders'],bins=bins,labels=lab,right=True)
print(gb.value_counts().sort_index())
