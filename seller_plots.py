"""
Visualisations EDA — CÔTÉ VENDEUR — AMGE / Olist.
Dashboard vendeur : concentration, géographie, performance de livraison,
logistique (fret) et assortiment. Génère des PNG individuels + une planche
combinée dans ./plots_sellers.
Usage:  python seller_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------------- style (cohérent avec make_plots.py)
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.edgecolor": "#cccccc", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#e7e7e7", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
INK = "#1b2a4a"
ACCENT = "#e8743b"
GREEN = "#3a8f5a"
BAND = "#9fb3d1"
OUT = "plots_sellers"
os.makedirs(OUT, exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sep(x, _=None):
    return f"{x:,.0f}".replace(",", " ")


# ----------------------------------------------------------------------------- data + agrégat vendeur
print("Lecture de data/train.csv ...")
df = pd.read_csv("data/train.csv")
df["dist_km"] = haversine(df.customer_lat, df.customer_lng, df.seller_lat, df.seller_lng)
df["line_value"] = df.price * df.quantity
df["freight_per_kg"] = df.freight_value / (df.product_weight_g / 1000 + 1e-3)
df["freight_per_price"] = df.freight_value / (df.price + 1e-3)
df["intra_city"] = (df.customer_city.astype(str) == df.seller_city.astype(str)).astype(int)

# table maître au niveau vendeur
sel = df.groupby("seller_id").agg(
    n_orders=("order_id", "nunique"),
    n_lines=("order_id", "size"),
    n_skus=("product_id", "nunique"),
    n_cats=("product_category_name_english", "nunique"),
    revenue=("line_value", "sum"),
    freight_mean=("freight_value", "mean"),
    fpk_mean=("freight_per_kg", "mean"),
    fpp_mean=("freight_per_price", "mean"),
    price_mean=("price", "mean"),
    dist_mean=("dist_km", "mean"),
    weight_mean=("product_weight_g", "mean"),
    deliv_med=("delivery_time_days", "median"),
    deliv_std=("delivery_time_days", "std"),
    seller_city=("seller_city", "first"),
    seller_lat=("seller_lat", "first"),
    seller_lng=("seller_lng", "first"),
)
N_SELLERS = len(sel)
print(f"{len(df):,} lignes · {df.order_id.nunique():,} commandes · {N_SELLERS:,} vendeurs.")

# ---- agrégats ville & catégorie ----
TOT_ORDERS = int(df.order_id.nunique())
city = df.groupby("seller_city").agg(sellers=("seller_id", "nunique"),
                                     orders=("order_id", "nunique"))
catdf = df.dropna(subset=["product_category_name_english"])
cat = catdf.groupby("product_category_name_english").agg(sellers=("seller_id", "nunique"),
                                                         orders=("order_id", "nunique"))


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    nn = len(x)
    return (2 * np.sum((np.arange(1, nn + 1)) * x)) / (nn * x.sum()) - (nn + 1) / nn


def _box(ax, txt, xy=(0.975, 0.96), va="top", ha="right", fc="white", ec="#cccccc"):
    ax.text(xy[0], xy[1], txt, transform=ax.transAxes, va=va, ha=ha, fontsize=9.2,
            parse_math=False,
            bbox=dict(boxstyle="round,pad=0.45", fc=fc, ec=ec, alpha=0.93))


# ===================================================================== CONCENTRATION
def plot_lorenz(ax):
    rev = np.sort(sel.revenue.values)
    cum = np.insert(np.cumsum(rev) / rev.sum(), 0, 0) * 100
    x = np.insert(np.arange(1, len(rev) + 1) / len(rev), 0, 0) * 100
    g = gini(rev)
    ax.plot([0, 100], [0, 100], color="#888888", ls="--", lw=1.3, label="Égalité parfaite")
    ax.fill_between(x, cum, x, color=BAND, alpha=0.35)
    ax.plot(x, cum, color=INK, lw=2.6, label="Lorenz (CA vendeur)")
    # repères
    for px, py, lab in [(50, np.interp(50, x, cum), "bottom 50 %"),
                        (80, np.interp(80, x, cum), None)]:
        ax.scatter([px], [py], color=ACCENT, zorder=5, s=28)
    _box(ax, f"Indice de Gini = {g:.3f}\nLes 50 % plus petits vendeurs\n= {np.interp(50, x, cum):.1f} % du CA\nTop 20 % = {100 - np.interp(80, x, cum):.0f} % du CA",
         xy=(0.04, 0.96), ha="left")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Part cumulée des vendeurs (%)")
    ax.set_ylabel("Part cumulée du CA (%)")
    ax.set_title("Concentration du CA — courbe de Lorenz")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.92)


def plot_pareto_ca(ax):
    rev = np.sort(sel.revenue.values)[::-1]
    cum = np.cumsum(rev) / rev.sum() * 100
    x = np.arange(1, len(rev) + 1) / len(rev) * 100
    ax.plot(x, cum, color=INK, lw=2.6)
    for p in [1, 5, 10, 20]:
        idx = max(int(np.ceil(p / 100 * len(rev))) - 1, 0)
        val = cum[idx]
        ax.scatter([p], [val], color=ACCENT, zorder=5, s=34)
        ax.annotate(f"{p}% → {val:.0f}%", xy=(p, val), xytext=(p + 6, val - 6),
                    fontsize=9, color="#333333",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel("Top X % des vendeurs (classés par CA décroissant)")
    ax.set_ylabel("Part cumulée du CA (%)")
    ax.set_title("Loi de Pareto du CA vendeur")


def plot_orders_log(ax):
    o = sel.n_orders.values
    bins = np.logspace(0, np.log10(o.max()), 34)
    ax.hist(o, bins=bins, color=INK, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    med, p90, p99 = np.median(o), np.percentile(o, 90), np.percentile(o, 99)
    mono = int((o == 1).sum())
    ax.axvline(med, color=ACCENT, lw=2.2, label=f"Médiane = {med:.0f}")
    ax.axvline(p90, color=GREEN, lw=1.8, ls="--", label=f"P90 = {p90:.0f}")
    ax.axvline(p99, color="#8a4fbf", lw=1.6, ls=":", label=f"P99 = {p99:.0f}")
    _box(ax, f"{mono} vendeurs mono-commande ({mono / len(o) * 100:.0f} %)\nmax = {o.max():.0f} commandes")
    ax.set_xlabel("Nombre de commandes par vendeur (échelle log)")
    ax.set_ylabel("Nombre de vendeurs")
    ax.set_title("Activité des vendeurs — commandes par vendeur")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, bbox_to_anchor=(1.0, 0.80))


# ===================================================================== GÉOGRAPHIE
def _back_to_back(ax, top, labels, title):
    """Barres en effectifs bruts : commandes (demande) et vendeurs (offre),
    chacun sur son propre axe x (échelles différentes, pas de normalisation)."""
    y = np.arange(len(top))
    h = 0.38
    omax, smax = top.orders.max(), top.sellers.max()
    b1 = ax.barh(y + h / 2, top.orders, height=h, color=INK, zorder=3,
                 label="Commandes (demande)")
    ax2 = ax.twiny()
    b2 = ax2.barh(y - h / 2, top.sellers, height=h, color=ACCENT, zorder=3,
                  label="Vendeurs (offre)")
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r.orders + omax * 0.012, i + h / 2,
                f"{int(r.orders):,}".replace(",", " "),
                va="center", ha="left", fontsize=7.6, color=INK)
        ax2.text(r.sellers + smax * 0.012, i - h / 2, f"{int(r.sellers)}",
                 va="center", ha="left", fontsize=7.6, color=ACCENT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlim(0, omax * 1.18)
    ax2.set_xlim(0, smax * 1.22)
    ax.set_xlabel("Nombre de commandes (demande)", color=INK, fontsize=10)
    ax2.set_xlabel("Nombre de vendeurs (offre)", color=ACCENT, fontsize=10)
    ax.tick_params(axis="x", colors=INK, labelsize=8.5)
    ax2.tick_params(axis="x", colors=ACCENT, labelsize=8.5)
    ax2.grid(False)
    ax.set_title(title, pad=30)
    ax.legend([b1, b2], [b1.get_label(), b2.get_label()],
              loc="lower right", fontsize=8.3, framealpha=0.92)


def plot_top_cities(ax):
    top = city.sort_values("orders", ascending=False).head(12).iloc[::-1]
    _back_to_back(ax, top, [c.title() for c in top.index],
                  "Top 12 villes vendeuses — demande vs offre")


def plot_geo(ax):
    s = sel.dropna(subset=["seller_lat", "seller_lng", "dist_mean"]).copy()
    sizes = np.sqrt(s.n_orders) * 4
    sc = ax.scatter(s.seller_lng, s.seller_lat, s=sizes,
                    c=s.dist_mean.clip(upper=1800), cmap="viridis",
                    alpha=0.55, edgecolors="none")
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
    cb.set_label("Distance médiane servie (km)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    # centroïde pondéré par commandes
    cx = np.average(s.seller_lng, weights=s.n_orders)
    cy = np.average(s.seller_lat, weights=s.n_orders)
    ax.scatter([cx], [cy], marker="*", s=320, color=ACCENT, edgecolor="white",
               linewidth=1.2, zorder=6, label="Centre de gravité (pondéré commandes)")
    ax.set_xlim(-62, -34); ax.set_ylim(-33, 1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Carte des vendeurs (taille ∝ √commandes)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)


# ===================================================================== PERFORMANCE
def plot_seller_delay(ax):
    q = sel[sel.n_orders >= 30]
    d = q.deliv_med
    ax.hist(d.clip(upper=20), bins=np.arange(4, 21, 1), color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.5)
    med = d.median()
    ax.axvline(med, color=ACCENT, lw=2.4, label=f"Médiane du parc = {med:.1f} j")
    ax.axvline(15, color="#c0392b", lw=1.8, ls="--", label="Seuil 15 j")
    n_slow = int((d > 15).sum())
    _box(ax, f"Vendeurs ≥ 30 commandes : {len(q)}\n{n_slow} vendeurs ({n_slow / len(q) * 100:.1f} %) > 15 j\nmeilleur {d.min():.1f} j · pire {d.max():.1f} j",
         xy=(0.975, 0.96))
    ax.set_xlabel("Délai de livraison médian du vendeur (jours)")
    ax.set_ylabel("Nombre de vendeurs")
    ax.set_title("Performance de livraison par vendeur (≥ 30 cmd)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, bbox_to_anchor=(1.0, 0.74))


def plot_dist_vs_delay(ax):
    q = sel[sel.n_orders >= 30].dropna(subset=["dist_mean"])
    x, y = q.dist_mean.values, q.deliv_med.values
    xc = np.percentile(x, 99)
    m = x <= xc
    ax.scatter(x[m], y[m], s=np.sqrt(q.n_orders.values[m]) * 3, alpha=0.4,
               color=INK, edgecolors="none")
    # tendance OLS
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), xc, 50)
    ax.plot(xs, a + b * xs, color=ACCENT, lw=2.6, label="Tendance (OLS)")
    sp = pd.Series(x).corr(pd.Series(y), method="spearman")
    _box(ax, f"Spearman ρ = {sp:.2f}\nDistance = le 1er driver du délai\n(8,97 j proche → 11,9 j lointain)",
         xy=(0.04, 0.96), ha="left", fc="#fff6f0", ec=ACCENT)
    ax.set_xlim(0, xc)
    ax.set_xlabel("Distance moyenne servie par le vendeur (km)")
    ax.set_ylabel("Délai médian du vendeur (jours)")
    ax.set_title("Distance servie → délai de livraison")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)


def plot_vol_vs_delay(ax):
    q = sel[sel.n_orders >= 30]
    x, y = q.n_orders.values, q.deliv_med.values
    ax.scatter(x, y, s=22, alpha=0.4, color=INK, edgecolors="none")
    ax.set_xscale("log")
    pe = pd.Series(x).corr(pd.Series(y))
    # médianes par quartile de volume
    qs = pd.qcut(q.n_orders, 4, labels=False, duplicates="drop")
    meds = q.groupby(qs).deliv_med.median()
    cuts = q.groupby(qs).n_orders.median()
    ax.plot(cuts.values, meds.values, color=ACCENT, lw=2.4, marker="s", ms=7,
            mfc="white", mec=ACCENT, label="Médiane par quartile de volume")
    _box(ax, f"Pearson r = {pe:.2f} → AUCUNE relation\nLes gros vendeurs ne livrent\nPAS plus vite (pas d'économie d'échelle)",
         xy=(0.04, 0.96), ha="left")
    ax.set_xlabel("Nombre de commandes du vendeur (log)")
    ax.set_ylabel("Délai médian du vendeur (jours)")
    ax.set_title("Volume du vendeur → délai (effet nul)")
    ax.legend(loc="lower right", fontsize=8.8, framealpha=0.92)


# ===================================================================== LOGISTIQUE
def plot_freight_drivers(ax):
    s = sel.dropna(subset=["dist_mean", "weight_mean", "freight_mean"]).copy()
    s = s[(s.weight_mean > 0)]
    xc = np.percentile(s.dist_mean, 99)
    yc = np.percentile(s.freight_mean, 99)
    m = (s.dist_mean <= xc) & (s.freight_mean <= yc)
    sc = ax.scatter(s.dist_mean[m], s.freight_mean[m], s=14,
                    c=np.log10(s.weight_mean[m]), cmap="plasma", alpha=0.55, edgecolors="none")
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
    cb.set_label("Poids moyen expédié (log₁₀ g)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cw = sel.freight_mean.corr(sel.weight_mean)
    _box(ax, f"Fret piloté surtout par le POIDS (couleur)\ncorr fret~poids = {cw:.2f}\n> corr fret~distance ≈ 0,30",
         xy=(0.975, 0.30), va="top")
    ax.set_xlim(0, xc); ax.set_ylim(0, yc)
    ax.set_xlabel("Distance moyenne servie (km)")
    ax.set_ylabel("Fret moyen du vendeur (R$)")
    ax.set_title("Déterminants du fret par vendeur")


def plot_fret_prix(ax):
    q = sel[sel.n_lines >= 10]
    r = q.fpp_mean.replace([np.inf, -np.inf], np.nan).dropna()
    hi = 1.4
    ax.hist(r.clip(upper=hi), bins=40, range=(0, hi), color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    med, p90 = r.median(), r.quantile(0.90)
    ax.axvline(med, color=ACCENT, lw=2.4, label=f"Médiane = {med:.2f}")
    ax.axvline(p90, color=GREEN, lw=1.8, ls="--", label=f"P90 = {p90:.2f}")
    ax.axvspan(1.0, hi, color="#c0392b", alpha=0.10)
    ax.axvline(1.0, color="#c0392b", lw=1.4, ls=":")
    pct_gt1 = (r > 1).mean() * 100
    _box(ax, f"Fret médian = {med * 100:.0f} % du prix\nFret > prix (non viable) : {pct_gt1:.1f} % des vendeurs",
         xy=(0.975, 0.96))
    ax.set_xlim(0, hi)
    ax.set_xlabel("Ratio fret / prix moyen du vendeur (≥ 10 lignes)")
    ax.set_ylabel("Nombre de vendeurs")
    ax.set_title("Pression du fret sur le prix, par vendeur")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, bbox_to_anchor=(1.0, 0.74))


# ===================================================================== ASSORTIMENT
def plot_n_cats(ax):
    c = sel[sel.n_cats >= 1].copy()
    edges = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 6, "5-6"), (7, 999, "7+")]
    labels = [e[2] for e in edges]
    counts = [int(((c.n_cats >= lo) & (c.n_cats <= hi)).sum()) for lo, hi, _ in edges]
    medca = [float(c.revenue[(c.n_cats >= lo) & (c.n_cats <= hi)].median()) for lo, hi, _ in edges]
    xpos = np.arange(len(labels))
    ax.bar(xpos, counts, width=0.72, color=INK, alpha=0.88, edgecolor="white", linewidth=0.5)
    for xi, yi in zip(xpos, counts):
        ax.text(xi, yi + max(counts) * 0.01, f"{yi}", ha="center", va="bottom",
                fontsize=8.6, color="#333333")
    ax2 = ax.twinx()
    ax2.plot(xpos, medca, color=ACCENT, lw=2.4, marker="o", ms=6, mfc="white",
             mec=ACCENT, mew=1.4)
    ax2.set_ylabel("CA médian du vendeur (R$)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=9)
    ax2.grid(False)
    ax2.annotate("CA médian ↑", xy=(len(labels) - 1, medca[-1]), xytext=(2.4, medca[-1] * 0.7),
                 color=ACCENT, fontsize=9, fontweight="bold")
    share1 = counts[0] / sum(counts) * 100
    _box(ax, f"{share1:.0f} % des vendeurs = 1 seule catégorie\nGénéralistes (≥2 cat) ≈ 42 % des vendeurs\nmais ~75 % du CA total",
         xy=(0.5, 0.96), ha="center")
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_xlabel("Nombre de catégories distinctes par vendeur")
    ax.set_ylabel("Nombre de vendeurs")
    ax.set_title("Assortiment : spécialistes vs généralistes")


def plot_top_cats(ax):
    top = cat.sort_values("orders", ascending=False).head(12).iloc[::-1]
    _back_to_back(ax, top, [c.replace("_", " ") for c in top.index],
                  "Top 12 catégories — demande vs offre")


def plot_seller_price(ax):
    p = sel.price_mean.dropna()
    hi = p.quantile(0.99)
    bins = np.logspace(np.log10(max(p.min(), 1)), np.log10(hi), 45)
    ax.hist(p.clip(upper=hi), bins=bins, color=INK, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    p10, med, p90 = p.quantile(.10), p.median(), p.quantile(.90)
    ax.axvline(p10, color="#8a4fbf", lw=1.8, ls=":", label=f"P10 (low-cost) = R$ {p10:.0f}")
    ax.axvline(med, color=ACCENT, lw=2.4, label=f"Médiane = R$ {med:.0f}")
    ax.axvline(p90, color=GREEN, lw=1.8, ls="--", label=f"P90 (premium) = R$ {p90:.0f}")
    _box(ax, f"Étalement prix P90/P10 = {p90 / p10:.1f}×\nvendeurs premium : CA médian ≫ low-cost",
         xy=(0.975, 0.96))
    ax.set_xlabel("Prix moyen du vendeur (R$, échelle log)")
    ax.set_ylabel("Nombre de vendeurs")
    ax.set_title("Positionnement prix des vendeurs")
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.92, bbox_to_anchor=(1.0, 0.72))


# ----------------------------------------------------------------------------- render
PANELS = [
    ("s01_lorenz_ca.png", plot_lorenz),
    ("s02_pareto_ca.png", plot_pareto_ca),
    ("s03_commandes_par_vendeur.png", plot_orders_log),
    ("s04_top_villes.png", plot_top_cities),
    ("s05_carte_vendeurs.png", plot_geo),
    ("s06_delai_par_vendeur.png", plot_seller_delay),
    ("s07_distance_vs_delai.png", plot_dist_vs_delay),
    ("s08_volume_vs_delai.png", plot_vol_vs_delay),
    ("s09_determinants_fret.png", plot_freight_drivers),
    ("s10_fret_sur_prix.png", plot_fret_prix),
    ("s11_assortiment_categories.png", plot_n_cats),
    ("s12_top_categories.png", plot_top_cats),
    ("s13_positionnement_prix.png", plot_seller_price),
]
for fname, fn in PANELS:
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    fn(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print("écrit :", os.path.join(OUT, fname))

# ---- planche combinée (dashboard vendeur) ----
LAYOUT = [
    [plot_lorenz, plot_pareto_ca, plot_orders_log],
    [plot_top_cities, plot_geo, plot_seller_delay],
    [plot_dist_vs_delay, plot_vol_vs_delay, plot_freight_drivers],
    [plot_fret_prix, plot_n_cats, plot_top_cats],
]
fig, axes = plt.subplots(4, 3, figsize=(23, 22))
for r, row in enumerate(LAYOUT):
    for cidx, fn in enumerate(row):
        fn(axes[r, cidx])
fig.suptitle("AMGE / Olist — DASHBOARD VENDEURS : concentration · géographie · livraison · logistique · assortiment",
             fontsize=18, fontweight="bold", y=1.005)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "s00_dashboard_vendeurs.png"), bbox_inches="tight")
plt.close(fig)
print("écrit :", os.path.join(OUT, "s00_dashboard_vendeurs.png"))
print("Terminé.")
