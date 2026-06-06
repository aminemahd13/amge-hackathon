"""
Visualisations EDA — AMGE / Olist (prédiction du délai de livraison).
Génère 3 figures (+ une planche combinée) dans le dossier ./plots :
  1. Valeur du fret en fonction de la distance vendeur -> client
  2. Distribution du ratio fret / prix
  3. Fréquence du temps d'approbation des commandes
Usage:  python make_plots.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e7e7e7",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
INK = "#1b2a4a"      # accent principal (indigo profond)
ACCENT = "#e8743b"   # accent chaud (orange) pour repères
BAND = "#9fb3d1"     # bande IQR
OUT = "plots"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------------- data
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def brl(x, _=None):
    return f"{x:,.0f}".replace(",", " ")


print("Lecture de data/train.csv ...")
df = pd.read_csv("data/train.csv")
df["dist_km"] = haversine(df.customer_lat, df.customer_lng, df.seller_lat, df.seller_lng)
df["ratio"] = df.freight_value / df.price.replace(0, np.nan)
ap = pd.to_datetime(df.order_approved_at, errors="coerce")
ts = pd.to_datetime(df.order_purchase_timestamp, errors="coerce")
df["appr_h"] = (ap - ts).dt.total_seconds() / 3600.0
n = len(df)

# ---- agrégation au niveau commande (panier = order_id) ----
df["line_value"] = df.price * df.quantity
orders = df.groupby("order_id").agg(
    n_sku=("product_id", "nunique"),
    units=("quantity", "sum"),
    merch=("line_value", "sum"),
    freight=("freight_value", "sum"),
    mean_price=("price", "mean"),
)
orders["total"] = orders.merch + orders.freight

# ---- agrégation au niveau client (commandes distinctes par customer_unique_id) ----
clients = df.groupby("customer_unique_id")["order_id"].nunique()
print(f"{n:,} lignes chargées · {len(orders):,} paniers · {len(clients):,} clients.")


# ============================================================================= PLOT 1
def plot_freight_vs_distance(ax):
    d = df[["dist_km", "freight_value"]].dropna()
    x, y = d.dist_km.values, d.freight_value.values
    xmax = np.quantile(x, 0.99)      # 2 480 km
    ymax = np.quantile(y, 0.99)      # ~82 R$
    m = (x <= xmax) & (y <= ymax)
    xv, yv = x[m], y[m]

    hb = ax.hexbin(xv, yv, gridsize=55, bins="log", cmap="BuPu",
                   mincnt=1, linewidths=0.2, edgecolors="none")
    cb = ax.figure.colorbar(hb, ax=ax, pad=0.015, fraction=0.046)
    cb.set_label("Nombre de commandes (échelle log)", fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)

    # tendance : médiane + bande interquartile sur bins à effectif égal
    nb = 24
    edges = np.unique(np.quantile(x[x <= xmax], np.linspace(0, 1, nb + 1)))
    idx = np.digitize(x, edges[1:-1])
    cx, med, q25, q75 = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 50:
            continue
        cx.append(x[sel].mean())
        med.append(np.median(y[sel]))
        q25.append(np.quantile(y[sel], 0.25))
        q75.append(np.quantile(y[sel], 0.75))
    ax.fill_between(cx, q25, q75, color=BAND, alpha=0.45, lw=0,
                    label="Intervalle interquartile (P25–P75)", zorder=3)
    ax.plot(cx, med, color=ACCENT, lw=2.6, marker="o", ms=4.5,
            mfc="white", mec=ACCENT, mew=1.4, label="Fret médian par tranche", zorder=4)

    r_p = d.dist_km.corr(d.freight_value)
    r_s = d.dist_km.corr(d.freight_value, method="spearman")
    ax.text(0.025, 0.965,
            f"Corrélation distance ↔ fret\nPearson  r = {r_p:.2f}\nSpearman ρ = {r_s:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.92))

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(FuncFormatter(brl))
    ax.set_xlabel("Distance vendeur → client (km, haversine)")
    ax.set_ylabel("Valeur du fret (R$)")
    ax.set_title("Valeur du fret en fonction de la distance")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9.3)


# ============================================================================= PLOT 2
def plot_ratio_freight_price(ax):
    r = df["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    hi = 2.0
    shown = r[r <= hi]
    med, mean = r.median(), r.mean()
    pct_gt1 = (r > 1).mean() * 100

    ax.hist(shown, bins=60, range=(0, hi), color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    ax.axvline(med, color=ACCENT, lw=2.2, ls="-",
               label=f"Médiane = {med:.2f}")
    ax.axvline(mean, color="#3a8f5a", lw=2.0, ls="--",
               label=f"Moyenne = {mean:.2f}")
    ax.axvline(1.0, color="#888888", lw=1.4, ls=":",
               label="Fret = prix (ratio 1)")

    ax.text(0.975, 0.96,
            f"Fret > prix : {pct_gt1:.1f} % des commandes\n"
            f"(au-delà de l'axe, ratio jusqu'à {r.max():.1f})",
            transform=ax.transAxes, va="top", ha="right", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff6f0", ec=ACCENT, alpha=0.92))

    ax.set_xlim(0, hi)
    ax.set_xlabel("Ratio fret / prix  (freight_value / price)")
    ax.set_ylabel("Fréquence (nombre de commandes)")
    ax.set_title("Distribution du ratio fret : prix")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.3,
              bbox_to_anchor=(1.0, 0.83))


# ============================================================================= PLOT 3
def plot_approval_time(ax):
    a = df["appr_h"].dropna()
    a = a[a >= 0]
    hi = 72
    shown = a[a <= hi]
    same_day = (a == 0).mean() * 100
    med = a.median()
    p95 = a.quantile(0.95)
    within48 = (a < 48).mean() * 100

    bins = np.arange(0, hi + 3, 3)
    ax.hist(shown, bins=bins, color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")

    ax.axvline(med, color=ACCENT, lw=2.2,
               label=f"Médiane = {med:.0f} h")
    ax.axvline(p95, color="#3a8f5a", lw=2.0, ls="--",
               label=f"P95 = {p95:.0f} h")

    ax.annotate(f"≈ {same_day:.0f} % approuvées\nquasi immédiatement\n(délai ≈ 0 h)",
                xy=(1.5, ax.get_ylim()[1] * 0.55), xytext=(14, ax.get_ylim()[1] * 0.40),
                fontsize=9.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6),
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff6f0", ec=ACCENT, alpha=0.95))
    ax.text(0.975, 0.10,
            f"{within48:.0f} % approuvées sous 48 h\nQueue jusqu'à {a.max()/24:.0f} j",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.92))

    ax.set_xlim(0, hi)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.set_xlabel("Temps d'approbation (heures)  =  order_approved_at − order_purchase_timestamp")
    ax.set_ylabel("Nombre de commandes (échelle log)")
    ax.set_title("Fréquence du temps d'approbation des commandes")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.3)


# ============================================================================= PLOT 4
def plot_photos_qty(ax):
    s = df["product_photos_qty"].dropna()
    n_nan = df["product_photos_qty"].isna().sum()
    vc = s.value_counts().sort_index()
    x = vc.index.astype(int).values
    y = vc.values
    tot = y.sum()

    bars = ax.bar(x, y, width=0.82, color=INK, alpha=0.88,
                  edgecolor="white", linewidth=0.5, zorder=3)
    # étiquettes de comptage sur les barres principales (>= 1 % du total)
    for xi, yi in zip(x, y):
        if yi >= 0.01 * tot:
            ax.text(xi, yi + tot * 0.006, f"{yi:,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=8.3, color="#333333")

    # courbe de Pareto (% cumulé) sur axe secondaire
    ax2 = ax.twinx()
    cum = np.cumsum(y) / tot * 100
    ax2.plot(x, cum, color=ACCENT, lw=2.2, marker="o", ms=4,
             mfc="white", mec=ACCENT, mew=1.3, zorder=4)
    # repère texte sur la courbe (axe secondaire déjà coloré -> pas de légende)
    ax2.annotate("% cumulé", xy=(4, cum[3]), xytext=(5.4, 70),
                 color=ACCENT, fontsize=9.5, fontweight="bold",
                 ha="left", va="center")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Part cumulée des produits (%)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=9)
    ax2.grid(False)
    ax2.axhline(90, color="#bbbbbb", lw=1.0, ls=":", zorder=1)
    ax2.text(x.max(), 91.5, "90 %", ha="right", va="bottom",
             fontsize=8.2, color="#888888")

    share1 = y[0] / tot * 100
    med = s.median()
    ax.text(0.97, 0.62,
            f"1 photo : {share1:.1f} % des produits\n"
            f"Médiane = {med:.0f} · Moyenne = {s.mean():.2f}\n"
            f"{n_nan:,} valeurs manquantes (non incluses)".replace(",", " "),
            transform=ax.transAxes, va="top", ha="right", fontsize=9.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.93))

    ax.set_xticks(x)
    ax.set_xlim(x.min() - 0.7, x.max() + 0.7)
    ax.set_xlabel("Nombre de photos du produit (product_photos_qty)")
    ax.set_ylabel("Nombre de commandes")
    ax.set_title("Fréquence des quantités de photos par produit")


# ============================================================================= PLOT 5
def plot_produits_par_panier(ax):
    u = orders["n_sku"].astype(int)              # produits distincts (SKU) par commande
    tot = len(u)
    cap = 5
    cats = list(range(1, cap)) + [cap]           # 1..4 puis 5+
    counts = [int((u == k).sum()) for k in range(1, cap)] + [int((u >= cap).sum())]
    labels = [str(k) for k in range(1, cap)] + [f"{cap}+"]
    xpos = np.arange(len(cats))

    ax.bar(xpos, counts, width=0.74, color=INK, alpha=0.88,
           edgecolor="white", linewidth=0.5, zorder=3)
    for xi, yi in zip(xpos, counts):
        if yi > 0:
            ax.text(xi, yi + tot * 0.006, f"{yi:,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=8.6, color="#333333")

    ax2 = ax.twinx()
    cum = np.cumsum(counts) / tot * 100
    ax2.plot(xpos, cum, color=ACCENT, lw=2.2, marker="o", ms=5,
             mfc="white", mec=ACCENT, mew=1.3, zorder=4)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Part cumulée des paniers (%)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=9)
    ax2.grid(False)
    ax2.annotate("% cumulé", xy=(1, cum[1]), xytext=(1.6, 72),
                 color=ACCENT, fontsize=9.5, fontweight="bold", ha="left", va="center")

    share1 = counts[0] / tot * 100
    ax.text(0.97, 0.60,
            f"{share1:.1f} % des paniers = 1 seul produit\n"
            f"Produits distincts / panier : moyenne {u.mean():.2f}, médiane {u.median():.0f}\n"
            f"Maximum observé : {u.max()} produits".replace(",", " "),
            transform=ax.transAxes, va="top", ha="right", fontsize=9.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.93))

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nombre de produits distincts (SKU) dans le panier")
    ax.set_ylabel("Nombre de paniers")
    ax.set_title("Nombre de produits par panier")


# ============================================================================= PLOT 6
def plot_panier_moyen(ax):
    v = orders["merch"]
    mean_m = v.mean()                 # panier moyen (marchandise)
    med_m = v.median()
    mean_t = orders["total"].mean()   # panier moyen frais inclus
    hi = 600                          # ~P97, lisibilité de la forme

    ax.hist(v[v <= hi], bins=60, range=(0, hi), color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    ax.axvline(mean_m, color=ACCENT, lw=2.6,
               label=f"Panier moyen = R$ {mean_m:.0f}")
    ax.axvline(med_m, color="#3a8f5a", lw=2.0, ls="--",
               label=f"Médiane = R$ {med_m:.0f}")

    ax.text(0.975, 0.74,
            f"Panier moyen (marchandise) : R$ {mean_m:.2f}\n"
            f"Panier moyen frais inclus : R$ {mean_t:.2f}\n"
            f"P95 = R$ {v.quantile(.95):.0f} · max = R$ {v.max():,.0f}".replace(",", " "),
            transform=ax.transAxes, va="top", ha="right", fontsize=9.3, parse_math=False,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff6f0", ec=ACCENT, alpha=0.93))

    ax.set_xlim(0, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(brl))
    ax.set_xlabel("Valeur du panier (R$, marchandise = Σ prix × quantité par commande)")
    ax.set_ylabel("Nombre de paniers")
    ax.set_title("Le panier moyen — distribution de la valeur des commandes")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.3)


# ============================================================================= PLOT 7
def plot_orders_par_client(ax):
    c = clients.astype(int)
    tot = len(c)
    cap = 5
    counts = [int((c == k).sum()) for k in range(1, cap)] + [int((c >= cap).sum())]
    labels = [str(k) for k in range(1, cap)] + [f"{cap}+"]
    xpos = np.arange(len(counts))

    ax.bar(xpos, counts, width=0.74, color=INK, alpha=0.88,
           edgecolor="white", linewidth=0.5, zorder=3)
    for xi, yi in zip(xpos, counts):
        if yi > 0:
            ax.text(xi, yi + tot * 0.006, f"{yi:,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=8.6, color="#333333")

    ax2 = ax.twinx()
    cum = np.cumsum(counts) / tot * 100
    ax2.plot(xpos, cum, color=ACCENT, lw=2.2, marker="o", ms=5,
             mfc="white", mec=ACCENT, mew=1.3, zorder=4)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Part cumulée des clients (%)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=9)
    ax2.grid(False)
    ax2.annotate("% cumulé", xy=(1, cum[1]), xytext=(1.6, 72),
                 color=ACCENT, fontsize=9.5, fontweight="bold", ha="left", va="center")

    share1 = counts[0] / tot * 100
    repeat = 100 - share1
    ax.text(0.97, 0.60,
            f"{share1:.1f} % des clients = 1 seule commande\n"
            f"Taux de réachat (≥ 2 commandes) : {repeat:.1f} %\n"
            f"Commandes / client : moyenne {c.mean():.2f} · max {c.max()}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc", alpha=0.93))

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nombre de commandes par client (customer_unique_id)")
    ax.set_ylabel("Nombre de clients")
    ax.set_title("Nombre de commandes par client")


# ============================================================================= PLOT 8
def plot_prix_moyen_commande(ax):
    p = orders["mean_price"]
    mean_p = p.mean()
    med_p = p.median()
    hi = 500                          # ~P97

    ax.hist(p[p <= hi], bins=60, range=(0, hi), color=INK, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    ax.axvline(mean_p, color=ACCENT, lw=2.6,
               label=f"Prix moyen = R$ {mean_p:.0f}")
    ax.axvline(med_p, color="#3a8f5a", lw=2.0, ls="--",
               label=f"Médiane = R$ {med_p:.0f}")

    ax.text(0.975, 0.74,
            f"Prix unitaire moyen / commande : R$ {mean_p:.2f}\n"
            f"Médiane : R$ {med_p:.2f}\n"
            f"P95 = R$ {p.quantile(.95):.0f} · max = R$ {p.max():,.0f}".replace(",", " "),
            transform=ax.transAxes, va="top", ha="right", fontsize=9.3, parse_math=False,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff6f0", ec=ACCENT, alpha=0.93))

    ax.set_xlim(0, hi)
    ax.xaxis.set_major_formatter(FuncFormatter(brl))
    ax.set_xlabel("Prix unitaire moyen par commande (R$, moyenne de price par order)")
    ax.set_ylabel("Nombre de commandes")
    ax.set_title("Panier moyen — prix moyen par commande")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.3)


# ----------------------------------------------------------------------------- render
specs = [
    ("01_fret_vs_distance.png", plot_freight_vs_distance, (8.2, 5.6)),
    ("02_ratio_fret_prix.png", plot_ratio_freight_price, (8.2, 5.2)),
    ("03_temps_approbation.png", plot_approval_time, (8.2, 5.2)),
    ("04_quantite_photos.png", plot_photos_qty, (8.6, 5.2)),
    ("05_produits_par_panier.png", plot_produits_par_panier, (8.6, 5.2)),
    ("06_panier_moyen.png", plot_panier_moyen, (8.2, 5.2)),
    ("07_commandes_par_client.png", plot_orders_par_client, (8.6, 5.2)),
    ("08_prix_moyen_commande.png", plot_prix_moyen_commande, (8.2, 5.2)),
]
for fname, fn, size in specs:
    fig, ax = plt.subplots(figsize=size)
    fn(ax)
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("écrit :", path)

# planche combinée (4 x 2)
fig, axes = plt.subplots(4, 2, figsize=(16, 22))
plot_freight_vs_distance(axes[0, 0])
plot_ratio_freight_price(axes[0, 1])
plot_approval_time(axes[1, 0])
plot_photos_qty(axes[1, 1])
plot_produits_par_panier(axes[2, 0])
plot_panier_moyen(axes[2, 1])
plot_orders_par_client(axes[3, 0])
plot_prix_moyen_commande(axes[3, 1])
fig.suptitle("AMGE / Olist — EDA : fret, prix, délai d'approbation, photos, paniers et clients",
             fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "00_planche_combinee.png"), bbox_inches="tight")
plt.close(fig)
print("écrit :", os.path.join(OUT, "00_planche_combinee.png"))
print("Terminé.")
