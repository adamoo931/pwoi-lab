import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_features import build_dataset

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
SEED = 42

df = pd.read_csv(RES / "texture_features.csv")
X = df.drop(columns="category").to_numpy()
y = df["category"].to_numpy()
classes = sorted(set(y))
print(f"Zbiór: {X.shape[0]} wektorów x {X.shape[1]} cech, klasy: {classes}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED)
print(f"Zbiór treningowy: {len(X_tr)}, testowy: {len(X_te)}")

models = {
    "SVM (RBF) + standaryzacja": make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
    "SVM (liniowy) + standaryzacja": make_pipeline(
        StandardScaler(), SVC(kernel="linear", C=1)),
    "k-NN (k=5) + standaryzacja": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM (RBF) bez standaryzacji": SVC(kernel="rbf", C=10, gamma="scale"),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    acc_tr = accuracy_score(y_tr, mdl.predict(X_tr))
    acc_te = accuracy_score(y_te, mdl.predict(X_te))
    cvs = cross_val_score(mdl, X, y, cv=cv)
    results[name] = dict(train_accuracy=round(acc_tr, 4), test_accuracy=round(acc_te, 4),
                         cv5_mean=round(float(cvs.mean()), 4), cv5_std=round(float(cvs.std()), 4))
    print(f"{name:34s} train {acc_tr:.4f} | TEST {acc_te:.4f} | CV5 {cvs.mean():.4f} ± {cvs.std():.4f}")

main = "SVM (RBF) + standaryzacja"
y_pred = models[main].predict(X_te)
cm = confusion_matrix(y_te, y_pred, labels=classes)
print("\nMacierz pomyłek (wiersze: prawdziwe, kolumny: przewidziane), kolejność klas", classes, "\n", cm)
print(classification_report(y_te, y_pred, labels=classes, digits=3))
report = classification_report(y_te, y_pred, labels=classes, digits=4, output_dict=True)

# wpływ liczby poziomów jasności (32 = 5 bitów, 64 = 6 bitów) na dokładność SVM (CV 5-krotna)
levels_cmp = {}
for lv in (32, 64):
    d, _ = build_dataset(HERE / "patches", levels=lv)
    s = cross_val_score(models[main], d.drop(columns="category").to_numpy(), d["category"].to_numpy(), cv=cv)
    levels_cmp[str(lv)] = dict(cv5_mean=round(float(s.mean()), 4), cv5_std=round(float(s.std()), 4))
    print(f"poziomy jasności {lv}: CV5 = {s.mean():.4f} ± {s.std():.4f}")

(RES / "lab3_results.json").write_text(json.dumps(dict(
    n_samples=int(len(X)), n_features=int(X.shape[1]), classes=classes, n_train=int(len(X_tr)), n_test=int(len(X_te)),
    seed=SEED, models=results, main_model=main, confusion_matrix=cm.tolist(), report=report,
    levels_comparison=levels_cmp), indent=2, ensure_ascii=False))

# ------------------------------------------------------------------ rysunki
names_pl = {"cegla": "cegła", "trawa": "trawa", "zwir": "żwir"}
# 1) obrazy źródłowe + przykładowe próbki
fig, axs = plt.subplots(3, 7, figsize=(11, 4.9), gridspec_kw=dict(width_ratios=[2.6, 1, 1, 1, 1, 1, 1]))
rng = np.random.default_rng(SEED)
for i, c in enumerate(classes):
    src = Image.open(next((HERE / "textures" / c).glob("*.png")))
    axs[i, 0].imshow(src, cmap="gray"); axs[i, 0].set_ylabel(names_pl[c], fontsize=11)
    axs[i, 0].set_title("obraz źródłowy 512×512" if i == 0 else "", fontsize=9)
    files = sorted((HERE / "patches" / c).glob("*.png"))
    for j, f in enumerate(rng.choice(files, 6, replace=False)):
        axs[i, j + 1].imshow(Image.open(f), cmap="gray", vmin=0, vmax=255)
        if i == 0 and j == 2:
            axs[i, j + 1].set_title("przykładowe próbki 64×64", fontsize=9)
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(FIG / "lab3_textures_patches.png", dpi=150)
plt.close(fig)

# 2) macierz pomyłek
fig, ax = plt.subplots(figsize=(4.2, 3.7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels([names_pl[c] for c in classes]); ax.set_yticklabels([names_pl[c] for c in classes])
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
ax.set_xlabel("klasa przewidziana"); ax.set_ylabel("klasa rzeczywista")
ax.set_title(f"SVM – zbiór testowy (dokładność {results[main]['test_accuracy']:.1%})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "lab3_confusion_svm.png", dpi=170)
plt.close(fig)

# 3) rzut PCA wektorów cech (po standaryzacji)
pca = PCA(n_components=2, random_state=SEED)
P = pca.fit_transform(StandardScaler().fit_transform(X))
fig, ax = plt.subplots(figsize=(5.4, 4.2))
for c, col in zip(classes, ["#c0392b", "#27ae60", "#2980b9"]):
    m = y == c
    ax.scatter(P[m, 0], P[m, 1], s=22, c=col, alpha=0.8, label=names_pl[c])
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} wariancji)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} wariancji)")
ax.set_title("Wektory cech GLCM (72 cechy) po rzutowaniu PCA", fontsize=10)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "lab3_pca.png", dpi=170)
plt.close(fig)
json.dump(dict(pc_var=pca.explained_variance_ratio_.round(4).tolist()), open(RES / "lab3_pca.json", "w"))
print("Gotowe.")
