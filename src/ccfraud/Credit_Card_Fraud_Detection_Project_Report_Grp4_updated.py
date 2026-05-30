# ============================================================
# IMPORT LIBRARIES
# ============================================================
%matplotlib inline

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

#Update your BASE_DIR accordingly | it will differ if you using a windows or mac computer

BASE_DIR = r"C:\Users\yalim\OneDrive - TAFE\AI Skill Set\Group project\updated_project_documentation" #WINDOWS

#BASE_DIR = "/Users/yalim/OneDrive - TAFE/AI Skill Set/Group project/updated_project_documentation" #MAC

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
FINAL_MODELS_DIR = os.path.join(BASE_DIR, "final_models")

MODEL_SAMPLE_SIZE = 200000

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

# ============================================================
# 2. LOAD DATASET & DATA QUALITY CHECKS
# ============================================================

# This dataset (A) represents the full dataset for EDA and class balance understanding. Dataset A = FULL DATA

def load_and_preprocess(file_path, sample_size=None, random_state=42, verbose=True):

    df = pd.read_csv(file_path)

    if verbose:
        print("Original dataset shape:", df.shape)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=random_state
        ).reset_index(drop=True)

        if verbose:
            print("Sampled dataset shape:", df.shape)

    missing_per_column = df.isnull().sum()
    duplicate_rows = df.duplicated().sum()

    if verbose:
        print("\nSummary statistics:")
        print(df.describe())

        print("\nDataset info:")
        df.info()

        print("\nTotal missing:", missing_per_column.sum())
        print("Duplicate rows:", duplicate_rows)

    return df


# Dataset A = FULL DATA is used for loading, data quality checks, EDA and understanding class imbalance.

df_full = load_and_preprocess(
    DATA_PATH,
    sample_size=None,
    random_state=RANDOM_STATE,
    verbose=True
)

print("\nDataset A (Full data) shape:", df_full.shape)

# Add readable class labels
df_full["Class_Label"] = df_full["Class"].map({
    0: "Legitimate",
    1: "Fraud"
})


def print_class_balance(df, dataset_name="Dataset"):
    counts = df["Class"].value_counts()
    pct = df["Class"].value_counts(normalize=True) * 100

    print(f"\n{dataset_name} class distribution:")
    print(counts)

    print(f"\n{dataset_name} class percentages:")
    print(pct.round(4))


print_class_balance(df_full, "Dataset A (Full data)")

print("\nFraud ratio in full dataset:")
print(df_full["Class"].mean())

# ============================================================
# 3. CLASS BALANCE CHECK
# ============================================================

class_counts = df_full["Class_Label"].value_counts()
class_pct = df_full["Class_Label"].value_counts(normalize=True) * 100

print("\nClass distribution by label:")
print(class_counts)

print("\nClass percentages by label:")
print(class_pct.round(3))

# ============================================================
# 4. EDA
# ============================================================

# ============================================================
# Figure 1 - Fraud vs Non-Fraud
# ============================================================
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df_full, x="Class_Label", order=["Legitimate", "Fraud"])
plt.title("Figure 1. Fraud vs Non-Fraud Transactions", fontsize=13, weight="bold")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig1_fraud_vs_nonfraud_improved.png"), dpi=600)
plt.show()

# ============================================================
# Figure 2 - Transaction Amount Distribution
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

xmax_cap = df_full["Amount"].quantile(0.99)

legit_amount = df_full.loc[df_full["Class"] == 0, "Amount"]
legit_amount_plot = legit_amount[legit_amount <= xmax_cap]

sns.histplot(
    legit_amount_plot + 1,
    bins=60,
    stat="count",
    kde=True,
    color="steelblue",
    ax=axes[0]
)

axes[0].set_title("Legitimate Transactions", fontsize=11, weight="bold")
axes[0].set_xlabel("Transaction Amount (+1, log scale)")
axes[0].set_ylabel("Number of Transactions")
axes[0].set_xscale("log")

legit_median = legit_amount.median() + 1
axes[0].axvline(legit_median, color="black", linestyle="--", linewidth=1)
axes[0].text(
    legit_median,
    axes[0].get_ylim()[1] * 0.9,
    f"Median = {legit_amount.median():.2f}",
    rotation=90,
    va="top",
    ha="right",
    fontsize=8
)

fraud_amount = df_full.loc[df_full["Class"] == 1, "Amount"]
fraud_amount_plot = fraud_amount[fraud_amount <= xmax_cap]

sns.histplot(
    fraud_amount_plot + 1,
    bins=60,
    stat="count",
    kde=True,
    color="darkorange",
    ax=axes[1]
)

axes[1].set_title("Fraudulent Transactions", fontsize=11, weight="bold")
axes[1].set_xlabel("Transaction Amount (+1, log scale)")
axes[1].set_ylabel("Number of Transactions")
axes[1].set_xscale("log")

fraud_median = fraud_amount.median() + 1
axes[1].axvline(fraud_median, color="black", linestyle="--", linewidth=1)
axes[1].text(
    fraud_median,
    axes[1].get_ylim()[1] * 0.9,
    f"Median = {fraud_amount.median():.2f}",
    rotation=90,
    va="top",
    ha="right",
    fontsize=8
)

xmin = 1
xmax = xmax_cap + 1
axes[0].set_xlim(xmin, xmax)
axes[1].set_xlim(xmin, xmax)

axes[1].text(
    0.95, 0.95,
    f"Fraud cases: {df_full['Class'].sum()}\nVisual cap: 99th percentile",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    fontsize=9
)

plt.suptitle(
    "Figure 2. Transaction Amount Distribution by Class",
    fontsize=13,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "fig2_trans_amt_by_class.png"),
    dpi=600
)

plt.show()

# ============================================================
# Figure 3 - Transaction Time Distribution
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

sns.histplot(
    df_full[df_full["Class"] == 0]["Time"],
    bins=50,
    stat="count",
    color="steelblue",
    ax=axes[0],
    kde=True
)

axes[0].set_title("Legitimate Transactions", fontsize=11, weight="bold")
axes[0].set_xlabel("Transaction Time (seconds)")
axes[0].set_ylabel("Number of Transactions")

sns.histplot(
    df_full[df_full["Class"] == 1]["Time"],
    bins=50,
    stat="count",
    color="darkorange",
    ax=axes[1],
    kde=True
)

axes[1].set_title("Fraudulent Transactions", fontsize=11, weight="bold")
axes[1].set_xlabel("Transaction Time (seconds)")
axes[1].set_ylabel("Number of Transactions")
axes[1].text(
    0.05, 0.95,
    f"Fraud cases: {df_full['Class'].sum()}",
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment='top'
)

xmin = df_full["Time"].min()
xmax = df_full["Time"].max()

axes[0].set_xlim(xmin, xmax)
axes[1].set_xlim(xmin, xmax)

plt.suptitle(
    "Figure 3. Transaction Time Distribution by Class",
    fontsize=13,
    weight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "fig3_trans_time_side_by_side.png"),
    dpi=600
)

plt.show()

# ============================================================
# Figure 4 - Transaction Amount by Class
# ============================================================
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df_full,
    x="Class_Label",
    y="Amount",
    order=["Legitimate", "Fraud"],
    palette={
        "Legitimate": "steelblue",
        "Fraud": "darkorange"
    }
)

plt.yscale("log")

plt.title(
    "Figure 4. Transaction Amount by Class (Log Scale)",
    fontsize=13,
    weight="bold"
)

plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount (log scale)")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "fig4_trans_amt_by_class_log.png"),
    dpi=600
)

plt.show()

# ============================================================
# Figure 5 - Correlation with Target Class
# ============================================================

corr_with_class = (
    df_full.drop(columns=["Class_Label"])
      .corr(numeric_only=True)["Class"]
      .drop("Class")
      .sort_values()
)

colors = ["darkorange" if val > 0 else "steelblue" for val in corr_with_class]

plt.figure(figsize=(8.5, 10))

ax = corr_with_class.plot(
    kind="barh",
    color=colors,
    edgecolor="black",
    linewidth=0.4
)

plt.title(
    "Figure 5. Feature Correlation",
    fontsize=14,
    weight="bold",
    pad=12
)

plt.xlabel("Correlation with Class", fontsize=11)
plt.ylabel("Features", fontsize=11)

plt.axvline(0, color="black", linewidth=1)
plt.grid(axis="x", linestyle="--", alpha=0.35)
plt.grid(axis="y", visible=False)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for i, v in enumerate(corr_with_class):
    ax.text(
        v + (0.005 if v > 0 else -0.005),
        i,
        f"{v:.3f}",
        va="center",
        ha="left" if v > 0 else "right",
        fontsize=8
    )

plt.text(
    0.98, 0.02,
    "Orange = more associated with Fraud\nBlue = more associated with Legitimate",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.9)
)

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "fig5_corr_with_class.png"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Figure 6 - t-SNE VISUALISATION OF FRAUD VS NON-FRAUD
# ============================================================

tsne_source_df = df_full.copy()

if "Class_Label" not in tsne_source_df.columns:
    tsne_source_df["Class_Label"] = tsne_source_df["Class"].map({
        0: "Legitimate",
        1: "Fraud"
    })

fraud_df = tsne_source_df[tsne_source_df["Class"] == 1].copy()
non_fraud_df = tsne_source_df[tsne_source_df["Class"] == 0].copy()

non_fraud_sample_n = min(5000, len(non_fraud_df))
non_fraud_sample = non_fraud_df.sample(
    n=non_fraud_sample_n,
    random_state=RANDOM_STATE
)

tsne_df = pd.concat([fraud_df, non_fraud_sample], axis=0)
tsne_df = tsne_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nt-SNE input shape:", tsne_df.shape)
print("Fraud cases in t-SNE sample:", (tsne_df["Class"] == 1).sum())
print("Legitimate cases in t-SNE sample:", (tsne_df["Class"] == 0).sum())

exclude_cols = ["Class", "Class_Label"]
feature_cols = [col for col in tsne_df.columns if col not in exclude_cols]

X_tsne = tsne_df[feature_cols]
y_tsne = tsne_df["Class"]
label_tsne = tsne_df["Class_Label"]

scaler_tsne = StandardScaler()
X_tsne_scaled = scaler_tsne.fit_transform(X_tsne)

tsne_model = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=30,
    learning_rate="auto",
    init="pca"
)

X_tsne_2d = tsne_model.fit_transform(X_tsne_scaled)

tsne_plot_df = pd.DataFrame({
    "TSNE_1": X_tsne_2d[:, 0],
    "TSNE_2": X_tsne_2d[:, 1],
    "Class": y_tsne.values,
    "Class_Label": label_tsne.values
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=tsne_plot_df,
    x="TSNE_1",
    y="TSNE_2",
    hue="Class_Label",
    hue_order=["Legitimate", "Fraud"],
    alpha=0.7
)

plt.title(
    "Figure 6. t-SNE Visualisation of Fraud vs Non-Fraud Transactions",
    fontsize=13,
    weight="bold"
)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "fig6_tsne_fraud_visualisation.png"),
    dpi=300
)

plt.show()

# ============================================================
# 5. SAMPLE DATA FOR FASTER RUNTIME
# ============================================================

# Dataset B = 200,000-row modelling subset sampled from full data. 
# This subset is used only for train/test split, SMOTE, scaling and hyperparameter tuning.

X_full = df_full.drop(columns=["Class", "Class_Label"], errors="ignore")
y_full = df_full["Class"]

X_model, _, y_model, _ = train_test_split(
    X_full,
    y_full,
    train_size=MODEL_SAMPLE_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE
)

df_model = X_model.copy()
df_model["Class"] = y_model.values

print("\nDataset B (Modelling subset) shape:", df_model.shape)

print("\nDataset B class distribution:")
print(df_model["Class"].value_counts())

print("\nDataset B class percentages:")
print(df_model["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 6. DEFINE FEATURES AND TARGET
# ============================================================

X = df_model.drop("Class", axis=1)
y = df_model["Class"]

# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# VISUAL VALIDATION OF TRAIN / TEST SPLIT
# ============================================================

split_summary = pd.DataFrame([
    {
        "Dataset": "Dataset B\n(Model)",
        "Legitimate": int((y == 0).sum()),
        "Fraud": int((y == 1).sum())
    },
    {
        "Dataset": "Training\n(70%)",
        "Legitimate": int((y_train == 0).sum()),
        "Fraud": int((y_train == 1).sum())
    },
    {
        "Dataset": "Testing\n(30%)",
        "Legitimate": int((y_test == 0).sum()),
        "Fraud": int((y_test == 1).sum())
    }
])

fraud_pct = [
    y.mean() * 100,
    y_train.mean() * 100,
    y_test.mean() * 100
]

# ============================================================
# Figure 7 and Figure 8
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ------------------------------------------------------------
# Figure 7: Stacked bar plot for total class composition
# ------------------------------------------------------------
axes[0].bar(
    split_summary["Dataset"],
    split_summary["Legitimate"],
    label="Legitimate",
    color="steelblue"
)

axes[0].bar(
    split_summary["Dataset"],
    split_summary["Fraud"],
    bottom=split_summary["Legitimate"],
    label="Fraud",
    color="darkorange"
)

axes[0].set_title(
    "Figure 7. Class Composition Before and After Stratified Split",
    fontsize=12,
    weight="bold"
)
axes[0].set_ylabel("Number of Transactions")
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].legend()

for i, row in split_summary.iterrows():
    total = row["Legitimate"] + row["Fraud"]

    axes[0].text(
        i,
        total + total * 0.01,
        f"Total = {total:,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# ------------------------------------------------------------
# Figure 8: Fraud-only bar plot
# ------------------------------------------------------------
axes[1].bar(
    split_summary["Dataset"],
    split_summary["Fraud"],
    color="darkorange",
    edgecolor="black",
    linewidth=0.6
)

axes[1].set_title(
    "Figure 8. Fraud Cases Across Dataset B, Training and Testing Sets",
    fontsize=12,
    weight="bold"
)
axes[1].set_ylabel("Number of Fraud Transactions")
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

for i, fraud_count in enumerate(split_summary["Fraud"]):
    axes[1].text(
        i,
        fraud_count + max(split_summary["Fraud"]) * 0.02,
        f"{fraud_count:,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.show()


# ============================================================
# SUMMARY TABLE
# ============================================================

split_table = pd.DataFrame([
    {
        "Subset": "Dataset B (Model)",
        "Rows": len(y),
        "Legitimate": int((y == 0).sum()),
        "Fraud": int((y == 1).sum()),
        "Fraud_%": y.mean() * 100
    },
    {
        "Subset": "Training (70%)",
        "Rows": len(y_train),
        "Legitimate": int((y_train == 0).sum()),
        "Fraud": int((y_train == 1).sum()),
        "Fraud_%": y_train.mean() * 100
    },
    {
        "Subset": "Testing (30%)",
        "Rows": len(y_test),
        "Legitimate": int((y_test == 0).sum()),
        "Fraud": int((y_test == 1).sum()),
        "Fraud_%": y_test.mean() * 100
    }
])

print("\nTrain/Test split summary:")
print(split_table.round(4))


# ============================================================
# 8. APPLY SMOTE TO TRAINING DATA ONLY
# ============================================================
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ============================================================
# 9. SCALE DATA FOR SVM, kNN, AND LDA
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================

rf_file = os.path.join(SAVED_MODELS_DIR, f"rf_bestparams_{MODEL_SAMPLE_SIZE}.pkl")
if os.path.exists(rf_file):
    rf_best_params = joblib.load(rf_file)
    print(f"Loaded saved RF model best parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", rf_best_params)
else:
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [100, 150],
            "max_depth": [10, 15],
            "min_samples_split": [2, 5]
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train_smote, y_train_smote)
    rf_best_params = rf_grid.best_params_
    joblib.dump(rf_best_params, rf_file)
    print(f"Best RF Parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", rf_best_params)
    print("Random Forest model saved.")

svm_file = os.path.join(SAVED_MODELS_DIR, f"svm_bestparams_{MODEL_SAMPLE_SIZE}.pkl")
if os.path.exists(svm_file):
    svm_best_params = joblib.load(svm_file)
    print(f"Loaded saved SVM model best parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", svm_best_params)
else:
    svm_grid = GridSearchCV(
        SVC(probability=True, class_weight="balanced"),
        param_grid={
            "C": [0.5, 1, 2],
            "gamma": ["scale", 0.1],
            "kernel": ["rbf"]
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    svm_grid.fit(X_train_scaled, y_train_smote)
    svm_best_params = svm_grid.best_params_
    joblib.dump(svm_best_params, svm_file)
    print(f"Best SVM Parameters: (Sample Size: {MODEL_SAMPLE_SIZE})", svm_best_params)
    print("SVM model saved.")

knn_file = os.path.join(SAVED_MODELS_DIR, f"knn_bestparams_{MODEL_SAMPLE_SIZE}.pkl")
if os.path.exists(knn_file):
    knn_best_params = joblib.load(knn_file)
    print(f"Loaded saved kNN model best parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", knn_best_params)
else:
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    knn_grid.fit(X_train_scaled, y_train_smote)
    knn_best_params = knn_grid.best_params_
    joblib.dump(knn_best_params, knn_file)
    print(f"Best kNN Parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", knn_best_params)
    print("kNN model saved.")

lda_file = os.path.join(SAVED_MODELS_DIR, f"lda_bestparams_{MODEL_SAMPLE_SIZE}.pkl")
if os.path.exists(lda_file):
    lda_best_params = joblib.load(lda_file)
    print(f"Loaded saved LDA model best parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", lda_best_params)
else:
    lda_grid = GridSearchCV(
        LDA(solver="lsqr"),
        param_grid={
            "shrinkage": [None, 0.1, 0.5, 0.9]
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    lda_grid.fit(X_train_scaled, y_train_smote)
    lda_best_params = lda_grid.best_params_
    joblib.dump(lda_best_params, lda_file)
    print(f"Best LDA Parameters (Sample Size: {MODEL_SAMPLE_SIZE}):", lda_best_params)
    print("LDA model saved.")

# ============================================================
# 11. EXTRACT BEST PARAMS
# ============================================================
rf_params = rf_best_params
svm_params = svm_best_params
knn_params = knn_best_params
lda_params = lda_best_params

# ============================================================
# 12. FINAL DATASET (FULL DATA)
# ============================================================

# Dataset C = full dataset for final training/evaluation.
# best parameters are learned on Dataset B, then final models are fitted/evaluated on full data.

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y_full,
    test_size=0.30,
    stratify=y_full,
    random_state=RANDOM_STATE
)

# ============================================================
# 13. SMOTE ON FULL TRAINING
# ============================================================
smote_full = SMOTE(random_state=RANDOM_STATE)
X_train_full_smote, y_train_full_smote = smote_full.fit_resample(
    X_train_full, y_train_full
)

# ============================================================
# 14. SCALING (FOR SVM, kNN, LDA ONLY)
# ============================================================
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full_smote)
X_test_full_scaled = scaler_full.transform(X_test_full)

# ============================================================
# 15. TRAIN & SAVE FINAL MODELS (NO GRIDSEARCH)
# ============================================================

svm_file = os.path.join(FINAL_MODELS_DIR, f"fullDataset_svm_modeled_on_{MODEL_SAMPLE_SIZE}.pkl")
rf_file = os.path.join(FINAL_MODELS_DIR, f"fullDataset_rf_modeled_on_{MODEL_SAMPLE_SIZE}.pkl")
knn_file = os.path.join(FINAL_MODELS_DIR, f"fullDataset_knn_modeled_on_{MODEL_SAMPLE_SIZE}.pkl")
lda_file = os.path.join(FINAL_MODELS_DIR, f"fullDataset_lda_modeled_on_{MODEL_SAMPLE_SIZE}.pkl")

if os.path.exists(svm_file):
    svm_final = joblib.load(svm_file)
    print("Loaded existing SVM final model")
else:
    svm_final = SVC(probability=True, **svm_params)
    svm_final.fit(X_train_full_scaled, y_train_full_smote)
    joblib.dump(svm_final, svm_file)
    print("Trained and saved SVM final model")

if os.path.exists(rf_file):
    rf_final = joblib.load(rf_file)
    print("Loaded existing RF final model")
else:
    rf_final = RandomForestClassifier(random_state=RANDOM_STATE, **rf_params)
    rf_final.fit(X_train_full_smote, y_train_full_smote)
    joblib.dump(rf_final, rf_file)
    print("Trained and saved RF final model")

if os.path.exists(knn_file):
    knn_final = joblib.load(knn_file)
    print("Loaded existing kNN final model")
else:
    knn_final = KNeighborsClassifier(**knn_params)
    knn_final.fit(X_train_full_scaled, y_train_full_smote)
    joblib.dump(knn_final, knn_file)
    print("Trained and saved kNN final model")

if os.path.exists(lda_file):
    lda_final = joblib.load(lda_file)
    print("Loaded existing LDA final model")
else:
    lda_final = LDA(solver="lsqr", **lda_params)
    lda_final.fit(X_train_full_scaled, y_train_full_smote)
    joblib.dump(lda_final, lda_file)
    print("Trained and saved LDA final model")

# ============================================================
# 16. MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(
    name,
    model,
    X_train_data,
    y_train_data,
    X_test_data,
    y_test_data,
    fit_model=True,
    save_plots=True,
    plots_dir=PLOTS_DIR
):
    print(f"\n{'='*70}")
    print(f"RUNNING {name}")
    print(f"{'='*70}")

    if fit_model:
        model.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test_data)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_data)[:, 1]
    else:
        y_prob = model.decision_function(X_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, zero_division=0)
    recall = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_data, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    report_dict = classification_report(
        y_test_data,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    rename_dict = {
        '0': '0 (Non Fraud)',
        '1': '1 (Fraud)',
    }
    report_df = report_df.rename(index=rename_dict)
    print("\nClassification Report:")
    print(report_df.round(3))

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_test_data, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Confusion Matrix - {name}")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test_data, y_prob)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title(f"ROC Curve - {name}")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()
    if save_plots:
        plt.savefig(
            os.path.join(plots_dir, f"{safe_name}_confusion_and_roc_modeled_on_{MODEL_SAMPLE_SIZE}.png"),
            bbox_inches="tight",
            dpi=600
        )
    plt.show()

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    }

### SUPPORT VECTOR MACHINE (SVM) ###

# ============================================================
# 17. EVALUATE SVM (SUPPORT VECTOR MACHINE)
# ============================================================
svm_results = evaluate_model(
    "SVM (Full Dataset)",
    svm_final,
    None, None,
    X_test_full_scaled,
    y_test_full,
    fit_model=False
)

### RANDOM FOREST ###

# ============================================================
# 18. EVALUATE RANDOM FOREST
# ============================================================
rf_results = evaluate_model(
    "Random Forest (Full Dataset)",
    rf_final,
    None, None,
    X_test_full,
    y_test_full,
    fit_model=False
)

### kNN ###

# ============================================================
# 19. EVALUATE kNN
# ============================================================
knn_results = evaluate_model(
    "kNN (Full Dataset)",
    knn_final,
    None, None,
    X_test_full_scaled,
    y_test_full,
    fit_model=False
)

### LINEAR DISCRIMINANT ANALYSIS (LDA) ###

# ============================================================
# 20. EVALUATE LDA
# ============================================================
lda_results = evaluate_model(
    "LDA (Full Dataset)",
    lda_final,
    None, None,
    X_test_full_scaled,
    y_test_full,
    fit_model=False
)

# ============================================================
# 21. FINAL MODEL COMPARISON TABLE
# ============================================================
results = [
    svm_results,
    rf_results,
    knn_results,
    lda_results
]

results_df = pd.DataFrame(results)
results_sorted = results_df.sort_values(by="ROC_AUC", ascending=False)

print("\nSummary of All Models:")
print(results_sorted.to_string(index=False))

# ============================================================
# 22. COMPARISON OF MODEL PERFORMANCE METRICS
# ============================================================

grid_filename = os.path.join(
    PLOTS_DIR,
    f"fig11_model_metric_grid_modeled_on_{MODEL_SAMPLE_SIZE}.png"
)

# Rebuild results table from evaluated models
results_grid = pd.DataFrame([
    svm_results,
    rf_results,
    knn_results,
    lda_results
])

# Clean model names for display
results_grid["Model"] = results_grid["Model"].str.replace(" (Full Dataset)", "", regex=False)

# Metrics to plot
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

# Create grid
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    axes[i].scatter(results_grid["Model"], results_grid[metric], s=90)
    axes[i].set_title(metric.replace("_", " "), fontsize=11, weight="bold")
    axes[i].set_xlabel("Model")
    axes[i].set_ylabel(metric.replace("_", " "))
    axes[i].grid(True, linestyle="--", alpha=0.4)
    axes[i].tick_params(axis="x", rotation=20)

    # Set sensible y-axis limits
    ymin = max(0, results_grid[metric].min() - 0.05)
    ymax = min(1.0, results_grid[metric].max() + 0.05)
    axes[i].set_ylim(ymin, ymax)

    # Add values next to points
    for x, y in zip(results_grid["Model"], results_grid[metric]):
        axes[i].annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8
        )

# Remove the unused 6th subplot
fig.delaxes(axes[5])

fig.suptitle(
    "Figure 9. Grid Comparison of Model Performance Metrics",
    fontsize=14,
    weight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(grid_filename, dpi=600, bbox_inches="tight")
plt.show()

# ============================================================
# 23. DATA FLOW / SAMPLING SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "Stage": "Dataset A - Full data (EDA)",
        "Rows": len(df_full),
        "Fraud_Cases": int(df_full["Class"].sum()),
        "Fraud_Percent": df_full["Class"].mean() * 100
    },
    {
        "Stage": "Dataset B - Modelling subset",
        "Rows": len(df_model),
        "Fraud_Cases": int(df_model["Class"].sum()),
        "Fraud_Percent": df_model["Class"].mean() * 100
    },
    {
        "Stage": "Dataset B - Train split",
        "Rows": len(y_train),
        "Fraud_Cases": int(y_train.sum()),
        "Fraud_Percent": y_train.mean() * 100
    },
    {
        "Stage": "Dataset B - Train after SMOTE",
        "Rows": len(y_train_smote),
        "Fraud_Cases": int(pd.Series(y_train_smote).sum()),
        "Fraud_Percent": pd.Series(y_train_smote).mean() * 100
    },
    {
        "Stage": "Dataset C - Final full train split",
        "Rows": len(y_train_full),
        "Fraud_Cases": int(y_train_full.sum()),
        "Fraud_Percent": y_train_full.mean() * 100
    },
    {
        "Stage": "Dataset C - Final full train after SMOTE",
        "Rows": len(y_train_full_smote),
        "Fraud_Cases": int(pd.Series(y_train_full_smote).sum()),
        "Fraud_Percent": pd.Series(y_train_full_smote).mean() * 100
    }
])

print("\nData flow summary:")
print(summary_df)

# ============================================================
# Figure A1. Row count across workflow stages
# ============================================================

plt.figure(figsize=(12, 6))
bars = plt.bar(summary_df["Stage"], summary_df["Rows"])

plt.title("Figure A1. Number of Rows Across Workflow Stages", fontsize=13, weight="bold")
plt.xlabel("Workflow Stage")
plt.ylabel("Number of Rows")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig9_dataflow_rows_improved.png"), dpi=600)
plt.show()

# ============================================================
# Figure A2. Fraud percentage across workflow stages
# ============================================================

plt.figure(figsize=(12, 6))
bars = plt.bar(summary_df["Stage"], summary_df["Fraud_Percent"])

plt.title("Figure A2. Fraud Percentage Across Workflow Stages", fontsize=13, weight="bold")
plt.xlabel("Workflow Stage")
plt.ylabel("Fraud Percentage (%)")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig10_dataflow_fraud_percent_improved.png"), dpi=600)
plt.show()