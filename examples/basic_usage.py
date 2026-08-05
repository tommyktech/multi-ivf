from multi_ivf import MultiIVF

#############################
# Helper functions
#############################
import shutil
def print_side_by_side(left_title, left_lines, right_title, right_lines):
    term_width = shutil.get_terminal_size(fallback=(120, 30)).columns

    gap = 4
    col_width = (term_width - gap) // 2

    print("=" * term_width)
    print(f"{left_title:<{col_width}}{' ' * gap}{right_title}")

    n = max(len(left_lines), len(right_lines))
    for i in range(n):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""

        # Shrink wide lines
        left = left[:col_width-1]
        right = right[:col_width-1]

        print(f"{left:<{col_width}}{' ' * gap}{right}")


#############################
# Load sample dataset
#############################
import pandas as pd
import numpy as np
import os
df = pd.read_parquet("./data/wiki40b_en_embeddings_30000.parquet")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


#############################
# Split dataset
#############################
train_end = len(df) - 5 

df_train = df.iloc[:train_end]
df_query = df.iloc[train_end:]
X_train  = np.stack(df_train.embedding.values)
X_query  = np.stack(df_query.embedding.values)


#############################
# Train multi ivf model
#############################
output_path = "output/multi_ivf.joblib"
if os.path.exists(output_path):
    # load mIVF if exists
    print("Loading existing MultiIVF object...")
    mivf = MultiIVF.load(output_path)
else:
    # train mIVF
    print("Training MultiIVF...")
    n_ensembles = 10
    N = X_train.shape[0]
    n_clusters = int(N / 100)

    mivf = MultiIVF(
        n_clusters=n_clusters, 
        n_ensembles=n_ensembles, 
        use_mean_centering=True,
        tqdm_disable=False
        )

    mivf.train(X_train)
    mivf.save(output_path)

#############################
# Assign mIVF labels
#############################
print("Assigning labels for train dataset...")
assignments = mivf.assign(X_train)


#############################
# Search embeddings and Evaluate model
#############################
print("Search embeddings and evaluating MultiIVF model...")

# Prepare index for ground truth
import faiss
dim = X_train.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X_train)

# Search conditions
top_k = 10
n_probe = 5

# Compare search results with ground truth
recalls = []
for i, query in enumerate(X_query):
    # Calculate ground truth
    _, gt_idxs = index.search(query.reshape(1, -1), k=top_k)
    gt_top_k_texts = [row["text"] for _, row in df_train.iloc[gt_idxs[0]].iterrows()]

    # ANN search
    q_ensemble, q_labels = mivf.search(query, n_probe=n_probe)
    candidate_idxs = set()
    for idx, assignment in enumerate(assignments):
        t_labels = assignment.get(q_ensemble)
        if len(q_labels & t_labels) > 0:
            candidate_idxs.add(idx)

    # Ranking with KNN
    candidate_idxs = np.array(list(candidate_idxs))
    X_candidates = X_train[candidate_idxs]
    sub_index = faiss.IndexFlatIP(dim)
    sub_index.add(X_candidates)
    _, search_idxs = sub_index.search(query.reshape(1, -1), k=top_k)
    preds_idxs = list(candidate_idxs[search_idxs[0]])
    preds_top_k_texts = [row["text"] for _, row in df_train.iloc[preds_idxs].iterrows()]

    # Calculate recall value
    recall = len(set(preds_idxs) & set(gt_idxs[0])) / top_k
    recalls.append(recall)

    # Show results
    left_lines, right_lines = [], []
    for j, t in enumerate(gt_top_k_texts):
        left_lines.append(t[:100].replace("\n", " "))
    for j, t in enumerate(preds_top_k_texts):
        right_lines.append(t[:100].replace("\n", " "))
    print_side_by_side(f"Ground Truth (Query {i+1}):", left_lines, f"Search Results (Query {i+1}):", right_lines)

print(f"\nAverage Recall@{top_k}: {sum(recalls) / len(recalls):.04f}")