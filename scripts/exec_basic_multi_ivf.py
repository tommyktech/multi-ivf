from multi_ivf import MultiIVF

#############################
# Load sample dataset
#############################
import pandas as pd
import numpy as np
df = pd.read_parquet("./examples/data/wiki40b_en_embeddings_30000.parquet")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#############################
# Split dataset
#############################
train_end = len(df) - 100

df_train  = df.iloc[:train_end]
df_query  = df.iloc[train_end:]

X_train  = np.stack(df_train.embedding.values)
X_query  = np.stack(df_query.embedding.values)

#############################
# Train multi ivf model
#############################
# train mIVF
print("Training MultiIVF...")
n_ensembles = 1
N = X_train.shape[0]
n_clusters = int(N / 100)

mivf = MultiIVF(
    n_clusters=n_clusters, 
    n_ensembles=n_ensembles, 
    use_mean_centering=True,
    tqdm_disable=False
)

mivf.train(X_train)

#############################
# Assign mIVF labels
#############################
print("Assigning labels for train dataset...")
assignments = mivf.assign(X_train, n_assignments=100)

#############################
# Evaluate recall
#############################
print("Evaluating MultiIVF model...")

# Prepare index for ground truth
import faiss
from tqdm import tqdm
dim = X_train.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X_train)

# search conditions
top_k = 10
n_probe = 5

recalls = []
pbar = tqdm(df_query["embedding"], total=len(df_query), bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
for query in pbar:
    pbar.set_description(f"n_probe={n_probe}")
    # ground truth
    _, gt_ids = index.search(query.reshape(1, -1), k=top_k)

    # ANN search
    q_seed, q_labels = mivf.search(query, n_probe=n_probe)
    candidate_ids = set()
    for i, assignment in enumerate(assignments):
        c_labels = assignment.get(q_seed)
        if len(set(q_labels) & set(c_labels)) > 0:
            candidate_ids.add(i)

    # ranking with KNN
    candidate_ids = np.array(list(candidate_ids))
    X_candidates = X_train[candidate_ids]
    sub_index = faiss.IndexFlatIP(dim)
    sub_index.add(X_candidates)
    _, pred_ids = sub_index.search(query.reshape(1, -1), k=top_k)
    
    # Evaluate recall
    recall = len(set(candidate_ids[pred_ids[0]]) & set(gt_ids[0])) / top_k
    recalls.append(recall)

    # update progress
    pbar.set_postfix(Recall=sum(recalls) / len(recalls))
    pbar.update(1)
