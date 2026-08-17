from multi_ivf import MultiIVF
import faiss
from tqdm import tqdm

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
faiss.normalize_L2(X_train)
faiss.normalize_L2(X_query)

#############################
# Build ground truth
#############################

# Prepare index for ground truth
def build_ground_truth(X_train, X_query, top_k):
    dim = X_train.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X_train)

    all_gt_idxs = []
    for query in X_query:
        _, gt_idxs = index.search(query.reshape(1, -1), k=top_k)
        all_gt_idxs.append(gt_idxs)

    return all_gt_idxs

top_k = 10
all_gt_idxs = build_ground_truth(X_train, X_query, top_k=top_k)

#############################
# Train multi ivf model
#############################
# train mIVF
def build_mivf(X_train, n_ensembles, n_clusters, init_centroids_list:list=None, use_mean_centering=True, gpu_id=None, copy=True):
    mivf = MultiIVF(
        n_clusters=n_clusters, 
        n_ensembles=n_ensembles, 
        use_mean_centering=use_mean_centering,
        tqdm_disable=True,
        gpu_id=gpu_id,
        copy=copy
    )

    mivf.train(X_train, init_centroids_list=init_centroids_list)
    return mivf


def search(mivf, assignments, n_probe, all_gt_idxs, ensemble_selection_method="top1"):
    D = X_query.shape[1]
    recalls = []
    candidate_sizes = []

    pbar = tqdm(X_query, total=len(X_query), bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    for i, query in enumerate(pbar):
        pbar.set_description(f"n_probe={n_probe}")

        gt_idxs = all_gt_idxs[i]

        # ANN search
        candidate_ids = set()
        results = mivf.search(query, n_probe=n_probe, ensemble_selection_method=ensemble_selection_method)
        if ensemble_selection_method != "full_weighted_mean":
            results = [results]
        for q_seed, q_labels in results:
            for j, assignment in enumerate(assignments):
                c_labels = assignment.get(q_seed)
                if len(set(q_labels) & set(c_labels)) > 0:
                    candidate_ids.add(j)

        # ranking with KNN
        candidate_ids = np.array(list(candidate_ids))
        X_candidates = X_train[candidate_ids]
        sub_index = faiss.IndexFlatIP(D)
        sub_index.add(X_candidates)
        _, pred_ids = sub_index.search(query.reshape(1, -1), k=top_k)
        
        # Evaluate recall
        recall = len(set(candidate_ids[pred_ids[0]]) & set(gt_idxs[0])) / top_k
        recalls.append(recall)
        candidate_sizes.append(X_candidates.shape[0])

        # update progress
        pbar.set_postfix(Recall=np.mean(recalls))

    return np.mean(recalls), np.std(recalls), np.mean(candidate_sizes), np.std(candidate_sizes)


N = X_train.shape[0]
n_ensembles = 3
n_clusters = int(N**0.5)
n_assignments = 100
all_ensemble_selection_methods = ["top1", "mean", "weighted_mean", "full_weighted_mean"]
centroids = np.load("./scripts/data/sample_cluster_centers_for_verification.npy")

test_conditions = [
    # init centroid test
    {
        "use_mean_centering": True,
        "gpu_id": 0,
        "assign_margin": 0.1,
        "n_probes": [5],
        "expects": [(0.940, 1652)],
        "ensemble_selection_methods": ["top1"],
        "init_centroids_list": [centroids],
    },
    # copy = False & use_mean_centering = ValueError
    {
        "use_mean_centering": True,
        "gpu_id": 0,
        "assign_margin": 0.1,
        "n_probes": [5],
        "expects": [()],
        "ensemble_selection_methods": ["top1"],
        "copy": False
    },
    # combination of use_mean_centering = [True, False] and ensemble_selection_methods
    {
        "use_mean_centering": True,
        "gpu_id": None,
        "assign_margin": 0.1,
        "n_probes": [5, 6, 5, 6],
        "expects": [(0.941, 1609),(0.941, 1846),(0.941, 1609),(0.943, 1383)],
        "ensemble_selection_methods": all_ensemble_selection_methods
    },
    {
        "use_mean_centering": True,
        "gpu_id": 0,
        "assign_margin": 0.1,
        "n_probes": [5, 6, 5, 6],
        "expects": [(0.941, 1609),(0.941, 1846),(0.941, 1609),(0.943, 1383)],
        "init_centroids_list": None,
        "ensemble_selection_methods": all_ensemble_selection_methods
    },
    {
        "use_mean_centering": False,
        "gpu_id": None,
        "assign_margin": 0.02,
        "n_probes": [5, 6, 5, 6],
        "expects": [(0.957, 2057),(0.957, 2319),(0.957, 2057),(0.960, 1700)],
        "ensemble_selection_methods": all_ensemble_selection_methods
    },
    {
        "use_mean_centering": False,
        "gpu_id": 0,
        "assign_margin": 0.02,
        "n_probes": [5, 6, 5, 6],
        "expects": [(0.951, 2046),(0.958, 2314),(0.951, 2046),(0.960, 1710)],
        "ensemble_selection_methods": all_ensemble_selection_methods
    },
]


for condition in test_conditions:
    use_mean_centering = condition["use_mean_centering"]
    gpu_id = condition["gpu_id"]
    assign_margin = condition["assign_margin"]
    n_probes = condition["n_probes"]
    expects = condition["expects"]
    ensemble_selection_methods = condition["ensemble_selection_methods"]
    init_centroids_list = condition.get("init_centroids_list", None)
    copy = condition.get("copy", True)
    should_raise_error = condition.get("should_raise_error", False)

    print(f"""
######################################################################################
# use_mean_centering={use_mean_centering}
# copy={copy}
# gpu_id={gpu_id}
# assign_margin={assign_margin}
# init_centroids_list={init_centroids_list}
######################################################################################""")
    results = []

    error = None
    try:
        mivf = build_mivf(
            X_train, 
            n_ensembles,
            n_clusters, 
            use_mean_centering=use_mean_centering, 
            gpu_id=gpu_id, 
            init_centroids_list=init_centroids_list,
            copy=copy
        )
    except ValueError as ve:
        error = ve

    # `use_mean_centering` should be `False` if `copy` == `False`
    should_raise_error = use_mean_centering and not copy
    if should_raise_error:
        assert isinstance(error, ValueError)
        print(f"Expected ValueError was caught: {error}")
        continue
    else:
        assert error is None

    assignments = mivf.assign(X_train, n_assignments=n_assignments, assign_margin=assign_margin)

    for ensemble_selection_method, n_probe, (ex_recall, ex_cand_size) in zip(ensemble_selection_methods, n_probes, expects):
        print(f"# n_probe={n_probe}, ensemble_selection_method={ensemble_selection_method}")
        recall_mean, recall_std, candidate_size_mean, candidate_size_std = search(mivf, assignments, n_probe, all_gt_idxs=all_gt_idxs, ensemble_selection_method=ensemble_selection_method)

        print(f"=> recall_mean={recall_mean:.03f}, recall_std={recall_std:.03f}, candidate_size_mean={candidate_size_mean:.0f}, candidate_size_std={candidate_size_std:.0f}\n")
        assert np.isclose(recall_mean, ex_recall, atol=1e-2)
        assert np.isclose(candidate_size_mean, ex_cand_size, atol=1e+2)
        results.append((ensemble_selection_method, n_probe, recall_mean, recall_std, candidate_size_mean, candidate_size_std))

    print()
    print("=" * 70)
    print(f"use_mean_centering={use_mean_centering}, gpu_id={gpu_id}, assign_margin={assign_margin}")
    print("=" * 70)

    for method, n_probe, recall, recall_std, candidate, candidate_std in results:
        print(
            f"{method:<20} "
            f"n_probe={n_probe:<2} "
            f"Recall={recall:.3f} "
            f"Candidate={candidate:.0f}"
        )