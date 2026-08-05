import os, argparse, json
from pathlib import Path
import numpy as np
import faiss

##########################################
# Helper functions
##########################################
def resolve_file_path(path: str, should_exist: bool = None) -> str:
    p = Path(path).expanduser().resolve(strict=False)
    if should_exist is None:
        return str(p)
    
    exists = p.exists()
    if should_exist and not exists:
        raise FileNotFoundError(f"File does not exist: {p}")
    if not should_exist and exists:
        raise FileExistsError(f"File already exists: {p}")
    return str(p)


##########################################
# Parse arguments
##########################################
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, metavar="INPUT_FILE_PATH", help="Path to the input .npy or .npz file.")
ap.add_argument("--output", required=False, metavar="OUTPUT_FILE_PATH", default="firevector_model/centroids_dict.npz", help="Path where the trained ANN model (.npz) will be saved.")
ap.add_argument("--train_size", required=True, type=int, default=None, metavar="TRAIN_SIZE", help="Number of records to use for training the ANN model.")
ap.add_argument("--query_size", required=False, type=int, default=0, metavar="QUERY_SIZE", help="Number of records to use to evaluate/benchmark the ANN model.")
ap.add_argument("--n_ensembles", required=False, type=int, default=1, metavar="NUMBER_OF_ENSEMBLES", help="Number of ensembles.")
ap.add_argument("--n_clusters", required=False, type=int, default=1, metavar="NUMBER_OF_CLUSTERS", help="Number of clusters (k) for ANN model (k-means centroid) computation.")
ap.add_argument("--n_probe", required=False, type=int, default=5, metavar="NUMBER_OF_PROBE", help="Number of clusters to probe during a search query.")

args = ap.parse_args()
input_path = args.input
output = args.output
train_size = args.train_size
query_size = args.query_size
test_size = 0
n_ensembles = args.n_ensembles
n_clusters = args.n_clusters
n_probe = args.n_probe
force_rebuild = False
use_mean_centering = True
n_assignments = 100
assign_margin = 0.1
recall_k = 100
kmeans_max_iter = 20
kmeans_base_seed = 112
kmeans_batch_size = 10000
kmeans_sample_size = 100
kmeans_n_init = 2
kmeans_n_refine_iters = 2

if query_size <= 0:
    raise Exception("query_size must be larger than 0. ")

# gpu parameter for faiss
gpu_id = 0 if faiss.get_num_gpus() > 0 else None

print("Input Arguments:")
print(f"input={input_path}")
print(f"output={output}")
print(f"train_size={train_size}")
print(f"query_size={query_size}")
print(f"n_clusters={n_clusters}")
print(f"assign_margin={assign_margin}")
print(f"n_assignments={n_assignments}")
print(f"n_probe={n_probe}")
print(f"gpu_id={gpu_id}")
print("")

##########################################
# Check parameters
##########################################
input_path = resolve_file_path(input_path, should_exist=True)
output_path = resolve_file_path(output)
print("input_path:", input_path)
print("output_path:", output_path)

print("")

####################################################################################
# preparing data
####################################################################################
from recall_evaluator import DatasetLoader
print("Loading dataset...")
dataLoader = DatasetLoader(train_size=train_size, test_size=test_size, query_size=query_size, use_memmap=True, chunk_size=100000, normalize_l2=True, random_state=kmeans_base_seed)
X_train, _, X_query = dataLoader.load(input_path)

print("X_train.shape:", X_train.shape)
print("X_query.shape:", X_query.shape)
print("")

#####################################################################################
# training k-means
#####################################################################################
from multi_ivf import MultiIVF

mivf_params = {
    "n_clusters": n_clusters,
    "n_ensembles": n_ensembles,
    "max_iter": kmeans_max_iter,
    "max_points_per_centroid": kmeans_sample_size,
    "flat_search_batch_size": kmeans_batch_size,
    "n_init": kmeans_n_init,
    "n_iters_finish": kmeans_n_refine_iters,
    "use_mean_centering": use_mean_centering,
    "gpu_id": gpu_id,
    "random_state": kmeans_base_seed,
    "tqdm_disable": False,
}

if force_rebuild or not os.path.exists(output_path):
    # Build model
    mivf = MultiIVF(**mivf_params)

    print("Training params:")
    print(f"n_clusters:{n_clusters}\nkmeans_n_seeds:{n_ensembles}\nkmeans_max_iter:{kmeans_max_iter}\nkmeans_base_seed:{kmeans_base_seed}\nbatch_size:{kmeans_batch_size}\n")
    print("Calculating ANN model with multiple random seeds...")
    
    mivf.train(X_train)

    print("Saving model data...")
    mivf.save(output_path)
    print(f"Saved model data to {output_path}")
else:
    print("Load existing ANN model file")
    mivf = MultiIVF.load(output_path)

    # Load n_clusters from centroid data
    n_clusters = next(iter(mivf.cluster_centers_.items()))[1].shape[0]
    print("n_clusters loaded from the model file:", n_clusters)

    # Check if parameters match. 
    for key, value in mivf_params.items():
        if getattr(mivf, key) != value:
            raise ValueError(f"{key}: expected={value}, actual={getattr(mivf, key)}")

print("")

####################################################################################
# Recall evaluation
####################################################################################
print("Calculating recall")
print(f"\nParameters used for recall:\nn_assignments={n_assignments}\nassign_margin={assign_margin}\nn_probe={n_probe}\n")
# Assign cluster labels to data
cluster_assignments = mivf.assign(X_train, n_assignments=n_assignments, assign_margin=assign_margin)

# Calculate recall values
from recall_evaluator import RecallEvaluator
recall_eval = RecallEvaluator(mivf, n_clusters=n_clusters)
recalls, cluster_size_lists, candidate_size_list = recall_eval.calc_recalls(X_train, X_query, cluster_assignments, n_probe, recall_at_k=recall_k)

####################################################################################
# Summarize and save results
####################################################################################
def summarize_final_results(recalls):
    mean_recall_at_10, mean_recall_at_50, mean_recall_at_100 = list(np.mean(recalls, axis=0))
    flat = [x for sub in cluster_size_lists for x in sub]

    final_results = {}
    final_results["mean_recall@10"] = mean_recall_at_10
    final_results["mean_recall@50"] = mean_recall_at_50
    final_results["mean_recall@100"] = mean_recall_at_100
    final_results["mean_cluster_size"] = np.mean(flat)
    final_results["mean_cluster_size_std"] = np.std(flat)
    final_results["mean_candidate_size"] = np.mean(candidate_size_list)
    final_results["mean_candidate_size_std"] = np.std(candidate_size_list)

    return final_results


final_results = summarize_final_results(recalls)
eval_results = {
    "train_size": train_size,
    "query_size": query_size,
    "n_ensembles": n_ensembles,
    "n_clusters": n_clusters,
    "assign_margin": assign_margin,
    "n_assignments": n_assignments,
    "n_probe": n_probe,
    "mean_recall@10": f'{final_results["mean_recall@10"]:.3f}',
    "mean_recall@50": f'{final_results["mean_recall@50"]:.3f}',
    "mean_recall@100": f'{final_results["mean_recall@100"]:.3f}',
    "mean_cluster_size": f'{final_results["mean_cluster_size"]:.0f}±{final_results["mean_cluster_size_std"]:.0f}',
    "mean_candidate_size": f'{final_results["mean_candidate_size"]:.0f}±{final_results["mean_candidate_size_std"]:.0f}',
}
print("="*100)
for k, v in eval_results.items():
    print(f"{k:<22}: {v}")

print(json.dumps(eval_results, ensure_ascii=False))