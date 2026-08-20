import os, argparse, json, faiss
from pathlib import Path
import numpy as np

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

def str2bool(v):
    v = str(v).lower()
    if v in ("1", "true", "y", "yes"):
        return True
    if v in ("0", "false", "n", "no"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


print(f"""
####################################################################################
# Parse arguments
####################################################################################""")
ap = argparse.ArgumentParser()
ap.add_argument("--input",  required=True, metavar="INPUT_FILE_PATH", help="Path to the input .npy or .npz file.")
ap.add_argument("--output", required=False, metavar="OUTPUT_FILE_PATH", default=None, help="Path where the trained ANN model (.npz) will be saved.")
ap.add_argument("--train_size", required=True, type=int, default=None, metavar="TRAIN_SIZE", help="Number of records to use for training the ANN model.")
ap.add_argument("--query_size", required=False, type=int, default=1000, metavar="QUERY_SIZE", help="Number of records to use to evaluate/benchmark the ANN model.")
ap.add_argument("--n_ensembles", required=False, type=int, default=1, metavar="NUMBER_OF_ENSEMBLES", help="Number of ensembles.")
ap.add_argument("--n_clusters",  required=False, type=int, default=1, metavar="NUMBER_OF_CLUSTERS", help="Number of clusters (k) for ANN model (k-means centroid) computation.")
ap.add_argument("--assign_margin", required=False, type=float, default=0.1, metavar="ASSIGN_MARGIN", help="Float Value of margin used for assigning clusters when indexing a vector.")
ap.add_argument("--n_assignments", required=False, type=int, default=100, metavar="NUMBER_OF_ASSIGNMENTS", help="Number of assigning clusters when indexing a vector.")
ap.add_argument("--use_mean_centering", required=False, metavar="USE_MEAN_CENTERING", type=str2bool, default=True, help="Whether to use mean centering method.")
ap.add_argument("--ensemble_selection_method", required=False, type=str, default=None, metavar="ENSEMBLE_SELECTION_METHOD", choices=["top1","mean","weighted_mean","full_weighted_mean"], help="Selection method within the ensemble")
ap.add_argument("--force_rebuild", required=False, metavar="FORCE_REBUILD", type=str2bool, default=False, help="Whether to rebuild model.")
ap.add_argument("--use_local_lib", required=False, metavar="USE_LOCAL_LIB", type=str2bool, default=False, help="Whether to import MultiIVF from the local src directory instead of the installed package.")
ap.add_argument("--gpu_id",  required=False, type=int, default=None, metavar="GPU_ID", help="GPU parameter for faiss.")

ap.add_argument("--n_probe", required=False, type=int, default=5, metavar="NUMBER_OF_PROBE", help="Number of clusters to probe during a search query.")
ap.add_argument("--target_recall",   required=False, type=float, default=None, metavar="TARGET_RECALL", help="Target recall value to search for via optimal n_probe search. If set, min_probe and max_probe must also be set, and n_probe is ignored.")
ap.add_argument("--recall_ks",       required=False, type=int,   nargs="+",    default=[1, 10, 100], metavar="K", help="List of k values at which to compute recall (e.g. recall@1, recall@10, recall@100).")
ap.add_argument("--target_recall_k", required=False, type=int,   default=10,   metavar="K", help="The k value (must be included in recall_ks) used to evaluate target_recall during optimal n_probe search.")
ap.add_argument("--min_probe",       required=False, type=int,   default=None, metavar="MIN_PROBE", help="Lower bound of n_probe values to search when finding the optimal n_probe for target_recall.")
ap.add_argument("--max_probe",       required=False, type=int,   default=None, metavar="MAX_PROBE", help="Upper bound of n_probe values to search when finding the optimal n_probe for target_recall.")


args = ap.parse_args()
# Common args
input_path, output = args.input, args.output
train_size, query_size = args.train_size, args.query_size
n_ensembles, n_clusters = args.n_ensembles, args.n_clusters
n_assignments, assign_margin = args.n_assignments, args.assign_margin
use_mean_centering = args.use_mean_centering
ensemble_selection_method = args.ensemble_selection_method
force_rebuild = args.force_rebuild
use_local_lib = args.use_local_lib
gpu_id = args.gpu_id

# Value for `recall_evaluator.evaluate`
n_probe = args.n_probe

# Values for `recall_evaluator.find_optimal_n_probe`
target_recall   = args.target_recall
recall_ks       = args.recall_ks
target_recall_k = args.target_recall_k
min_probe, max_probe = args.min_probe, args.max_probe

# hardcoded, adjust if needed
test_size = 0
max_iter = 20
random_state = 112
flat_search_batch_size = 10000
max_points_per_centroid = 100
n_init = 2
n_iters_finish = 0

# check args conditions
if query_size <= 0:
    raise Exception("query_size must be larger than 0. ")

if gpu_id is not None and faiss.get_num_gpus() == 0:
    raise ValueError("GPU is not available.")

if target_recall is not None:
    if target_recall_k not in recall_ks:
        raise ValueError(f"`target_recall_k` ({target_recall_k}) should be in `recall_ks` ({recall_ks}).")
    if min_probe is None or max_probe is None:
        raise ValueError(f"`min_probe` and `max_probe` should be set if `target_recall` is set.")
    if min_probe >= max_probe:
        raise ValueError(f"`max_probe` should be larger than `min_probe`.")


print("Input Arguments:")
print(f"input={input_path}")
print(f"output={output}")
print(f"train_size={train_size}")
print(f"query_size={query_size}")
print(f"n_ensembles={n_ensembles}")
print(f"n_clusters={n_clusters}")
print(f"assign_margin={assign_margin}")
print(f"n_assignments={n_assignments}")
print(f"use_mean_centering={use_mean_centering}")
print(f"ensemble_selection_method={ensemble_selection_method}")
print(f"force_rebuild={force_rebuild}")
print(f"use_local_lib={use_local_lib}")
print(f"gpu_id={gpu_id}")

print(f"n_probe={n_probe}")
print(f"target_recall={target_recall}")
print(f"recall_ks={recall_ks}")
print(f"target_recall_k={target_recall_k}")
print(f"min_probe={min_probe}")
print(f"max_probe={max_probe}")
print("")


input_path = resolve_file_path(input_path, should_exist=True)
print("input_path:", input_path)
if output:
    output_path = resolve_file_path(output)
    print("output_path:", output_path)


print(f"""
####################################################################################
# Prepare data
####################################################################################""")
from recall_evaluator import DatasetLoader
print("Loading dataset...")
dataLoader = DatasetLoader(train_size=train_size, test_size=test_size, query_size=query_size, use_memmap=True, chunk_size=100000, normalize_l2=True, random_state=random_state)
X_train, _, X_query = dataLoader.load(input_path)

print("X_train.shape:", X_train.shape)
print("X_query.shape:", X_query.shape)

print(f"""
####################################################################################
# Prepare Multi IVF instance
####################################################################################""")
if use_local_lib:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent / "../../src"))
    from multi_ivf.multi_ivf import MultiIVF
else:
    from multi_ivf import MultiIVF

mivf_params = {
    "n_clusters": n_clusters,
    "n_ensembles": n_ensembles,
    "max_iter": max_iter,
    "max_points_per_centroid": max_points_per_centroid,
    "flat_search_batch_size": flat_search_batch_size,
    "n_init": n_init,
    "n_iters_finish": n_iters_finish,
    "use_mean_centering": use_mean_centering,
    "gpu_id": gpu_id,
    "random_state": random_state,
    "tqdm_disable": False,
}

if force_rebuild or output is None or not os.path.exists(output_path):
    # Build model
    print(f"""Calculating ANN model. Training params:
- n_clusters={n_clusters}
- n_ensembles={n_ensembles}
- max_iter={max_iter}
- random_state={random_state}
- flat_search_batch_size={flat_search_batch_size}
""")
    mivf = MultiIVF(**mivf_params)
    mivf.train(X_train)

    if output:
        mivf.save(output_path)
        print(f"Saved model data to {output_path}")
else:
    print("Load existing ANN model file. ")
    mivf = MultiIVF.load(output_path, gpu_id=gpu_id)

    # Check if parameters match. 
    for key, value in mivf_params.items():
        if key in ["gpu_id", "tqdm_disable"]:
            continue
        if getattr(mivf, key) != value:
            raise ValueError(f"{key}: expected={value}, actual={getattr(mivf, key)}")

    
# Calculate recall values
from recall_evaluator import RecallEvaluator
recall_eval = RecallEvaluator(
    X_train, 
    X_query, 
    mivf, 
    assign_margin=assign_margin, 
    n_assignments=n_assignments, 
    recall_ks=recall_ks, 
    gpu_id=gpu_id)

if target_recall is not None:
    print(f"""
####################################################################################
# Finding optimal n_probe for targer_recall
# 
# Parameters: 
# - n_assignments:{n_assignments}
# - assign_margin={assign_margin}
# - target_recall={target_recall}
# - recall_ks={recall_ks}
# - target_recall_k={target_recall_k}
# - min_probe={min_probe}
# - max_probe={max_probe}
####################################################################################""")
    
    optimal_n_probe, achieved_recall, mean_candidate_size = recall_eval.find_optimal_n_probe(
            target_recall, 
            target_recall_k,
            min_probe, 
            max_probe,
            ensemble_selection_method)
    final_result = dict(optimal_n_probe=optimal_n_probe, achieved_recall=achieved_recall, mean_candidate_size=mean_candidate_size)
    print(json.dumps(final_result))

else: 
    print(f"""
########################################################f############################
# Calculating recall
# 
# Parameters: 
# - n_assignments:{n_assignments}
# - assign_margin={assign_margin}
# - n_probe={n_probe}
####################################################################################""")
    
    recalls, cluster_size_lists, candidate_size_list = recall_eval.evaluate(
        n_probe, 
        ensemble_selection_method=ensemble_selection_method, 
    )

    print(f"""
####################################################################################
# Summarize and save results
####################################################################################""")
    
    final_results = {
        "train_size": train_size,
        "query_size": query_size,
        "n_ensembles": n_ensembles,
        "n_clusters": n_clusters,
        "assign_margin": assign_margin,
        "n_assignments": n_assignments,
        "n_probe": n_probe
    }
    for k, v in recalls.items():
        final_results[f"mean_{k}"] = v
    flat = [x for sub in cluster_size_lists for x in sub]
    final_results["mean_cluster_size"] = np.mean(flat)
    final_results["mean_cluster_size_std"] = np.std(flat)
    final_results["mean_candidate_size"] = np.mean(candidate_size_list)
    final_results["mean_candidate_size_std"] = np.std(candidate_size_list)

    print("="*100)
    for k, v in final_results.items():
        print(f"{k:<22}: {v}")

    print(json.dumps(final_results, ensure_ascii=False))