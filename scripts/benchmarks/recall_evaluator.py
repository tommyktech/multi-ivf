from collections import defaultdict
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

class DatasetLoader:
    """
    Loads embedding datasets and prepares train/test/query splits.
    Supports optional shuffling, L2 normalization, and memory-mapped arrays for large datasets.
    """
    def __init__(
            self, 
            train_size:int, 
            test_size:int=0, 
            query_size:int=0, 
            use_memmap=False, 
            chunk_size=10000, 
            normalize_l2=True, 
            random_state=None,
        ):
        self.train_size = train_size
        self.test_size  = test_size
        self.query_size = query_size
        self.use_memmap = use_memmap
        self.chunk_size = chunk_size
        self.normalize_l2 = normalize_l2
        self.random_state = random_state


    def load(self, input_path: str):
        if input_path.endswith(".npz"):
            npz = np.load(input_path, mmap_mode="r", allow_pickle=False)
            if len(npz.files) != 1:
                raise ValueError(f"NPZ must contain exactly one array, found: {npz.files}")
            X_raw = npz[npz.files[0]]
        else:
            X_raw = np.load(input_path, mmap_mode="r", allow_pickle=False)

        N, _ = X_raw.shape

        if N < 10000:
            raise Exception("Dataset too small. Prepare more than 10,000 embeddings.")
        if self.train_size > N:
            raise ValueError(f"train_size too large. train_size:{self.train_size} N:{N}")
        if self.query_size > N:
            raise ValueError(f"query_size too large. query_size:{self.query_size} N:{N}")
        if self.test_size > N:
            raise ValueError(f"test_size too large. test_size:{self.test_size} N:{N}")
        if self.train_size + self.test_size + self.query_size > N:
            raise ValueError(f"train_size + test_size + query_size too large. train_size:{self.train_size} test_size:{self.test_size} query_size:{self.query_size} N:{N}")

        if self.random_state:
            # shuffle
            rng = np.random.default_rng(self.random_state)
            X_raw_idx = rng.permutation(N)
        else:
            X_raw_idx = np.arange(N)

        query_index = None
        test_index  = None
        if self.query_size:
            query_index = X_raw_idx[:self.query_size]
        if self.test_size:
            test_index  = X_raw_idx[self.query_size:self.query_size+self.test_size]
        train_index = X_raw_idx[self.query_size+self.test_size:self.query_size+self.test_size+self.train_size]

        # Prepare temporary output directory
        base_dir = Path(__file__).resolve().parent
        tmp_output_dir = base_dir / "output"
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        train_mmap_path = tmp_output_dir / "tmp_X_train.npy"
        test_mmap_path  = tmp_output_dir / "tmp_X_test.npy"
        query_mmap_path = tmp_output_dir / "tmp_X_query.npy"

        X_test, X_query = None, None
        if self.use_memmap:
            X_train = self._to_memmap(X_raw, train_index, str(train_mmap_path))
            if test_index is not None:
                X_test  = self._to_memmap(X_raw, test_index, str(test_mmap_path))
            if query_index is not None:
                X_query = self._to_memmap(X_raw, query_index, str(query_mmap_path))
        else:
            X_train = np.asarray(X_raw[train_index], dtype=np.float32)
            if test_index is not None:
                X_test  = np.asarray(X_raw[test_index], dtype=np.float32)
            if query_index is not None:
                X_query = np.asarray(X_raw[query_index], dtype=np.float32)

            if self.normalize_l2:
                faiss.normalize_L2(X_train)
                if X_test is not None:
                    faiss.normalize_L2(X_test)
                if X_query is not None:
                    faiss.normalize_L2(X_query)

        return X_train, X_test, X_query

    
    def _to_memmap(self, X, data_index, output_path):
        _, D = X.shape

        mmap = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(data_index), D))
        for i in range(0, len(data_index), self.chunk_size):
            end = min(i + self.chunk_size, len(data_index))
            idx_chunk = data_index[i:end]
            Xc = X[idx_chunk]
            
            if self.normalize_l2:
                faiss.normalize_L2(Xc)

            mmap[i:end] = Xc

        mmap.flush()
        del mmap

        return np.load(output_path, mmap_mode="r+")


from typing import Literal
class RecallEvaluator:
    """
    Evaluates MultiIVF search performance using recall metrics.
    Supports ground truth generation, approximate search evaluation, and candidate size measurement.
    """
    def __init__(self, 
                 X:np.ndarray, 
                 X_query:np.ndarray, 
                 multi_ivf, 
                 assign_margin:float,
                 n_assignments:int,
                 recall_ks:list[int] = [10, 100],
                 gpu_id:int=None,
                 tqdm_disable:bool=True):
        """
        Args:
            X: Corpus embeddings of shape (n_corpus, d).
            X_query: Query embeddings of shape (n_query, d).
            multi_ivf: MultiIVF index instance used for approximate search and cluster assignment.
            assign_margin: Margin used when assigning corpus points to multiple clusters.
            n_assignments: Number of cluster assignments per corpus point.
            recall_ks: List of k values at which recall is measured.
            gpu_id: GPU device id to use for FAISS indices, or None for CPU.
            tqdm_disable: If True, disables progress bars.
        """
                
        self.X = X 
        self.X_query = X_query
        self.mivf = multi_ivf
        self.recall_ks = recall_ks
        self.gpu_id = gpu_id
        self.gpu_res_ = None

        recall_k_max = max(self.recall_ks)
        self.ground_truth_idxs_, _ = self.exact_knn_batched(X, X_query, recall_k_max, tqdm_disable=tqdm_disable)
        assignments = self.mivf.assign(X, n_assignments=n_assignments, assign_margin=assign_margin)

        print("Restructuring cluster_assignments data ...")
        labels_to_idxs_dict = defaultdict(lambda: defaultdict(list))
        for i, assignment in enumerate(assignments):
            for ensemble_label, cluster_labels in assignment.items():
                sub_dict = labels_to_idxs_dict[ensemble_label]
                for label in cluster_labels:
                    sub_dict[label].append(i)

        self.labels_to_idxs_dict_ = labels_to_idxs_dict


    def _get_faiss_flat_index(self, dimension):
        if self.gpu_id is None:
            return faiss.IndexFlatIP(dimension)

        if self.gpu_res_ is None:
            self.gpu_res_ = faiss.StandardGpuResources()

        cpu_index = faiss.IndexFlatIP(dimension)
        index = faiss.index_cpu_to_gpu(self.gpu_res_, 0, cpu_index)
        return index
    

    def exact_knn_batched(
        self,
        X_corpus:np.ndarray,
        X_query:np.ndarray,
        topk:int,
        query_batch_size:int=1000,
        corpus_batch_size:int=100000,
        tqdm_disable:bool=False
    ) -> np.ndarray:
        n_q, d = X_query.shape
        topk = min(topk, len(X_corpus))

        best_D:np.ndarray = None
        best_I:np.ndarray = None

        for offset in tqdm(
            range(0, len(X_corpus), corpus_batch_size),
            desc="Calculating ground truth data...",
            disable=tqdm_disable
        ):
            corpus_chunk = X_corpus[offset : offset + corpus_batch_size]

            index = self._get_faiss_flat_index(d)
            index.add(corpus_chunk)

            chunk_D = np.empty((n_q, min(topk, len(corpus_chunk))), dtype=np.float32)
            chunk_I = np.empty((n_q, min(topk, len(corpus_chunk))), dtype=np.int64)

            for i in range(0, n_q, query_batch_size):
                j = min(i + query_batch_size, n_q)
                D, I = index.search(X_query[i:j], chunk_D.shape[1])
                chunk_D[i:j] = D
                chunk_I[i:j] = I + offset

            if best_D is None:
                best_D = chunk_D
                best_I = chunk_I
            else:
                merged_D = np.concatenate([best_D, chunk_D], axis=1)
                merged_I = np.concatenate([best_I, chunk_I], axis=1)

                order  = np.argsort(-merged_D, axis=1, kind="stable")[:, :topk]
                best_D = np.take_along_axis(merged_D, order, axis=1)
                best_I = np.take_along_axis(merged_I, order, axis=1)

        return best_I, best_D

    
    def evaluate(
        self, 
        n_probe:int, 
        ensemble_selection_method: Literal[
            "top1",
            "mean",
            "weighted_mean",
            "full_weighted_mean"] = "weighted_mean",
    ) -> tuple[list, list, list]:
        
        """
        Evaluate recall@k and candidate set sizes over all queries for a given n_probe.

        Args:
            n_probe: Number of clusters to probe per query in the MultiIVF search.
            ensemble_selection_method: Strategy for selecting/combining cluster search results across all ensembles.

        Returns:
            A tuple of (final_results, all_cluster_size_list, candidate_size_list):
                final_results: Dict mapping "recall_at_{k}" to the mean recall over all queries, for each k in self.recall_ks.
                all_cluster_size_list: Per-query list of probed cluster sizes.
                candidate_size_list: Per-query size of the deduplicated candidate index set.
        """

        if n_probe > self.mivf.n_clusters:
            raise ValueError("n_probe must be smaller than n_clusters. n_clusters=", self.mivf.n_clusters)

        recall_k_max = max(self.recall_ks)

        candidate_size_list   = []
        all_cluster_size_list = []
        recalls = []

        # Print progress header
        headers = [f"{'iteration':>9}"]
        headers.extend([f"{f'recall@{recall_k}':>{8+len(str(abs(recall_k)))}}" for recall_k in self.recall_ks])
        headers.append(f"{'candidate_size':>14}")
        header_str = " ".join(headers)
        print(header_str)
        print("-" * len(header_str))

        for i, query in enumerate(self.X_query):
            cluster_size_list = []

            # Search for cluster labels using the MultiIVF index
            results = self.mivf.search(query, n_probe, ensemble_selection_method=ensemble_selection_method)
            if ensemble_selection_method != "full_weighted_mean":
                results = [results]

            # Collect candidate embedding indices and cluster size information
            candidate_idx_set = set()
            for nearest_ensemble_label, nearest_cluster_labels in results:
                for cluster_label in nearest_cluster_labels:
                    corpus_idxs_in_cluster = self.labels_to_idxs_dict_[nearest_ensemble_label][cluster_label]
                    if len(corpus_idxs_in_cluster) == 0:
                        continue

                    cluster_size_list.append(len(corpus_idxs_in_cluster))
                    candidate_idx_set.update(corpus_idxs_in_cluster)
            all_cluster_size_list.append(cluster_size_list)
            candidate_size_list.append(len(candidate_idx_set))

            if candidate_idx_set:
                # Execute exact KNN and retrieve top-K indices
                candidate_idxs = np.array(sorted(candidate_idx_set), dtype=np.int64)

                sub_vectors = np.ascontiguousarray(self.X[candidate_idxs], dtype=np.float32)
                query_vec = query.reshape(1, -1).astype(np.float32).copy()
                
                sub_index = self._get_faiss_flat_index(dimension=self.X.shape[1])
                sub_index.add(sub_vectors)

                k = min(recall_k_max, len(candidate_idxs))
                _, local_idx = sub_index.search(query_vec, k)
                top_k_vector_idx = candidate_idxs[local_idx[0]].tolist()
            else:
                top_k_vector_idx = []

            # Calculate recall values
            if len(self.ground_truth_idxs_[i]) == 0:
                recalls.extend([0.0 for _ in self.recall_ks])
            else:
                recall_result = []
                for recall_k in self.recall_ks:
                    true_idx_set  = set(self.ground_truth_idxs_[i][:recall_k])
                    recall_at_k  = len(true_idx_set & set(top_k_vector_idx[:recall_k])) / len(true_idx_set)
                    recall_result.append(recall_at_k)
                recalls.append(recall_result)

            if i % 50 == 0:
                # Print progress
                progress = f"{i:>9d} "
                for recall_k, avg_recall in zip(self.recall_ks, list(np.mean(recalls, axis=0))):
                    progress += f"{avg_recall:>{8+len(str(abs(recall_k)))}.4f} "
                print(progress + f"{np.mean(candidate_size_list):>14.0f}")

        # Print final progress
        progress = f"{i:>9d} "
        for recall_k, avg_recall in zip(self.recall_ks, list(np.mean(recalls, axis=0))):
            progress += f"{avg_recall:>{8+len(str(abs(recall_k)))}.4f} "
        print(progress + f"{np.mean(candidate_size_list):>14.0f}")

        final_results = {}
        for recall_k, avg_recall in zip(self.recall_ks, list(np.mean(recalls, axis=0))):
            final_results[f"recall_at_{recall_k}"] = avg_recall
        return final_results, all_cluster_size_list, candidate_size_list

    
    def find_optimal_n_probe(
        self, 
        target_recall: float, 
        target_recall_k: int,
        min_probe: int = 5, 
        max_probe: int = 30,
        ensemble_selection_method: Literal[
            "top1",
            "mean",
            "weighted_mean",
            "full_weighted_mean"] = "weighted_mean",
    ) -> tuple[int, float, int]:

        """
        Find the minimum n_probe that achieves target_recall using binary search.

        Args:
            target_recall: The recall value to achieve (e.g. 0.9 for 90%).
            target_recall_k: The k value (from self.recall_ks) whose recall is checked against target_recall.
            min_probe: Lower bound of the binary search range for n_probe.
            max_probe: Upper bound of the binary search range for n_probe.
            ensemble_selection_method: Strategy for selecting/combining cluster search results across all ensembles.

        Returns:
            A tuple of (best_n_probe, best_recall, best_mean_candidate_size):
                best_n_probe: The smallest n_probe that achieved target_recall, or None if not found.
                best_recall: The recall_at_{target_recall_k} value achieved at best_n_probe.
                best_mean_candidate_size: The mean candidate set size at best_n_probe.
        """

        if target_recall_k not in self.recall_ks:
            raise ValueError(f"target_recall_k ({target_recall_k}) does not exist in recall_ks ({self.recall_ks}) ")

        low = min_probe
        high = max_probe
        
        best_n_probe = None
        best_recall = None
        best_mean_candidate_size = None

        while low <= high:
            mid = (low + high) // 2
            print("="*15)
            print("n_probe:", mid)
            print("="*15)

            recalls, _, search_size_list = self.evaluate(
                n_probe=mid, 
                ensemble_selection_method=ensemble_selection_method, 
            )
            mean_candidate_size = np.mean(search_size_list)
            target_recall_k_key = f"recall_at_{target_recall_k}"

            if target_recall_k_key not in recalls:
                raise ValueError(f"target_recall_k_key ({target_recall_k_key}) does not exist in recalls ({recalls}).")
            
            mean_recall_at_k = recalls[target_recall_k_key]

            # Check whether the target recall has been achieved
            if mean_recall_at_k >= target_recall:
                # Save as the current best, then search the left side (smaller n_probe) to see if it's still achievable
                best_n_probe = mid
                best_recall = mean_recall_at_k
                best_mean_candidate_size = mean_candidate_size
                high = mid - 1
            else:
                # Target not met, search the right side (larger n_probe)
                low = mid + 1

        return best_n_probe, best_recall, best_mean_candidate_size
