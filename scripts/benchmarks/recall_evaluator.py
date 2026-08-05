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
            random_state=None
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

        return np.load(output_path, mmap_mode="r")


class RecallEvaluator:
    """
    Evaluates MultiIVF search performance using recall metrics.
    Supports ground truth generation, approximate search evaluation, and candidate size measurement.
    """
    def __init__(self, multi_ivf, n_clusters:int):
        self.mivf = multi_ivf
        self.n_clusters = n_clusters

    def _get_faiss_flat_index(self, dimension):
        return faiss.IndexFlatIP(dimension)

    def _generate_ground_truth(
        self,
        X_corpus:np.ndarray,
        X_query:np.ndarray,
        topk:int,
        query_batch_size:int=10000,
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

        return best_I
    

    def calc_recalls(
            self, 
            X:np.ndarray, 
            X_query:np.ndarray, 
            cluster_assignments:list, 
            n_probe:int, 
            recall_at_k:int=100
        ) -> tuple[list, list, list]:
        if n_probe > self.n_clusters:
            raise ValueError("n_probe must be smaller than self.n_clusters. self.n_clusters:", self.n_clusters)
        
        search_size_list   = []
        cluster_size_lists = []
        emb_idxs_list_list = []
        recalls = []

        # Reorganize assignments for efficient lookup
        labels_to_idxs_dict = defaultdict(lambda: defaultdict(list))
        for i, assignment in enumerate(cluster_assignments):
            for ensemble_label, cluster_labels in assignment.items():
                for label in cluster_labels:
                    labels_to_idxs_dict[ensemble_label][label].append(i)

        # Calculate ground truth
        ground_truth_idxs = self._generate_ground_truth(X, X_query, recall_at_k)

        # Print progress
        print(f"{'iteration':>9} {'recall@10':>10} {'recall@50':>10} {'recall@100':>11}")
        print("-" * 45)

        for i, query in enumerate(X_query):
            cluster_size_list, emb_idxs_list = [], []

            # Search for cluster labels using the MultiIVF index
            nearest_ensemble_label, nearest_cluster_labels = self.mivf.search(query, n_probe)

            # Collect candidate embedding indices and cluster size information
            candidate_idx_set = set()
            for cluster_label in nearest_cluster_labels:
                corpus_idxs_in_cluster = labels_to_idxs_dict[nearest_ensemble_label][cluster_label]
                if len(corpus_idxs_in_cluster) == 0:
                    continue

                cluster_size_list.append(len(corpus_idxs_in_cluster))
                emb_idxs_list.append(corpus_idxs_in_cluster)
                candidate_idx_set.update(corpus_idxs_in_cluster)

            cluster_size_lists.append(cluster_size_list)
            emb_idxs_list_list.append(emb_idxs_list)

            if candidate_idx_set:
                # Execute exact KNN and retrieve top-K indices
                candidate_idxs = np.array(sorted(candidate_idx_set), dtype=np.int64)
                search_size_list.append(len(candidate_idxs))

                sub_vectors = np.ascontiguousarray(X[candidate_idxs], dtype=np.float32)
                sub_index = self._get_faiss_flat_index(dimension=X.shape[1])
                sub_index.add(sub_vectors)

                k = min(recall_at_k, len(candidate_idxs))
                _, local_idx = sub_index.search(query.reshape(1, -1).astype(np.float32), k)
                top_k_vector_idx = candidate_idxs[local_idx[0]].tolist()
            else:
                search_size_list.append(0)
                top_k_vector_idx = []

            # Calculate recall values
            if len(ground_truth_idxs[i]) == 0:
                recall_at_10  = 0.0
                recall_at_50  = 0.0
                recall_at_100 = 0.0
            else:
                true_idx_set  = set(ground_truth_idxs[i][:10])
                recall_at_10  = len(true_idx_set & set(top_k_vector_idx[:10])) / len(true_idx_set)
                true_idx_set  = set(ground_truth_idxs[i][:50])
                recall_at_50  = len(true_idx_set & set(top_k_vector_idx[:50])) / len(true_idx_set)
                true_idx_set  = set(ground_truth_idxs[i])
                recall_at_100 = len(true_idx_set & set(top_k_vector_idx)) / len(true_idx_set)

            recalls.append([recall_at_10, recall_at_50, recall_at_100])

            if i % 50 == 0:
                # Print progress
                avg_recall_at_10, avg_recall_at_50, avg_recall_at_100 = list(np.mean(recalls, axis=0))
                print(f"{i:>9d} {avg_recall_at_10:>10.4f} {avg_recall_at_50:>10.4f} {avg_recall_at_100:>11.4f}")

        return recalls, cluster_size_lists, search_size_list