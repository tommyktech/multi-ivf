import numpy as np
import random, faiss, joblib
from tqdm import tqdm
from typing import Literal

class MultiIVF:
    """
    Enhanced IVF index with support for multi-cluster, multi-assign, multi-probe search strategies, and mean centering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for each ensemble.
    n_ensembles : int, default=10
        Number of ensembles.
    max_iter : int, default=20
        Maximum number of KMeans iterations.
    max_points_per_centroid : int, default=100
        Number of samples used for Faiss KMeans training.
    flat_search_batch_size : int, default=8192
        Batch size used during faiss flat index search.
    n_init : int, default=1
        Number of KMeans initializations.
    n_iters_finish : int, default=0
        Number of additional exact K-means refinement iterations (via `_exact_kmeans`)
        applied after Faiss K-means training. If 0, this refinement step is skipped.
    tol : float, default=1e-5
        Convergence tolerance used only by the exact K-means refinement step
        (`_exact_kmeans`), i.e. only when n_iters_finish > 0. Faiss itself has
        no tol parameter; this is unrelated to Faiss's KMeans convergence.
    use_mean_centering : bool, default=False
        Whether to apply mean centering to the vectors.
    normalize : bool, default=True
            Whether to L2-normalize input vectors via faiss.normalize_L2 during train/assign/search.
    copy : bool, default=True
        Whether array-mutating operations (validation, normalization, mean centering)
        copy the input first rather than modifying it in place. Must not be False when
        use_mean_centering=True (enforced by the ValueError at the top of __init__).
    gpu_id : int | None, default=None
        GPU device ID. If None, CPU is used.
    max_seed : int, default=99999
        Maximum value used when generating random seeds.
    random_state : int, default=432
        Random seed for reproducibility.
    tqdm_disable : bool, default=True
        Whether to disable tqdm progress bars.
    """
    def __init__(
            self, 
            n_clusters:int, 
            n_ensembles:int=10, 
            max_iter:int=20, 
            max_points_per_centroid:int=256,
            flat_search_batch_size:int=8192,
            n_init:int=1, 
            n_iters_finish:int=0, 
            tol:float=1e-5, 
            use_mean_centering:bool=True, 
            normalize:bool=True,
            copy:bool=True,
            gpu_id:int=None,
            max_seed:int=99999, 
            random_state:int=432, 
            tqdm_disable:bool=True,
        ):
        if copy == False and use_mean_centering == True:
            raise ValueError("copy=False cannot be used together with use_mean_centering=True because mean centering modifies the input data in-place and irreversibly. Please set use_mean_centering=False and provide data that has already been mean-centered.")

        self.n_clusters = n_clusters
        self.n_ensembles = n_ensembles
        self.max_iter = max_iter
        self.max_points_per_centroid = max_points_per_centroid
        self.flat_search_batch_size = flat_search_batch_size
        self.n_init = n_init
        self.n_iters_finish = n_iters_finish
        self.tol = tol
        self.use_mean_centering = use_mean_centering
        self.max_seed = max_seed
        self.gpu_id = gpu_id
        self.random_state = random_state
        self.tqdm_disable = tqdm_disable
        self.copy = copy
        self.normalize = normalize
        # parameters filled after `train`
        self.cluster_centers_ = None
        self.gpu_res_ = None
        self.mean_centers_ = None

        if not self.is_gpu_available():
            self.gpu_id = None


    ########################################
    # save and load
    ########################################
    def save(self, path:str, level:int=3):
        joblib.dump(self, path, compress=("gzip", level))


    @classmethod
    def load(cls, path:str, gpu_id:int=None):
        obj = joblib.load(path)
        if gpu_id is not None and cls.is_gpu_available():
            obj.gpu_id = gpu_id
        else:
            obj.gpu_id = None
        return obj


    def __getstate__(self):
        # Exclude environmet dependent variables from joblib dump
        state = self.__dict__.copy()
        state.pop("gpu_id", None)
        state.pop("gpu_res", None)
        return state


    def __setstate__(self, state: dict):
        attr_renames = {
            "mean": "mean_centers_",
            "n_seeds": "n_ensembles",
            "sample_size": "max_points_per_centroid",
            "batch_size": "flat_search_batch_size",
        }
        for old_name, new_name in attr_renames.items():
            if old_name in state:
                state[new_name] = state.pop(old_name)

        self.__dict__.update(state)

        self.gpu_id = None
        self.gpu_res_ = None


    ########################################
    # helpers
    ########################################
    def _iter_batches(self, X: np.ndarray, batch_size: int):
        n = X.shape[0]
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            yield start, X[start:end]


    def _generate_faiss_flat_index(self, dim:int):
        if self.gpu_id is None:
            return faiss.IndexFlatIP(dim)

        if self.gpu_res_ is None:
            self.gpu_res_ = faiss.StandardGpuResources()

        cpu_index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(self.gpu_res_, 0, cpu_index)
        return index


    @classmethod
    def is_gpu_available(cls):
        try:
            n_gpus = faiss.get_num_gpus()
            gpu_available = n_gpus > 0
        except AttributeError:
            gpu_available = False

        return gpu_available


    def _validate_input_array(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be an instance of ndarray.")

        if self.copy:
            return np.ascontiguousarray(X, dtype=np.float32)

        if X.dtype != np.float32 or not X.flags.c_contiguous:
            raise ValueError("X must be of dtype float32 and contiguous.")

        return X

    
    def _exec_mean_centering(self, X: np.ndarray) -> np.ndarray:
        if self.mean_centers_ is None:
            return X
        
        if self.copy:
            X = X - self.mean_centers_
        else:
            X -= self.mean_centers_

        X = self._exec_normalize(X)

        return X
    

    def _exec_normalize(self, X: np.ndarray) -> np.ndarray:
        if self.copy:
            X = X.copy()

        faiss.normalize_L2(X)

        return X


    def is_trained(self):
        if self.cluster_centers_ is None:
            return False
        
        return True
        

    ########################################
    # kmeans trainer functions
    ########################################
    def _faiss_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_points_per_centroid: int,
        seed: int,
        max_iter: int = 10,
        nredo: int = 1,
        init_centroids:np.ndarray=None
    ) -> np.ndarray:
        N, D = X.shape
        if N == 0:
            raise ValueError("X must not be an empty data. ")

        kmeans = faiss.Kmeans(
            D,
            n_clusters,
            niter=max_iter,
            nredo=nredo if init_centroids is None else 1,
            spherical=True,
            verbose=False,
            seed=seed,
            max_points_per_centroid=max_points_per_centroid,
            gpu=False if self.gpu_id is None else True,
        )
        kmeans.train(X, init_centroids=init_centroids)
        centroids = np.asarray(kmeans.centroids, dtype=np.float32, order="C")

        return centroids


    def _exact_kmeans(
        self,
        X: np.ndarray,
        centroids: np.ndarray | None,
        batch_size: int = 8192,
        n_iters: int = 3,
        tol: float = 1e-4,
        seed: int = 0,
    ):
        if n_iters <= 0:
            raise ValueError("n_iters must be larger than 0.")
        
        rng = np.random.default_rng(seed)

        if centroids is None:
            if self.n_clusters is None:
                raise ValueError("n_clusters is required when centroids is None")
            idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            cent = np.asarray(X[idx], dtype=np.float32, order="C")
        else:
            cent = np.asarray(centroids, dtype=np.float32, order="C")

        if self.normalize:
            faiss.normalize_L2(cent)

        K, D = cent.shape
        index = self._generate_faiss_flat_index(D)

        for _ in tqdm(range(n_iters), desc=f"iter 0/{n_iters}", disable=self.tqdm_disable):
            sums = np.zeros((K, D), dtype=np.float64)
            counts = np.zeros((K,), dtype=np.int64)
            index.reset()
            index.add(cent)

            all_labels = []
            for _, X_chunk in self._iter_batches(X, batch_size):
                _, labels = index.search(X_chunk, 1)
                labels = labels.reshape(-1)
                all_labels.append(labels)

                np.add.at(counts, labels, 1)
                np.add.at(sums, labels, X_chunk.astype(np.float64))

            new_cent = cent.copy()
            nonzero = counts > 0
            if np.any(nonzero):
                new_cent[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)

            if np.any(~nonzero):
                m = int((~nonzero).sum())
                repl = rng.integers(0, cent.shape[0], size=m)
                new_cent[~nonzero] = cent[repl]

            if self.normalize:
                faiss.normalize_L2(new_cent)

            shift = np.max(np.linalg.norm(cent - new_cent, axis=1))
            cent = new_cent
            if shift < tol:
                break

        all_labels = np.concatenate(all_labels)
        return cent, all_labels


    def train(self, X:np.ndarray, init_centroids_list:list=None):
        X = self._validate_input_array(X)

        if self.normalize:
            X = self._exec_normalize(X)

        if self.use_mean_centering:
            self.mean_centers_ = X.mean(axis=0).astype(np.float32)
            X = self._exec_mean_centering(X)

        cluster_centers_ = {}
        base_rng = np.random.RandomState(self.random_state)
        for ensemble_seed in tqdm(base_rng.randint(0, self.max_seed, size=self.n_ensembles), desc="Calculating K-means centroids...", disable=self.tqdm_disable):
            ensemble_seed = int(ensemble_seed)
            random.seed(ensemble_seed)
            np.random.seed(ensemble_seed)

            init_centroids = None
            if init_centroids_list is not None:
                centroids_idx = min(len(cluster_centers_), len(init_centroids_list) - 1)
                init_centroids = init_centroids_list[centroids_idx]

            centroids = self._faiss_kmeans(
                X=X,
                n_clusters=self.n_clusters,
                max_points_per_centroid=self.max_points_per_centroid,
                seed=ensemble_seed,
                max_iter=self.max_iter,
                nredo=self.n_init,
                init_centroids=init_centroids
            )
        
            if self.n_iters_finish:
                print(f"Refining {self.n_iters_finish} times")
                centroids, _ = self._exact_kmeans(
                    X=X,
                    centroids=centroids,
                    batch_size=self.flat_search_batch_size,
                    n_iters=self.n_iters_finish,
                    tol=self.tol,
                    seed=ensemble_seed,
                )

            cluster_centers_[ensemble_seed] = centroids

        self.cluster_centers_ = cluster_centers_


    def _assign_once(self, X:np.ndarray, centroids:np.ndarray, assign_margin: float | None, n_assignments: int | None) -> list[list[int]]:
        if centroids.dtype != np.float32:
            raise ValueError("centroids.dtype must be np.float32")
        if X.dtype != np.float32:
            raise ValueError("X.dtype must be np.float32")

        N, D = X.shape
        index = self._generate_faiss_flat_index(D)
        index.add(centroids)

        labels = np.empty(N, dtype=object)

        # prepare topk variable for `index.search`
        search_topk = self.n_clusters
        if n_assignments is not None and n_assignments < self.n_clusters:
            # Limit the number of labels to n_assignments
            search_topk = n_assignments

        for i, X_chunk in self._iter_batches(X, self.flat_search_batch_size):
            if self.use_mean_centering:
                X_chunk = self._exec_mean_centering(X_chunk)

            d_chunk, l_chunk = index.search(X_chunk, search_topk)

            for r in range(len(l_chunk)):
                labels_r = l_chunk[r]
                sims_r = d_chunk[r]
                if assign_margin is not None:
                    keep = sims_r >= (sims_r[0] - assign_margin)
                    labels_r = labels_r[keep]

                labels[i + r] = labels_r

        return list(labels)
    

    def assign(self, X:np.ndarray, assign_margin:float=0.1, n_assignments:int=None) -> list[dict[int, list[int]]]:
        """
        Assign cluster labels for each embedding data
        """
        X = self._validate_input_array(X)
        if self.normalize:
            X = self._exec_normalize(X)

        if n_assignments is not None and n_assignments > self.n_clusters:
            raise ValueError("n_assignments > n_clusters")
        if n_assignments is not None and n_assignments < 1:
            raise ValueError("n_assignments < 1")

        N = X.shape[0]
        assignments = np.empty(N, dtype=object)
        for i in range(N):
            assignments[i] = {}

        for ensemble_label, centroids in tqdm(
            self.cluster_centers_.items(),
            desc="Assigning vectors to clusters...",
            disable=self.tqdm_disable,
        ):
            cluster_ids_per_vector  = self._assign_once(X, centroids, assign_margin, n_assignments)
            for i, cluster_ids in enumerate(cluster_ids_per_vector):
                assignments[i][ensemble_label] = cluster_ids

        return assignments


    def _choose_ensemble_by_mean_distance(self, query:np.ndarray, n_probe:int, use_weighted_mean=False) -> tuple[int, np.ndarray]:
        """
        Select the seed (cluster group) whose top-`n_probe` centroids have the
        highest mean similarity to the query, and return that seed along with
        the similarity scores of all its centroids.
        """
        best_score = None
        best_ensemble_label = None
        best_sims = None

        for ensemble_label, centroids in self.cluster_centers_.items():
            sims = centroids.dot(query)
            k = min(n_probe, centroids.shape[0])

            if k <= 0:
                raise ValueError("n_probe must be larger than 0")
            
            idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = idx[np.argsort(-sims[idx])]
            top_sims = sims[top_idx]

            if use_weighted_mean:
                # rank = 1, 2, ..., k
                weights = 1.0 / np.arange(1, k + 1)
                score = float(np.average(top_sims, weights=weights))
            else:
                score = float(top_sims.mean())
                
            if best_score is None or score > best_score:
                best_score = score
                best_ensemble_label = ensemble_label
                best_sims = sims

        return best_ensemble_label, best_sims


    def _choose_ensembles_by_mean_distance(
        self, query:np.ndarray, n_probe:int
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """
        Select the top-`n_probe` centroids across all ensembles based on
        rank-weighted similarity, and return their ensemble/centroid labels
        along with their similarity scores.
        """
        if n_probe <= 0:
            raise ValueError("n_probe must be larger than 0")

        candidates = []

        for ensemble_label, centroids in self.cluster_centers_.items():
            sims = centroids.dot(query)

            k = min(n_probe, centroids.shape[0])
            if k <= 0:
                continue

            # Select the top-k within this ensemble
            idx = np.argpartition(-sims, k - 1)[:k]

            # In descending order of similarity
            top_idx = idx[np.argsort(-sims[idx])]
            top_sims = sims[top_idx]

            # rank = 1, 2, ..., k
            weights = 1.0 / np.arange(1, k + 1)

            # Assign a rank-weighted score to each centroid
            weighted_scores = top_sims * weights

            for centroid_idx, score, sim in zip(
                top_idx, weighted_scores, top_sims
            ):
                candidates.append(
                    (float(score), ensemble_label, int(centroid_idx), float(sim))
                )

        # Select across all ensembles in descending order of weighted score
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:n_probe]

        labels = {}
        for _, ensemble_label, centroid_idx, _ in selected:
            if ensemble_label in labels:
                labels[ensemble_label].add(centroid_idx)
            else:
                labels[ensemble_label] = set([centroid_idx])
        labels = tuple(labels.items())

        similarities = np.array(
            [sim for _, _, _, sim in selected],
            dtype=np.float32,
        )

        return labels, similarities

    
    def search(self, 
               query:np.ndarray,
               n_probe: int = 1, 
               ensemble_selection_method: Literal[
                   "mean",
                   "weighted_mean",
                   "full_weighted_mean",
               ] | None = "weighted_mean",
    ) -> tuple[int, list[int]] | list[tuple[int, list[int]]]:
        """
        Find the best-matching ensemble label for `query` and return the indices of its
        top-`n_probe` nearest centroid labels within that ensemble label.

        Returns:
            A tuple of (ensemble label, set of centroid labels).
        """

        if not (query.ndim == 1 or (query.ndim == 2 and query.shape[1] == 1)):
            raise ValueError("")
        
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        n_probe = min(n_probe, self.n_clusters)

        if self.use_mean_centering:
            query = self._exec_mean_centering(query)
        if self.normalize:
            faiss.normalize_L2(query)

        query = query.reshape(-1)

        if ensemble_selection_method == "full_weighted_mean":
            best_ensemble_labels, _ = self._choose_ensembles_by_mean_distance(query, n_probe)
            return best_ensemble_labels
        elif ensemble_selection_method == "weighted_mean":
            best_ensemble_labels, best_sims = self._choose_ensemble_by_mean_distance(query, n_probe, use_weighted_mean=True)
        elif ensemble_selection_method == "mean":
            best_ensemble_labels, best_sims = self._choose_ensemble_by_mean_distance(query, n_probe)
        else:
            best_ensemble_labels, best_sims = self._choose_ensemble_by_mean_distance(query, n_probe, use_weighted_mean=True)

        nearest_labels = np.argpartition(-best_sims, n_probe - 1)[:n_probe]
        return best_ensemble_labels, set(nearest_labels)