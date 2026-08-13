<img width="2115" height="744" alt="" src="https://github.com/user-attachments/assets/b0808da6-a407-481f-bd61-059edd3db738" />


# `multi-ivf`: Efficient IVF with Multi-Cluster Ensembles, Multi-Assignment, and Multi-Probe

`multi-ivf` is a high-efficiency IVF library for approximate nearest neighbor search.

It achieves the same recall as conventional IVF with significantly fewer candidate vectors (around 40-50% reduction), reducing the computational cost of the final exact KNN search.

Unlike end-to-end ANN libraries, this library focuses only on clustering, making it easy to integrate with any database.

This library is particularly well suited for systems that perform exact reranking on top of a vector database (e.g., `Firestore`), because reducing the candidate set directly reduces the cost of the final search.

## Benchmarks

### Benchmark Setup

- **Dataset size:** 1,000,000+ vectors
- **Dataset:** [wiki40b (English)](https://huggingface.co/datasets/google/wiki40b)
- **Embedding model:** [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)
- **Target recall:** Recall@10 = 0.95, Recall@100 = 0.95
- **Dataset Download:** Available from [GitHub Releases](https://github.com/tommyktech/multi-ivf/releases/tag/benchmark-0.1.0)

### Evaluation

We compare **Conventional IVF** (with Multi-Probe) and **Multi IVF** by measuring the number of candidate vectors required to achieve approximately the same recall.

A smaller candidate size means fewer vectors need to be re-ranked, resulting in lower KNN computation cost.

### Results

#### Candidate Size (Recall@10 = 0.95)
| n_clusters | Conventional IVF | Multi IVF |
|-----------:|---------:|------:|
| 1000 | 32504 | 20019 |
| 2000 | 25075 | 15924 |
| 4000 | 18536 | 11155 |
| 6000 | 15521 | 8910 |
| 8000 | 14342 | 7654 |

#### Candidate Size (Recall@100 = 0.95)
| n_clusters | Conventional IVF | Multi IVF |
|-----------:|---------:|------:|
| 1000 | 36511 | 20019 |
| 2000 | 29607 | 15924 |
| 4000 | 22946 | 12302 |
| 6000 | 20466 | 10807 |
| 8000 | 19134 | 9885 |

## Installation

```bash
pip install https://github.com/tommyktech/multi-ivf/releases/download/0.1.0/multi_ivf-0.1.0-py3-none-any.whl
```

> **Note:** PyPI support is coming soon. Please use the GitHub Release package for now.


## Basic Usage

For a complete working example, see `examples/basic_usage.py`.

```python
from multi_ivf import MultiIVF

# Load sample dataset
X_train, X_query = ...  # Your embedding vectors (NumPy arrays)

# Determine n_clusters. 50–100 vectors per cluster are generally sufficient.
N = X_train.shape[0]
n_clusters = int(N / 100)

# Initialize
mivf = MultiIVF(
    n_clusters = n_clusters,
    n_ensembles = 10, # Sufficient for most use cases
    use_mean_centering = True,
    gpu_id = 0, # If not set, only CPUs will be used
)

# Train
mivf.train(X_train)

# Save the model as a joblib file
model_path = ...
mivf.save(model_path)

# Load model from file
mivf = MultiIVF.load(model_path)

# Assign labels to data
assignments = mivf.assign(
    X_train,
    assign_margin = 0.1, # Sufficient for most use cases
    n_assignments = 50 # Specify a value smaller than n_clusters. Depending on n_clusters, around 50–100 is usually sufficient.
)

# Search for candidate data
for query in X_query:
    q_ensemble, q_labels = mivf.search(query, n_probe=10) # Choose n_probe based on your requirements.

    # Then collect candidate vectors from your local data or vector database using the query's cluster labels

```

## Algorithm Overview


### Multi-Cluster Ensembles

Multi-Cluster Ensembles builds multiple independent K-means models (ensembles) and selects the best ensemble for each query.

The ensemble selection process is as follows:

1. For each ensemble, find the `n_probe` centroids closest to the query.
2. Compute the average distance to those centroids.
3. Select the ensemble with the smallest average distance.
4. Return the labels of the selected centroids from the chosen ensemble.

This approach achieves the same recall with a smaller candidate size than conventional IVF.

To disable Multi-Cluster Ensembles, set `n_ensembles=1`.


### Multi-Assignment

Multi-Assignment is a method that assigns multiple nearby cluster labels to each vector during indexing.
It is used to reduce missed results for vectors located near cluster boundaries.
This library supports two methods for determining neighboring clusters:
- Distance-based assignment
- Top-k assignment


### Multi-Probe

Multi-Probe is a method that improves recall by searching multiple nearby clusters during query time.
While Multi-Assignment is applied during indexing, Multi-Probe is applied during search.
The `n_probe` nearest centroids are selected in order of distance, and their corresponding cluster labels are searched.


### Mean-Centering

Mean-Centering is a technique that centers the vector space by subtracting the mean vector during training.
It is used to reduce bias (anisotropy) in the embedding space and improve the discriminative power of embeddings.
Since embeddings generated by LLMs can have strong anisotropy, Mean-Centering can be effective in such cases.


## Example Application

- Real-time news aggregation that identifies major stories across multiple sources
  https://bsky.app/profile/news-jp.bsky.social


## Planned Features

* [ ] Learning-to-Rank–based ranking (inspired by https://arxiv.org/pdf/2404.11731)


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

See the LICENSE file for details.