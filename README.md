# Graph-Based Movie Genre Prediction

**Unveiling cinematic patterns through network analysis**

Final group project for **CS89.23: Network Science and Complex Systems**, Dartmouth College.

We predict a movie's genre from the structure of its character/actor interaction network. Each movie is a graph, and we classify whole graphs using Graph Neural Networks (GraphSAGE, GIN, and GAT), combining node, graph, and edge features into a custom hybrid model.

Full report: [`WRITE_UP.pdf`](WRITE_UP.pdf)

## Why This Project

Existing movie network research mostly focuses on node or edge tasks, like finding popular actors or analyzing gendered interactions. Graph classification for **genre** is far less explored. We wanted to know:

- Which network features are most indicative of genre?
- Which GNN models work best for this task?
- Can combining models improve accuracy?

## Data

| Dataset | Graphs | Genres | What a node / edge means |
| --- | --- | --- | --- |
| **IMDb-BINARY** | 1,000 | Action, Romance | Actor; edge if two actors appear in the same movie |
| **IMDb-MULTI** | 1,500 | Comedy, Romance, Sci-Fi (500 each) | Actor; edge if two actors appear in the same movie |
| **Moviegalaxies** | 773 | Labeled with 3 genres (Comedy, Romance, Sci-Fi) | Character; edge if two characters interact |

- IMDb-BINARY and IMDb-MULTI come from the [Network Repository](https://networkrepository.com) (`imdb_files/`).
- Moviegalaxies comes from [Harvard Dataverse](https://doi.org/10.7910/DVN/T4HBA3) (`dataverse_files/`). It ships without genres, so we looked them up by IMDb ID using the [Cinemagoer](https://cinemagoer.github.io/) package and sampled a balanced set of Comedy, Romance, and Sci-Fi movies.

## Approach

**Features**

- *Node level:* clustering coefficient, eigenvector centrality, plus 4 learned embeddings from a 2-layer GraphSAGE (6 features per node)
- *Graph level:* normalized triangle count, number of communities, average shortest path length, node-to-edge ratio
- *Edge level:* cosine similarity between connected nodes' features (chosen over Pearson correlation after testing both)

**Models**

We started with a simple GAT and built up to a **Hybrid GIN-GAT** model: a GIN layer first, to better tell graph structures apart, followed by GAT layers with attention that also uses the edge weights. Graph-level features are concatenated into the final classifier. We also tested Transformer, GATv2Conv, and AGNNConv layers, which did not help on these small, simple graphs.

**Tuning:** dropout (0.6), attention heads, pooling (mean/add/max), batch norm, Adam with AMSGrad, and cross-entropy vs. NLL loss.

## Results

Best model: **Hybrid GIN-GAT with node, graph, and edge features** (`GIN GAT PlusGE`).

| Dataset | Our best model | Reference baselines |
| --- | --- | --- |
| IMDb-BINARY | 72.0% | 70.7-75.1% |
| IMDb-MULTI | 49.8% | 45.2-49.8% |
| Moviegalaxies | 43.0% | No prior benchmark found |

Baselines include WL, RetGK, PF, PSCN, and WKPI, taken from Zhao & Wang (2019).

**Key takeaways**

- Node features like eigenvector centrality and clustering coefficients carry real genre signal. Eigenvector centrality surfaces the key characters, and clustering coefficients capture tight-knit character groups.
- Node features alone fall short. Adding graph and edge features improved accuracy by about 10 percentage points over node-only models.
- A plain GAT was not enough. Combining GIN with GAT gave the biggest gains.

**Limitations:** the datasets skew toward widely recognized, Hollywood-centric films, and IMDb-MULTI has only three genres and no IMDb IDs to pull richer metadata from.

**Future work:** multi-label classification, since real movies often blend genres.

## Repository Structure

```
CS89.23/
├── Final_Project.ipynb   # Feature extraction, models, experiments, and plots
├── WRITE_UP.pdf          # Full project report
├── imdb_files/           # IMDb-BINARY and IMDb-MULTI data
└── dataverse_files/      # Moviegalaxies data
```

## About the Code

`Final_Project.ipynb` is included as a record of the analysis, with all outputs, plots, and results visible in the notebook. It is not set up to run end to end, since it depends on external datasets that must be downloaded separately (see [Data](#data)).

## My Contributions

This was a team project with equal contribution from all four members. My work:

- Fine-tuned the IMDb-MULTI models and integrated GraphSAGE into the pipeline
- Ran the experiments
- Labeled the Moviegalaxies dataset with genres and balanced it
- Prepared the data and built visualizations
- Interpreted the graph metrics (centrality, clustering, and so on) to understand what they imply about a movie's structure

## Team

- Angelic McPherson ([@Angelic-Work](https://github.com/Angelic-Work))
- Sunishka Jain
- Kate Spillane
- Atul Agarwal
