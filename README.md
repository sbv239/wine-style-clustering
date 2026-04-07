# Wine Style Clustering

**Unsupervised discovery of wine style archetypes from 10,000+ professional tasting notes**

Using NLP and clustering to answer: *what are the fundamental style categories that emerge from how wine professionals describe wine?*

## The problem

Wine search is broken. A query for "dried cherry" won't find wines described as "kirsch". A search for "forest floor" misses "sous bois". Professional tasting notes use inconsistent, multilingual, and highly context-dependent vocabulary — keyword search systematically fails.

This project takes a different approach: embed all descriptions with a domain-fine-tuned transformer, then let the semantic structure of the text reveal natural style groups.

## Results

*(To be populated after Sprint 3)*

**7 red wine styles | 7 white wine styles**  
Each cluster described by its top TF-IDF descriptors and a one-sentence human label.

## Project structure

```
wine-style-clustering/
├── data/
│   ├── raw/          ← Decanter reviews (wines_clean.csv)
│   └── processed/    ← filtered subset for modelling
├── notebooks/
│   ├── 01_eda.ipynb              ← corpus exploration (this sprint)
│   ├── 02_embeddings_umap.ipynb  ← encode + project to 2D
│   ├── 03_clustering.ipynb       ← HDBSCAN + K-Means + TF-IDF labels
│   └── 04_visualisation.ipynb    ← publication-ready charts
├── models/           ← saved embeddings, cluster assignments
├── results/
│   └── figures/      ← charts used in the article
└── reports/          ← final article draft
```

## Tech stack

- **Embeddings:** fine-tuned SentenceTransformer (wine domain)
- **Dimensionality reduction:** UMAP
- **Clustering:** HDBSCAN → K-Means
- **Descriptor extraction:** TF-IDF per cluster
- **Visualisation:** Matplotlib, Seaborn, Plotly

## Data

10,000+ professional wine reviews from Decanter.com, scraped via custom Selenium pipeline (see [wine-semantic-search](https://github.com/sbv239/wine-semantic-search)).

Fields: description, score, Producer, Vintage, Wine Type, Colour, Country, Region, Appellation, Grapes, Body, Oak.

## Related project

This builds on [wine-semantic-search](https://github.com/sbv239/wine-semantic-search) — a fine-tuned transformer model for wine semantic search (+14.7% retrieval accuracy vs base model).

## Article

*"What 10,000 tasting notes reveal about wine styles"* — Medium / Towards Data Science  
*(link to be added)*
