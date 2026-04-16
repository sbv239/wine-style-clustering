# Wine Style Clustering

**Unsupervised discovery of wine style archetypes from 23,000+ professional tasting notes**

Using NLP and clustering to answer: *what are the natural style categories that emerge from how wine professionals describe wine?*

## The problem

Wine search is broken. A query for "dried cherry" won't find wines described as "kirsch". A search for "forest floor" misses "sous bois". Professional tasting notes use inconsistent, multilingual, and highly context-dependent vocabulary — keyword search systematically fails.

This project takes a different approach: fine-tune a sentence transformer on domain-specific pairs, embed all descriptions, then let the semantic structure of the text reveal natural style groups.

## Results

**6 red wine styles | 6 white wine styles | 3 rosé styles**

| Colour | k | Best silhouette | Noise removed |
|--------|---|-----------------|---------------|
| Red    | 6 | 0.20 (k=2)      | ~12%          |
| White  | 6 | 0.24 (k=5)      | ~19%          |
| Rosé   | 3 | 0.20 (k=2)      | ~10%          |

**Key finding:** White wines cluster cleanly — the algorithm recovers intuitive style groups (Riesling, Sauvignon Blanc, Chardonnay, etc.) that match domain knowledge. Red and Rosé are diffuse: tasting note language converges across stylistically different wines, making semantic separation harder. HDBSCAN confirmed 77% noise on Red.

### Red wine styles
| Cluster | Style |
|---------|-------|
| 0 | Firm-structured, floral reds with a savoury, earthy core |
| 1 | Silky, red-fruited aromatics with forest floor and mineral depth |
| 2 | Dark, oak-framed power reds built on cassis and graphite |
| 3 | Spice-driven dark reds, approachable and food-ready |
| 4 | Bright, juicy, light reds with crunchy red fruit |
| 5 | Fresh, mineral, well-balanced reds with honest grip |

### White wine styles
| Cluster | Style |
|---------|-------|
| 0 | Rich, mineral-driven whites with creamy orchard depth |
| 1 | Aromatic, fruit-expressive whites across the spectrum |
| 2 | Vibrant, tropical-citrus whites with herbal freshness |
| 3 | Full-bodied, generous whites with honeyed stone fruit weight |
| 4 | Fresh, florally aromatic whites with citrus-saline drive |
| 5 | Precise, mineral-citrus whites with electric acidity |

### Rosé wine styles
| Cluster | Style |
|---------|-------|
| 0 | Deep, structured rosés with dark fruit and tannic grip |
| 1 | Elegant, mineral pale rosés with floral lift and stone fruit |
| 2 | Fresh, vivid rosés with wild berry and citrus-saline snap |

## Project structure

```
wine-style-clustering/
├── data/
│   ├── raw/                      ← Decanter reviews (not committed — large files)
│   └── processed/                ← filtered subset for modelling (not committed)
├── notebooks/
│   ├── 00_preprocessing.ipynb   ← cleaning, LLM denoising, grape normalisation
│   ├── 01_eda.ipynb              ← corpus exploration
│   ├── 02_embeddings_umap.ipynb  ← encode + project to 2D
│   ├── 03_clustering.ipynb       ← K-Means + silhouette analysis + labels
│   └── 04_visualisation.ipynb    ← publication-ready charts (Sprint 4)
├── src/
│   ├── clean_descriptions.py     ← LLM cleaning via Claude API (with checkpointing)
│   ├── build_pairs_style.py      ← training pair construction for fine-tuning
│   └── run_finetune_style.py     ← SentenceTransformer fine-tuning
├── models/                       ← saved model weights (not committed)
├── results/
│   ├── figures/                  ← charts for the article
│   ├── cluster_samples/          ← 200 tasting notes per cluster (for LLM labelling)
│   └── cluster_labels.json       ← final cluster labels
└── reports/                      ← article drafts
```

## Pipeline

```
Decanter scraper
      ↓
00_preprocessing.ipynb    ← basic cleaning + LLM denoising + grape normalisation
      ↓
src/build_pairs_style.py  ← build ~316k training pairs (positive / hard_neg / easy_neg)
      ↓
src/run_finetune_style.py ← fine-tune all-MiniLM-L6-v2 on style pairs (val loss early stopping)
      ↓
02_embeddings_umap.ipynb  ← encode 19,600 wines → UMAP 2D projection
      ↓
03_clustering.ipynb       ← silhouette analysis → K-Means → manual labels
      ↓
04_visualisation.ipynb    ← publication-ready charts
```

## Model

Fine-tuned `all-MiniLM-L6-v2` (384-dim) on ~316k style-focused pairs.

Pair scoring uses structured metadata: grape variety, appellation, region, body, oak, sweetness.  
Training stratified by Colour × Body × Oak group.  
Early stopping on validation loss (MultipleNegativesRankingLoss).

Sanity check (synonym similarity on held-out pairs):
- "dried cherry / kirsch" → 0.79
- "forest floor / sous bois" → 0.70
- "cassis / graphite minerality" → 0.78
- Negative control (red fruit vs oyster shell) → 0.23

## Data

23,266 professional wine reviews from Decanter.com, scraped via custom Selenium pipeline.  
Fields: description, score, Producer, Vintage, Wine Type, Colour, Country, Region, Appellation, Grapes, Body, Oak, Sweetness.

Descriptions LLM-cleaned via Claude Haiku: winemaking details, provenance, and food pairing removed.

## Tech stack

- **Embeddings:** fine-tuned SentenceTransformer (`all-MiniLM-L6-v2`)
- **Dimensionality reduction:** UMAP (n_neighbors=15, min_dist=0.05, cosine)
- **Clustering:** K-Means with silhouette-based k selection
- **Noise removal:** silhouette_samples < 0 → excluded
- **Visualisation:** Matplotlib

## Related project

Builds on [wine-semantic-search](https://github.com/sbv239/wine-semantic-search) — fine-tuned transformer for wine semantic search (+14.7% retrieval accuracy vs base model).

## Article

[What 23,000 tasting notes reveal about wine style — and why the language doesn't always help](https://medium.com/p/903ecbd93680)  
Published on Medium, April 2026
