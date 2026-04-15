"""
src/build_pairs_style.py

Builds training pairs for style-focused SentenceTransformer fine-tuning.

Pair scoring uses structured metadata (body, oak, sweetness, grape variety,
appellation, region) plus semantic similarity on LLM-cleaned descriptions.
Pairs are stratified by Colour×Body group to avoid class imbalance.

Pair types:
    positive             — style score >= threshold OR semantic sim >= SEM_POSITIVE_MIN
    positive_cross_group — high semantic sim across body groups, same colour
    hard_neg             — mid-range semantic sim (challenging negatives)
    easy_neg             — cross-colour pairs (clear negatives)

Output:
    data/processed/style_pairs/train_pairs.parquet
    data/processed/style_pairs/val_pairs.parquet
    data/processed/style_pairs/train_wines.csv
    data/processed/style_pairs/val_wines.csv
    data/processed/style_pairs/pairs_meta.json

Usage:
    python src/build_pairs_style.py
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_PATH = 'data/raw/wines_clean_llm.csv'
OUTPUT_DIR = Path('data/processed/style_pairs')
BASE_MODEL = 'all-MiniLM-L6-v2'

# Pair scoring weights
W_BODY      = 1.5
W_OAK       = 1.0
W_SWEETNESS = 1.0
W_GRAPE     = 3.5   # uses grape_normalized; colour-only fallbacks excluded
W_APPELLATION = 2.5
W_REGION      = 1.5
W_SAME_PRODUCER_VINTAGE = -1.0  # penalty: same producer, different vintage

# Generic colour-only grape labels — too broad to signal style similarity
GENERIC_BLENDS = {'red blend', 'white blend', 'rosé blend', 'orange blend', 'fortified blend'}

# Pair selection thresholds
POSITIVE_THRESHOLD = 5.5   # style score >= this → positive pair
SEM_POSITIVE_MIN   = 0.82  # cosine sim >= this → positive pair (regardless of score)
HARD_NEG_SIM_MIN   = 0.55  # cosine sim range for hard negatives
HARD_NEG_SIM_MAX   = 0.78

# Dataset construction
MAX_POSITIVE_PER_GROUP = 5000   # cap per Colour×Body group
MAX_GROUP_SIZE         = 3000   # max wines sampled per group for pair scoring
HARD_NEG_RATIO         = 1.0   # hard negatives = 1× positives
EASY_NEG_RATIO         = 2.0   # easy negatives = 2× positives
CROSS_GROUP_SAMPLE     = 500_000

VAL_SIZE     = 0.15
RANDOM_STATE = 42
MIN_WORDS    = 15
BATCH_SIZE   = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(val) -> str:
    return str(val).strip().lower() if isinstance(val, str) else ''


def get_text(row) -> str:
    """Return LLM-cleaned description if available, else raw."""
    clean = row.get('description_clean', '')
    if pd.notna(clean) and str(clean).strip():
        return str(clean).strip()
    return str(row.get('description', '')).strip()


def compute_style_score(a: pd.Series, b: pd.Series) -> float:
    """Score stylistic similarity between two wines based on structured metadata."""
    score = 0.0

    # Sensory attributes
    if _norm(a.get('Body', ''))      == _norm(b.get('Body', ''))      != 'unknown': score += W_BODY
    if _norm(a.get('oak_normalized', '')) == _norm(b.get('oak_normalized', '')) != 'unknown': score += W_OAK
    if _norm(a.get('Sweetness', '')) == _norm(b.get('Sweetness', '')) != 'unknown': score += W_SWEETNESS

    # Grape variety (named blends score; colour-only fallbacks do not)
    grape_a = _norm(a.get('grape_normalized', ''))
    grape_b = _norm(b.get('grape_normalized', ''))
    if grape_a and grape_b and grape_a == grape_b and grape_a not in GENERIC_BLENDS:
        score += W_GRAPE

    # Geography
    app_a = _norm(a.get('Appellation', ''))
    app_b = _norm(b.get('Appellation', ''))
    if app_a and app_b and app_a != 'unknown' and app_a == app_b:
        score += W_APPELLATION

    reg_a = _norm(a.get('Region', ''))
    reg_b = _norm(b.get('Region', ''))
    if reg_a and reg_b and reg_a != 'unknown' and reg_a == reg_b:
        score += W_REGION

    # Penalty: same producer, different vintage → trivial pair
    prod_a = _norm(a.get('Producer', ''))
    prod_b = _norm(b.get('Producer', ''))
    if prod_a and prod_a != 'unknown' and prod_a == prod_b:
        if str(a.get('Vintage', '')).strip() != str(b.get('Vintage', '')).strip():
            score += W_SAME_PRODUCER_VINTAGE

    return score


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info('Loaded: %d rows', len(df))
    if 'grape_normalized' not in df.columns:
        raise ValueError('grape_normalized not found — run 00_preprocessing.ipynb first.')
    df['_text'] = df.apply(get_text, axis=1)
    df = df[df['_text'].str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
    log.info('After length filter: %d rows', len(df))
    return df


def split_wines(df: pd.DataFrame):
    df = df.copy()
    df['_stratum'] = (
        df['Colour'].fillna('Unknown').str.strip() + '_' +
        df['Body'].fillna('Unknown').str.strip() + '_' +
        df['oak_normalized'].fillna('unknown').str.strip()
    )
    rare = df['_stratum'].value_counts()
    df.loc[df['_stratum'].isin(rare[rare < 2].index), '_stratum'] = 'Other_Other'

    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, stratify=df['_stratum'], random_state=RANDOM_STATE
    )
    train_df = train_df.drop(columns=['_stratum']).reset_index(drop=True)
    val_df   = val_df.drop(columns=['_stratum']).reset_index(drop=True)
    log.info('Split: %d train / %d val', len(train_df), len(val_df))
    return train_df, val_df


def encode(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    log.info('Encoding %d descriptions...', len(df))
    emb = model.encode(
        df['_text'].tolist(), batch_size=BATCH_SIZE,
        show_progress_bar=True, normalize_embeddings=True,
    )
    log.info('Embeddings shape: %s', emb.shape)
    return emb


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def build_pairs(df: pd.DataFrame, embeddings: np.ndarray, label: str = 'train') -> pd.DataFrame:
    records = df.reset_index(drop=True)
    records['_group'] = (
        records['Colour'].fillna('Unknown').str.strip() + '_' +
        records['Body'].fillna('Unknown').str.strip() + '_' +
        records['oak_normalized'].fillna('unknown').str.strip()
    )

    positive_rows:       list[dict] = []
    hard_neg_candidates: list[dict] = []

    # Within-group pairs
    groups = list(records.groupby('_group'))
    log.info('[%s] Scoring pairs across %d groups...', label, len(groups))

    for g_idx, (group_name, group_df) in enumerate(groups, 1):
        idx = group_df.index.tolist()
        if len(idx) < 2:
            continue
        if len(idx) > MAX_GROUP_SIZE:
            idx = random.sample(idx, MAX_GROUP_SIZE)
        if g_idx % 5 == 0 or g_idx == len(groups):
            log.info('[%s] Group %d/%d (%s): %d wines', label, g_idx, len(groups), group_name, len(idx))

        group_positives: list[dict] = []

        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                i, j    = idx[ii], idx[jj]
                a, b    = records.iloc[i], records.iloc[j]
                sem_sim = float(np.dot(embeddings[i], embeddings[j]))

                if compute_style_score(a, b) >= POSITIVE_THRESHOLD or sem_sim >= SEM_POSITIVE_MIN:
                    group_positives.append({
                        'desc_a': a['_text'], 'desc_b': b['_text'],
                        'label': 1.0, 'pair_type': 'positive',
                    })
                elif HARD_NEG_SIM_MIN <= sem_sim <= HARD_NEG_SIM_MAX:
                    hard_neg_candidates.append({
                        'desc_a': a['_text'], 'desc_b': b['_text'],
                        'label': 0.0, 'pair_type': 'hard_neg', '_sim': sem_sim,
                    })

        random.shuffle(group_positives)
        positive_rows.extend(group_positives[:MAX_POSITIVE_PER_GROUP])
        log.info('[%s] Group %s — positives: %d → capped: %d',
                 label, group_name, len(group_positives),
                 min(len(group_positives), MAX_POSITIVE_PER_GROUP))

    log.info('[%s] Within-group — positive: %d | hard_neg candidates: %d',
             label, len(positive_rows), len(hard_neg_candidates))

    # Cross-group semantic positives (same colour only)
    log.info('[%s] Sampling %d cross-group pairs...', label, CROSS_GROUP_SAMPLE)
    all_idx: list[int]           = records.index.tolist()
    seen_pairs: set[tuple]       = set()
    cross_positive               = 0
    n_sample                     = min(CROSS_GROUP_SAMPLE, len(all_idx) * 50)

    while len(seen_pairs) < n_sample:
        i, j = random.sample(all_idx, 2)
        key  = (min(i, j), max(i, j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if _norm(records.iloc[i].get('Colour', '')) != _norm(records.iloc[j].get('Colour', '')):
            continue
        if float(np.dot(embeddings[i], embeddings[j])) >= SEM_POSITIVE_MIN:
            positive_rows.append({
                'desc_a': records.iloc[i]['_text'],
                'desc_b': records.iloc[j]['_text'],
                'label': 1.0, 'pair_type': 'positive_cross_group',
            })
            cross_positive += 1

    log.info('[%s] Cross-group semantic positives: %d', label, cross_positive)

    # Assemble
    n_pos = len(positive_rows)
    random.shuffle(positive_rows)

    hard_neg_candidates.sort(key=lambda x: x['_sim'], reverse=True)
    hard_neg_rows = [
        {k: v for k, v in r.items() if k != '_sim'}
        for r in hard_neg_candidates[:int(n_pos * HARD_NEG_RATIO)]
    ]

    easy_rows: list[dict] = []
    attempts  = 0
    n_easy    = int(n_pos * EASY_NEG_RATIO)
    while len(easy_rows) < n_easy and attempts < n_easy * 10:
        i, j = random.sample(all_idx, 2)
        if _norm(records.iloc[i].get('Colour', '')) != _norm(records.iloc[j].get('Colour', '')):
            easy_rows.append({
                'desc_a': records.iloc[i]['_text'],
                'desc_b': records.iloc[j]['_text'],
                'label': 0.0, 'pair_type': 'easy_neg',
            })
        attempts += 1

    all_rows = positive_rows + hard_neg_rows + easy_rows
    random.shuffle(all_rows)
    result = pd.DataFrame(all_rows)

    log.info('[%s] Final: %d positive | %d hard_neg | %d easy_neg = %d total',
             label, n_pos, len(hard_neg_rows), len(easy_rows), len(result))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if all((OUTPUT_DIR / f).exists() for f in ('train_pairs.parquet', 'val_pairs.parquet')):
        log.info('Pairs already exist in %s — delete to rebuild.', OUTPUT_DIR)
        return

    df = load_and_filter(INPUT_PATH)
    train_df, val_df = split_wines(df)
    train_df.to_csv(OUTPUT_DIR / 'train_wines.csv', index=False)
    val_df.to_csv(OUTPUT_DIR / 'val_wines.csv', index=False)

    model     = SentenceTransformer(BASE_MODEL)
    train_emb = encode(train_df, model)
    val_emb   = encode(val_df, model)

    log.info('Building TRAIN pairs...')
    train_pairs = build_pairs(train_df, train_emb, label='train')

    log.info('Building VAL pairs...')
    val_pairs = build_pairs(val_df, val_emb, label='val')

    train_pairs.to_parquet(OUTPUT_DIR / 'train_pairs.parquet', index=False)
    val_pairs.to_parquet(OUTPUT_DIR / 'val_pairs.parquet', index=False)

    meta = {
        'built_at':   datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'base_model': BASE_MODEL,
        'input':      INPUT_PATH,
        'thresholds': {
            'positive_score': POSITIVE_THRESHOLD,
            'sem_positive':   SEM_POSITIVE_MIN,
            'hard_neg_range': [HARD_NEG_SIM_MIN, HARD_NEG_SIM_MAX],
        },
        'weights': {
            'body': W_BODY, 'oak': W_OAK, 'sweetness': W_SWEETNESS,
            'grape': W_GRAPE, 'appellation': W_APPELLATION, 'region': W_REGION,
            'same_producer_vintage': W_SAME_PRODUCER_VINTAGE,
        },
        'train_wines':      len(train_df),
        'val_wines':        len(val_df),
        'train_pairs':      len(train_pairs),
        'val_pairs':        len(val_pairs),
        'train_pair_types': train_pairs['pair_type'].value_counts().to_dict(),
        'val_pair_types':   val_pairs['pair_type'].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / 'pairs_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info('=' * 60)
    log.info('Done. Train: %d | Val: %d', len(train_pairs), len(val_pairs))
    log.info('Next: python src/run_finetune_style.py')
    log.info('=' * 60)


if __name__ == '__main__':
    main()