"""
src/clean_descriptions.py

Cleans wine tasting notes via Claude API.
Keeps only organoleptic content — aroma, taste, style, character.
Removes winemaking process, personal comments, provenance details.

Part of the preprocessing pipeline:
    00_preprocessing.ipynb → clean_descriptions.py → 00_preprocessing.ipynb (merge step)

Usage:
    python src/clean_descriptions.py

Checkpointing: processed rows are cached to data/processed/descriptions_cache.json
and skipped on re-run. Safe to interrupt and resume at any time.

Input:  data/raw/wines_clean.csv          (from 00_preprocessing.ipynb step 1)
Output: data/raw/wines_clean_llm.csv      (merged back in 00_preprocessing.ipynb step 2)
"""

import os
import re
import json
import time
import pandas as pd
import anthropic
from tqdm import tqdm

# --- Config ---
INPUT_PATH  = 'data/raw/wines_clean.csv'
OUTPUT_PATH = 'data/raw/wines_clean_llm.csv'
CACHE_PATH  = 'data/processed/descriptions_cache.json'

MODEL      = 'claude-haiku-4-5-20251001'
BATCH_SIZE = 10    # descriptions per API call
SAVE_EVERY = 50    # save cache + CSV every N descriptions
SLEEP_SEC  = 0.1   # pause between API calls

SYSTEM_PROMPT = """You are a wine editor. Clean professional wine tasting notes for style segmentation.

Keep only:
- Organoleptic description: colour, aroma, taste, texture, structure, finish, ageing potential
- Overall style and character of the wine

Remove:
- Winemaking details (fermentation, ageing, barrels, clones, yields)
- Critic's personal comments unrelated to this wine's taste
- Provenance, geography, vineyard or producer history
- Food pairing suggestions
- Technical data (pH, production volume, bottle size, alcohol)
- Generic filler phrases that don't describe this wine specifically

Rules:
- Do not rephrase or rewrite — only remove sentences or phrases
- Keep the original wording of everything you retain
- If a note is already clean, return it unchanged
- If nothing remains after cleaning, return the original unchanged
- Return only the cleaned notes in [1], [2], ... format. No explanations.

Tasting notes:
"""

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
client = anthropic.Anthropic()


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def clean_batch(descriptions: list) -> list:
    """Send a batch of descriptions in one API call, return cleaned list."""
    numbered = '\n\n'.join(f'[{i+1}]\n{d}' for i, d in enumerate(descriptions))

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': numbered}],
    )

    text  = response.content[0].text.strip()
    parts = re.split(r'\[(\d+)\]', text)

    parsed = {}
    for i in range(1, len(parts), 2):
        idx     = int(parts[i]) - 1
        content = parts[i + 1].strip() if i + 1 < len(parts) else ''
        if content:
            parsed[idx] = content

    return [parsed.get(i, descriptions[i]) for i in range(len(descriptions))]


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f'Loaded: {len(df):,} rows')

    cache = load_cache()
    print(f'Cache: {len(cache):,} already processed')

    key_col = 'url' if 'url' in df.columns else ('title' if 'title' in df.columns else None)

    def get_key(row, idx):
        return str(row[key_col]) if key_col else str(idx)

    results    = [None] * len(df)
    to_process = []

    for idx, row in df.iterrows():
        key = get_key(row, idx)
        if key in cache:
            results[idx] = cache[key]
        else:
            to_process.append((idx, key, str(row['description'])))

    print(f'To process: {len(to_process):,} | Cached: {len(df) - len(to_process):,}')

    changed   = 0
    errors    = 0
    processed = 0

    for batch_start in tqdm(range(0, len(to_process), BATCH_SIZE), desc='Cleaning'):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]
        idxs  = [b[0] for b in batch]
        keys  = [b[1] for b in batch]
        descs = [b[2] for b in batch]

        try:
            cleaned = clean_batch(descs)
            for idx, key, orig, clean in zip(idxs, keys, descs, cleaned):
                cache[key]    = clean
                results[idx]  = clean
                if clean != orig:
                    changed += 1
            time.sleep(SLEEP_SEC)
        except Exception as e:
            print(f'\nError at row {idxs[0]}: {e}')
            for idx, orig in zip(idxs, descs):
                results[idx] = orig
            errors += len(batch)

        processed += len(batch)

        if processed % SAVE_EVERY == 0:
            save_cache(cache)
            df['description_clean'] = results
            df.to_csv(OUTPUT_PATH, index=False)

    save_cache(cache)
    df['description_clean'] = results
    df.to_csv(OUTPUT_PATH, index=False)

    print(f'\nDone.')
    print(f'  Rows:    {len(df):,}')
    print(f'  Changed: {changed:,}')
    print(f'  Errors:  {errors}')
    print(f'  Output:  {OUTPUT_PATH}')


if __name__ == '__main__':
    main()