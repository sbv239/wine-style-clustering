"""
src/run_finetune_style.py

Fine-tunes all-MiniLM-L6-v2 on style-focused wine pairs.
Uses MultipleNegativesRankingLoss with gradient accumulation (effective batch = 64).
Early stopping on val loss — lower is better.

Pairs built by: src/build_pairs_style.py

Usage:
    python src/run_finetune_style.py
    python src/run_finetune_style.py --epochs 5 --batch-size 32 --accum-steps 2

Output:
    models/style-finetuned/final/   ← set MODEL_PATH to this in 02_embeddings_umap.ipynb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

BASE_MODEL = 'all-MiniLM-L6-v2'
PAIRS_DIR  = Path('data/processed/style_pairs')
OUTPUT_DIR = Path('models/all-MiniLM-L6-v2-finetuned-v6')

EPOCHS       = 7
BATCH_SIZE   = 32   # physical batch per step
ACCUM_STEPS  = 2    # effective batch = BATCH_SIZE * ACCUM_STEPS = 64
LR           = 2e-5
WARMUP_RATIO = 0.1
MAX_LENGTH   = 128
PATIENCE     = 2    # early stopping patience on val loss
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, examples: list[InputExample], tokenizer, max_length: int):
        log.info('Tokenizing %d pairs...', len(examples))
        texts_a = [e.texts[0] for e in examples]
        texts_b = [e.texts[1] for e in examples]
        self.enc_a = tokenizer(texts_a, padding=True, truncation=True,
                               max_length=max_length, return_tensors='pt')
        self.enc_b = tokenizer(texts_b, padding=True, truncation=True,
                               max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor([e.label for e in examples], dtype=torch.float32)
        log.info('Done.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {k: v[idx] for k, v in self.enc_a.items()},
            {k: v[idx] for k, v in self.enc_b.items()},
            self.labels[idx],
        )


def collate_fn(batch):
    a_list, b_list, labels = zip(*batch)
    def stack(dicts):
        return {k: torch.stack([d[k] for d in dicts]) for k in dicts[0]}
    return stack(a_list), stack(b_list), torch.stack(labels)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, dataloader, loss_fn, device, accum_steps,
              optimizer=None, scheduler=None) -> float:
    """One pass over dataloader. If optimizer is None, runs in eval mode (no grad)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, steps = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    if training:
        optimizer.zero_grad()

    with ctx:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc='train' if training else 'val  ', unit='batch')

        for step, (enc_a, enc_b, labels) in pbar:
            enc_a  = {k: v.to(device) for k, v in enc_a.items()}
            enc_b  = {k: v.to(device) for k, v in enc_b.items()}
            labels = labels.to(device)

            if training:
                loss_val = loss_fn([enc_a, enc_b], labels) / accum_steps
                loss_val.backward()
                total_loss += loss_val.item() * accum_steps
                steps += 1

                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix({'loss': f'{loss_val.item() * accum_steps:.4f}'})
            else:
                loss_val = loss_fn([enc_a, enc_b], labels)
                total_loss += loss_val.item()
                steps += 1
                pbar.set_postfix({'val_loss': f'{loss_val.item():.4f}'})

        # Flush remaining gradients if dataset not divisible by accum_steps
        if training and (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=EPOCHS)
    parser.add_argument('--batch-size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--accum-steps', type=int,   default=ACCUM_STEPS)
    parser.add_argument('--lr',          type=float, default=LR)
    parser.add_argument('--patience',    type=int,   default=PATIENCE)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    effective_batch = args.batch_size * args.accum_steps

    train_path = PAIRS_DIR / 'train_pairs.parquet'
    val_path   = PAIRS_DIR / 'val_pairs.parquet'
    if not train_path.exists():
        raise FileNotFoundError(f'Pairs not found. Run: python src/build_pairs_style.py')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load positive pairs only — MNRL doesn't use explicit negatives
    log.info('Loading pairs from %s', PAIRS_DIR)
    train_pos = pd.read_parquet(train_path).query("label == 1.0")
    val_pos   = pd.read_parquet(val_path).query("label == 1.0")
    log.info('Train positives: %d | Val positives: %d', len(train_pos), len(val_pos))

    def make_examples(df):
        return [InputExample(texts=[r['desc_a'], r['desc_b']], label=1.0)
                for _, r in df.iterrows()]

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info('Device: %s | Effective batch: %d', device, effective_batch)

    # Model
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_LENGTH
    model = model.to(device)

    # Tokenize
    train_dataset = PairDataset(make_examples(train_pos), model.tokenizer, MAX_LENGTH)
    val_dataset   = PairDataset(make_examples(val_pos),   model.tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, shuffle=True,  batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   shuffle=False, batch_size=args.batch_size,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # Loss / optimiser / scheduler
    loss_fn         = losses.MultipleNegativesRankingLoss(model).to(device)
    optimizer_steps = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps    = int(optimizer_steps * WARMUP_RATIO)
    optimizer       = AdamW(model.parameters(), lr=args.lr, eps=1e-6)
    scheduler       = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                total_iters=max(warmup_steps, 1))
    log.info('Optimizer steps: %d | Warmup: %d', optimizer_steps, warmup_steps)

    # Training loop — early stopping on val loss (lower = better)
    best_val_loss = float('inf')
    best_epoch    = 0
    no_improve    = 0
    best_ckpt     = str(OUTPUT_DIR / 'best_checkpoint')
    history       = []

    for epoch in range(1, args.epochs + 1):
        log.info('--- Epoch %d / %d ---', epoch, args.epochs)

        train_loss = run_epoch(model, train_loader, loss_fn, device, args.accum_steps,
                               optimizer=optimizer, scheduler=scheduler)
        val_loss   = run_epoch(model, val_loader,   loss_fn, device, args.accum_steps)

        log.info('train_loss=%.4f  val_loss=%.4f', train_loss, val_loss)
        history.append({'epoch': epoch, 'train_loss': round(train_loss, 4),
                        'val_loss': round(val_loss, 4)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            model.save(best_ckpt)
            log.info('✓ Best val_loss=%.4f — checkpoint saved', best_val_loss)
        else:
            no_improve += 1
            log.info('No improvement %d/%d (best %.4f @ epoch %d)',
                     no_improve, args.patience, best_val_loss, best_epoch)
            if no_improve >= args.patience:
                log.info('Early stopping.')
                break

    # Restore best and save as final
    model = SentenceTransformer(best_ckpt)
    final_path = str(OUTPUT_DIR / 'final')
    model.save(final_path)

    summary = {
        'run_at':        datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'base_model':    BASE_MODEL,
        'output':        final_path,
        'best_epoch':    best_epoch,
        'best_val_loss': best_val_loss,
        'history':       history,
        'config': {
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'accum_steps': args.accum_steps, 'effective_batch': effective_batch,
            'lr': args.lr, 'patience': args.patience,
        },
    }
    with open(OUTPUT_DIR / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info('=' * 60)
    log.info('Done. Best epoch %d  val_loss %.4f', best_epoch, best_val_loss)
    log.info("MODEL_PATH = 'models/all-MiniLM-L6-v2-finetuned-v6'")
    log.info('=' * 60)


if __name__ == '__main__':
    main()