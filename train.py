"""Train the emotion classifier from scratch.

Reads data/train.csv, data/validation.csv and data/test.csv (columns: text,label),
learns a TF-IDF + softmax regression model with hand-written gradient descent
(Adam, L2 regularisation, early stopping) using only NumPy, and writes the
weights to EmotionDetection/model.json.

    pip install -r requirements-train.txt
    python download_data.py      # once, gets the dataset
    python train.py
"""
# Numeric training code passes many arrays around, so these style checks are off.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from EmotionDetection.emotion_detection import (DEFAULT_LABELS, DEFAULT_MODEL_PATH,
                                                extract_features)


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def read_csv(path):
    """Read a CSV with 'text' and 'label' columns."""
    texts, labels = [], []
    with open(path, newline='', encoding='utf-8') as csv_file:
        for row in csv.DictReader(csv_file):
            texts.append(row['text'])
            labels.append(row['label'])
    return texts, labels


def order_labels(present):
    """Keep the dataset's usual label order, then any extra labels alphabetically."""
    known = [label for label in DEFAULT_LABELS if label in present]
    return known + sorted(set(present) - set(known))


def build_vocabulary(documents, max_features, min_df):
    """Pick the most common features and compute their IDF (rarity) weights."""
    doc_freq = Counter()
    for features in documents:
        doc_freq.update(set(features))
    kept = [(feature, df) for feature, df in doc_freq.items() if df >= min_df]
    kept.sort(key=lambda item: (-item[1], item[0]))
    kept = kept[:max_features]
    n_docs = len(documents)
    index = {feature: i for i, (feature, _) in enumerate(kept)}
    idf = np.array([math.log((1 + n_docs) / (1 + df)) + 1.0 for _, df in kept])
    return index, idf


def vectorize(documents, index, idf):
    """Turn feature lists into a sparse TF-IDF matrix (CSR arrays, L2 normalised)."""
    indptr, indices, data = [0], [], []
    for features in documents:
        counts = Counter(feature for feature in features if feature in index)
        row = [(index[f], (1.0 + math.log(c)) * idf[index[f]])
               for f, c in counts.items()]
        norm = math.sqrt(sum(value * value for _, value in row))
        for col, value in row:
            indices.append(col)
            data.append(value / norm)
        indptr.append(len(indices))
    return (np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int64),
            np.array(data, dtype=np.float64))


def csr_rows(matrix, rows):
    """Pull the given rows out of a CSR matrix as (row number, column, value) arrays."""
    indptr, indices, data = matrix
    starts = indptr[rows]
    lengths = indptr[rows + 1] - starts
    total = int(lengths.sum())
    batch_rows = np.repeat(np.arange(len(rows)), lengths)
    offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    positions = np.repeat(starts, lengths) + offsets
    return batch_rows, indices[positions], data[positions]


# --------------------------------------------------------------------------
# The model: softmax regression trained with Adam
# --------------------------------------------------------------------------
def softmax(logits):
    """Turn raw scores into probabilities that sum to 1 in every row."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def forward(matrix, rows, weights, bias):
    """Probabilities for the given rows of the matrix."""
    batch_rows, cols, vals = csr_rows(matrix, rows)
    logits = np.tile(bias, (len(rows), 1))
    for k in range(weights.shape[1]):
        logits[:, k] += np.bincount(batch_rows, weights=vals * weights[cols, k],
                                    minlength=len(rows))
    return softmax(logits), (batch_rows, cols, vals)


def predict_proba(matrix, weights, bias):
    """Probabilities for every row of the matrix."""
    n_rows = len(matrix[0]) - 1
    return forward(matrix, np.arange(n_rows), weights, bias)[0]


def cross_entropy(probs, labels):
    """Average negative log-likelihood of the true labels."""
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def batch_gradient(matrix, rows, labels, sample_weight, weights, bias, l2):
    """Gradient of the weighted cross-entropy loss (plus L2) for one mini-batch."""
    n_features, n_classes = weights.shape
    probs, (batch_rows, cols, vals) = forward(matrix, rows, weights, bias)
    err = probs.copy()
    err[np.arange(len(rows)), labels[rows]] -= 1.0
    err *= sample_weight[rows][:, None] / len(rows)
    grad_w = l2 * weights
    for k in range(n_classes):
        grad_w[:, k] += np.bincount(cols, weights=vals * err[batch_rows, k],
                                    minlength=n_features)
    return grad_w, err.sum(axis=0)


def fit(train, train_y, val, val_y, n_features, n_classes, *, l2, epochs, lr,
        batch_size, patience, seed, class_weights, log):
    """Gradient descent with Adam and early stopping. Returns the best weights."""
    rng = np.random.default_rng(seed)
    n_docs = len(train_y)
    weights = np.zeros((n_features, n_classes))
    bias = np.zeros(n_classes)
    m_w, v_w = np.zeros_like(weights), np.zeros_like(weights)
    m_b, v_b = np.zeros_like(bias), np.zeros_like(bias)
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    sample_weight = class_weights[train_y]
    best = {'acc': -1.0, 'loss': math.inf, 'w': weights.copy(), 'b': bias.copy(),
            'epoch': 0}
    stale = 0

    for epoch in range(1, epochs + 1):
        rate = lr * (0.95 ** (epoch - 1))
        order = rng.permutation(n_docs)
        for start in range(0, n_docs, batch_size):
            rows = order[start:start + batch_size]
            grad_w, grad_b = batch_gradient(train, rows, train_y, sample_weight,
                                            weights, bias, l2)
            step += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * grad_w ** 2
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2
            scale_1, scale_2 = 1 - beta1 ** step, 1 - beta2 ** step
            weights -= rate * (m_w / scale_1) / (np.sqrt(v_w / scale_2) + eps)
            bias -= rate * (m_b / scale_1) / (np.sqrt(v_b / scale_2) + eps)

        val_probs = predict_proba(val, weights, bias)
        val_acc = float((val_probs.argmax(axis=1) == val_y).mean())
        val_loss = cross_entropy(val_probs, val_y)
        log(f'  epoch {epoch:3d}  validation accuracy {val_acc:.4f}  loss {val_loss:.4f}')
        if val_acc > best['acc'] or (val_acc == best['acc'] and val_loss < best['loss']):
            best.update(acc=val_acc, loss=val_loss, w=weights.copy(), b=bias.copy(),
                        epoch=epoch)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f'  no improvement for {patience} epochs, stopping early')
                break
    return best


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(probs, y_true, labels, neutral_below):
    """Accuracy, macro F1 and per-emotion precision/recall/F1."""
    n_classes = len(labels)
    y_pred = probs.argmax(axis=1)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1
    per_class = {}
    for k, label in enumerate(labels):
        tp = confusion[k, k]
        precision = tp / confusion[:, k].sum() if confusion[:, k].sum() else 0.0
        recall = tp / confusion[k].sum() if confusion[k].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        per_class[label] = {'precision': float(precision), 'recall': float(recall),
                            'f1': float(f1), 'support': int(confusion[k].sum())}
    confident = probs.max(axis=1) >= neutral_below
    correct = y_pred == y_true
    return {
        'accuracy': float(correct.mean()),
        'macro_f1': float(np.mean([c['f1'] for c in per_class.values()])),
        'per_class': per_class,
        'confusion': confusion,
        'confident_share': float(confident.mean()),
        'confident_accuracy': float(correct[confident].mean()) if confident.any() else 0.0,
    }


def print_report(title, metrics, labels, log):
    """Print an evaluation report."""
    log(f'\n{title}')
    log(f"  accuracy {metrics['accuracy']:.4f}    macro F1 {metrics['macro_f1']:.4f}")
    log(f"  {'emotion':10s} {'precision':>9s} {'recall':>7s} {'f1':>6s} {'count':>6s}")
    for label in labels:
        row = metrics['per_class'][label]
        log(f"  {label:10s} {row['precision']:9.3f} {row['recall']:7.3f} "
            f"{row['f1']:6.3f} {row['support']:6d}")
    log('  confusion matrix (rows = true emotion, columns = predicted):')
    log('  ' + ' ' * 10 + ' '.join(f'{label[:7]:>7s}' for label in labels))
    for k, label in enumerate(labels):
        log(f'  {label:10s}' + ' '.join(f'{v:7d}' for v in metrics['confusion'][k]))
    log(f"  the model is confident on {metrics['confident_share']:.1%} of texts, "
        f"with accuracy {metrics['confident_accuracy']:.4f} on those")


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------
def train_model(train_data, val_data, test_data=None, *, max_features=20000,
                min_df=2, l2_grid=(1e-7, 1e-6, 1e-5), epochs=40, lr=0.1,
                batch_size=256, patience=5, seed=0, neutral_below=0.4,
                class_weight='none', log=print):
    """Train on (texts, labels) pairs and return the model dictionary."""
    started = time.time()
    labels = order_labels(set(train_data[1]))
    label_id = {label: i for i, label in enumerate(labels)}
    for name, data in (('validation', val_data), ('test', test_data)):
        if data is not None and not set(data[1]) <= set(labels):
            raise ValueError(f'The {name} set has labels the training set does not have')

    log('Extracting features...')
    train_docs = [extract_features(text) for text in train_data[0]]
    index, idf = build_vocabulary(train_docs, max_features, min_df)
    n_features, n_classes = len(index), len(labels)
    log(f'  {len(train_docs):,} training texts, {n_features:,} features, '
        f'{n_classes} emotions: {", ".join(labels)}')

    train_x = vectorize(train_docs, index, idf)
    val_x = vectorize([extract_features(t) for t in val_data[0]], index, idf)
    train_y = np.array([label_id[label] for label in train_data[1]])
    val_y = np.array([label_id[label] for label in val_data[1]])

    counts = np.bincount(train_y, minlength=n_classes).astype(float)
    if class_weight == 'balanced':
        class_weights = len(train_y) / (n_classes * np.maximum(counts, 1))
    else:
        class_weights = np.ones(n_classes)

    best = {'acc': -1.0}
    for l2 in l2_grid:
        log(f'\nTraining with L2 strength {l2:g}')
        run = fit(train_x, train_y, val_x, val_y, n_features, n_classes, l2=l2,
                  epochs=epochs, lr=lr, batch_size=batch_size, patience=patience,
                  seed=seed, class_weights=class_weights, log=log)
        log(f"  best validation accuracy {run['acc']:.4f} at epoch {run['epoch']}")
        if run['acc'] > best['acc']:
            best = dict(run, l2=l2)
    log(f"\nChosen L2 strength: {best['l2']:g}")

    val_metrics = evaluate(predict_proba(val_x, best['w'], best['b']), val_y,
                           labels, neutral_below)
    print_report('Validation results', val_metrics, labels, log)
    meta = {'train_docs': len(train_y), 'vocab_size': n_features,
            'l2': best['l2'], 'epochs_used': best['epoch'],
            'val_accuracy': round(val_metrics['accuracy'], 4),
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'algorithm': 'TF-IDF + softmax regression, trained from scratch with NumPy'}

    if test_data is not None:
        test_x = vectorize([extract_features(t) for t in test_data[0]], index, idf)
        test_y = np.array([label_id[label] for label in test_data[1]])
        test_metrics = evaluate(predict_proba(test_x, best['w'], best['b']), test_y,
                                labels, neutral_below)
        print_report('Test results (texts the model never saw)', test_metrics,
                     labels, log)
        meta.update(test_docs=len(test_y), test_accuracy=round(test_metrics['accuracy'], 4),
                    test_macro_f1=round(test_metrics['macro_f1'], 4))

    features = sorted(index, key=index.get)
    terms = {feature: [round(float(idf[index[feature]]), 4)] +
             [round(float(w), 4) for w in best['w'][index[feature]]]
             for feature in features}
    log(f'\nDone in {time.time() - started:.0f} seconds.')
    return {'version': 1, 'labels': labels, 'neutral_below': neutral_below,
            'bias': [round(float(b), 4) for b in best['b']], 'terms': terms,
            'meta': meta}


def save_model(model, path):
    """Write the model as compact JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as model_file:
        json.dump(model, model_file, separators=(',', ':'))


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    parser.add_argument('--data', default='data', help='folder with train/validation/test CSVs')
    parser.add_argument('--out', default=DEFAULT_MODEL_PATH, help='where to write model.json')
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--l2', default='1e-7,1e-6,1e-5',
                        help='comma-separated L2 strengths to try')
    parser.add_argument('--neutral-below', type=float, default=0.4,
                        help='report "neutral" when the top probability is below this')
    parser.add_argument('--class-weight', choices=('none', 'balanced'), default='none',
                        help='"balanced" helps rare emotions such as surprise')
    args = parser.parse_args()

    paths = {name: os.path.join(args.data, f'{name}.csv')
             for name in ('train', 'validation', 'test')}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise SystemExit('Missing ' + ', '.join(missing) +
                         '\nRun `python download_data.py` first.')
    model = train_model(read_csv(paths['train']), read_csv(paths['validation']),
                        read_csv(paths['test']), max_features=args.max_features,
                        epochs=args.epochs, lr=args.lr,
                        l2_grid=tuple(float(v) for v in args.l2.split(',')),
                        neutral_below=args.neutral_below,
                        class_weight=args.class_weight)
    save_model(model, args.out)
    print(f'Saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
