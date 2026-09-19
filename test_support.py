"""Helpers for the tests: train a tiny model on made-up sentences.

This only checks that the training and prediction code works end to end. It says
nothing about accuracy on real data: train.py reports that on the real dataset.
"""
import os
import random
import tempfile

import train

CUES = {
    'sadness': ['sad', 'lonely', 'miserable', 'heartbroken', 'gloomy', 'crying'],
    'joy': ['happy', 'glad', 'delighted', 'thrilled', 'cheerful', 'grateful'],
    'love': ['adore', 'affectionate', 'tender', 'caring', 'devoted', 'fond'],
    'anger': ['furious', 'mad', 'irritated', 'outraged', 'annoyed', 'angry'],
    'fear': ['terrified', 'afraid', 'scared', 'anxious', 'nervous', 'worried'],
    'surprise': ['amazed', 'shocked', 'stunned', 'astonished', 'startled', 'surprised'],
}
FILLERS = ['today', 'lately', 'about this', 'right now', 'again', 'so much',
           'because of everything', 'this morning', 'at work', 'at home']


def make_sentences(per_class, seed):
    """Return (texts, labels) of simple 'i feel <cue> <filler>' sentences."""
    rng = random.Random(seed)
    texts, labels = [], []
    for label, cues in CUES.items():
        for _ in range(per_class):
            texts.append(f'i feel {rng.choice(cues)} {rng.choice(FILLERS)}')
            labels.append(label)
    return texts, labels


_CACHE = {}


def tiny_model_path():
    """Train once per test run and return the path of the model file."""
    if 'path' not in _CACHE:
        folder = tempfile.mkdtemp(prefix='tiny_model_')
        model = train.train_model(make_sentences(120, 1), make_sentences(30, 2),
                                  make_sentences(30, 3), max_features=500,
                                  l2_grid=(1e-6,), epochs=15, log=lambda *_: None)
        path = os.path.join(folder, 'model.json')
        train.save_model(model, path)
        _CACHE['path'] = path
        _CACHE['model'] = model
    return _CACHE['path']


def tiny_model():
    """The model dictionary that tiny_model_path() saved."""
    tiny_model_path()
    return _CACHE['model']
