"""
Emotion detection module.

Predicts the emotion in a piece of text with a classifier that was trained from
scratch by ``train.py`` (TF-IDF features + softmax regression, written with
plain NumPy). The learned weights live in ``model.json`` next to this file, and
prediction below uses only the Python standard library, so the web app stays
tiny and quick to deploy.

The result is a dictionary with one score per emotion plus a ``dominant_emotion``.
"""
import json
import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache

DEFAULT_LABELS = ('sadness', 'joy', 'love', 'anger', 'fear', 'surprise')
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'model.json')
MODEL_MISSING_MESSAGE = ("The emotion model has not been trained yet. "
                         "Run `python train.py`, then restart the server.")

logger = logging.getLogger(__name__)

_NEGATORS = {
    'not', 'no', 'never', 'cannot', 'cant', 'dont', 'didnt', 'doesnt',
    'isnt', 'wasnt', 'arent', 'werent', 'wont', 'wouldnt', 'couldnt',
    'shouldnt', 'havent', 'hasnt', 'hadnt', 'aint', 'without', 'hardly',
    'neither', 'nor',
}
_CLAUSE_BREAKS = {'but', 'however', 'though', 'although', 'while', 'yet'}
_NEGATION_WINDOW = 3


class ModelNotFoundError(RuntimeError):
    """Raised when model.json does not exist yet."""


# --------------------------------------------------------------------------
# Text -> features. train.py imports these too, so training and prediction
# always see exactly the same features.
# --------------------------------------------------------------------------
def tokenize(text):
    """Lowercase the text and split it into plain words (apostrophes dropped)."""
    cleaned = text.lower().replace("'", "").replace("\u2019", "")
    return re.findall(r"[a-z]+", cleaned)


def mark_negation(tokens):
    """Prefix the words right after a negator with 'not_' ("not happy" -> not_happy)."""
    marked = []
    remaining = 0
    for token in tokens:
        if token in _CLAUSE_BREAKS:
            remaining = 0
            marked.append(token)
        elif token in _NEGATORS:
            remaining = _NEGATION_WINDOW
            marked.append(token)
        elif remaining > 0:
            marked.append('not_' + token)
            remaining -= 1
        else:
            marked.append(token)
    return marked


def extract_features(text):
    """Turn text into a list of word and word-pair features."""
    tokens = mark_negation(tokenize(text))
    pairs = [f'{first}_{second}' for first, second in zip(tokens, tokens[1:])]
    return tokens + pairs


# --------------------------------------------------------------------------
# Loading the trained model
# --------------------------------------------------------------------------
def _model_path():
    """Where model.json lives (override with the MODEL_PATH env variable)."""
    return os.environ.get('MODEL_PATH') or DEFAULT_MODEL_PATH


@lru_cache(maxsize=8)
def _read_model(path):
    """Read and sanity-check a model file (cached per path)."""
    with open(path, encoding='utf-8') as model_file:
        model = json.load(model_file)
    labels, bias, terms = model['labels'], model['bias'], model['terms']
    if not labels or len(bias) != len(labels):
        raise ValueError('model.json is corrupt: labels and bias do not match')
    for entry in terms.values():
        if len(entry) != len(labels) + 1:
            raise ValueError('model.json is corrupt: bad term weights')
    return model


def load_model(path=None):
    """Return the trained model, or raise ModelNotFoundError if it is missing."""
    path = path or _model_path()
    if not os.path.isfile(path):
        raise ModelNotFoundError(MODEL_MISSING_MESSAGE)
    return _read_model(path)


def model_ready():
    """True if a usable model.json can be loaded."""
    try:
        load_model()
    except (ModelNotFoundError, ValueError, KeyError, TypeError, OSError):
        logger.exception('The emotion model is not usable')
        return False
    return True


def model_info():
    """Summary of the trained model for display, or None if it is not usable."""
    if not model_ready():
        return None
    model = load_model()
    info = dict(model.get('meta', {}))
    info['labels'] = list(model['labels'])
    return info


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _empty_result(value, labels=DEFAULT_LABELS):
    """Build a result dictionary with every field set to the same value."""
    result = {label: value for label in labels}
    result['dominant_emotion'] = value
    return result


def _softmax(logits):
    """Turn raw scores into probabilities that add up to 1."""
    top = max(logits)
    exps = [math.exp(logit - top) for logit in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _probabilities(model, text):
    """Softmax probabilities for the text, or None if no feature is known."""
    terms = model['terms']
    counts = Counter(feature for feature in extract_features(text)
                     if feature in terms)
    if not counts:
        return None
    weighted = []
    for feature, count in counts.items():
        entry = terms[feature]
        weighted.append(((1.0 + math.log(count)) * entry[0], entry))
    norm = math.sqrt(sum(value * value for value, _ in weighted))
    logits = list(model['bias'])
    for value, entry in weighted:
        scaled = value / norm
        for k, _ in enumerate(logits):
            logits[k] += scaled * entry[k + 1]
    return _softmax(logits)


def _dominant(model, probabilities):
    """Name of the most likely emotion, or 'neutral' if the model is unsure."""
    best = max(range(len(probabilities)), key=lambda k: probabilities[k])
    if probabilities[best] < model.get('neutral_below', 0.0):
        return 'neutral'
    return model['labels'][best]


def emotion_detector(text_to_analyze):
    """
    Analyzes text and returns emotion scores and dominant emotion.

    Scores are probabilities between 0 and 1 that add up to 1. If none of the
    words are known to the model, or the model is not confident about any
    emotion, the dominant emotion is 'neutral'. Empty or non-text input
    returns None for every field.

    Raises ModelNotFoundError if ``train.py`` has not been run yet.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result(None)

    model = load_model()
    probabilities = _probabilities(model, text_to_analyze)
    if probabilities is None:
        result = _empty_result(0.0, model['labels'])
        result['dominant_emotion'] = 'neutral'
        return result

    result = {label: round(prob, 4)
              for label, prob in zip(model['labels'], probabilities)}
    result['dominant_emotion'] = _dominant(model, probabilities)
    return result
