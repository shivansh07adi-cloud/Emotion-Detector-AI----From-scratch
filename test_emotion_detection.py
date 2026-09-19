import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import test_support
import train
from EmotionDetection import emotion_detection as ed
from EmotionDetection.emotion_detection import emotion_detector

LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {'MODEL_PATH': test_support.tiny_model_path()})
        env.start()
        self.addCleanup(env.stop)


class TestFeatures(unittest.TestCase):
    def test_words_are_lowercased_and_apostrophes_dropped(self):
        self.assertEqual(ed.tokenize("I DON'T feel it, I\u2019m fine!"),
                         ['i', 'dont', 'feel', 'it', 'im', 'fine'])

    def test_negation_marks_the_next_words(self):
        marked = ed.mark_negation(ed.tokenize("i am not really happy but i am glad"))
        self.assertIn('not_happy', marked)
        self.assertIn('glad', marked)
        self.assertNotIn('not_glad', marked)

    def test_features_include_word_pairs(self):
        features = ed.extract_features('i feel sad')
        self.assertTrue({'i', 'feel', 'sad', 'i_feel', 'feel_sad'} <= set(features))


class TestPrediction(ModelTestCase):
    def test_each_emotion_is_recognised(self):
        for label, cues in test_support.CUES.items():
            result = emotion_detector(f'i feel {cues[0]} today')
            self.assertEqual(result['dominant_emotion'], label)

    def test_result_shape_and_probabilities(self):
        result = emotion_detector('I am so happy about this')
        self.assertEqual(set(result), set(LABELS) | {'dominant_emotion'})
        scores = [result[label] for label in LABELS]
        self.assertAlmostEqual(sum(scores), 1.0, places=2)
        self.assertEqual(result['dominant_emotion'], LABELS[scores.index(max(scores))])

    def test_empty_or_invalid_input(self):
        for bad in ('', '   ', None, 42):
            result = emotion_detector(bad)
            self.assertIsNone(result['dominant_emotion'])
            self.assertIsNone(result['joy'])

    def test_unknown_words_are_neutral(self):
        result = emotion_detector('zzz qqq xxx')
        self.assertEqual(result['dominant_emotion'], 'neutral')
        self.assertEqual(result['joy'], 0.0)

    def test_low_confidence_is_neutral(self):
        model = dict(test_support.tiny_model(), neutral_below=0.999999)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'strict.json')
            train.save_model(model, path)
            with mock.patch.dict(os.environ, {'MODEL_PATH': path}):
                result = emotion_detector('i feel happy today')
        self.assertEqual(result['dominant_emotion'], 'neutral')
        self.assertGreater(result['joy'], 0.5)

    def test_model_info(self):
        info = ed.model_info()
        self.assertEqual(info['labels'], LABELS)
        self.assertEqual(info['train_docs'], 720)
        self.assertIn('test_accuracy', info)


class TestModelFile(unittest.TestCase):
    def test_missing_model_is_reported_clearly(self):
        with mock.patch.dict(os.environ, {'MODEL_PATH': '/nonexistent/model.json'}):
            self.assertFalse(ed.model_ready())
            self.assertIsNone(ed.model_info())
            with self.assertRaises(ed.ModelNotFoundError):
                emotion_detector('i feel happy')
            self.assertIsNone(emotion_detector('')['dominant_emotion'])

    def test_corrupt_model_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'bad.json')
            with open(path, 'w', encoding='utf-8') as bad:
                json.dump({'labels': ['a', 'b'], 'bias': [0.0], 'terms': {}}, bad)
            with mock.patch.dict(os.environ, {'MODEL_PATH': path}):
                self.assertFalse(ed.model_ready())


class TestTraining(unittest.TestCase):
    def test_learns_held_out_data(self):
        meta = test_support.tiny_model()['meta']
        self.assertGreater(meta['test_accuracy'], 0.95)
        self.assertGreater(meta['test_macro_f1'], 0.95)

    def test_numpy_and_plain_python_predictions_agree(self):
        model = test_support.tiny_model()
        features = sorted(model['terms'])
        index = {feature: i for i, feature in enumerate(features)}
        idf = np.array([model['terms'][f][0] for f in features])
        weights = np.array([model['terms'][f][1:] for f in features])
        bias = np.array(model['bias'])
        texts = ['i feel happy today', 'i feel scared about this', 'i feel amazed again',
                 'i feel not sad right now']
        matrix = train.vectorize([ed.extract_features(t) for t in texts], index, idf)
        expected = train.predict_proba(matrix, weights, bias)
        for text, row in zip(texts, expected):
            actual = ed._probabilities(model, text)
            np.testing.assert_allclose(actual, row, atol=1e-9)

    def test_rows_are_unit_length(self):
        docs = [ed.extract_features(t) for t in ('i feel sad', 'i feel so very happy')]
        index, idf = train.build_vocabulary(docs, 100, 1)
        indptr, indices, data = train.vectorize(docs, index, idf)
        for row in range(2):
            values = data[indptr[row]:indptr[row + 1]]
            self.assertAlmostEqual(float(np.sqrt((values ** 2).sum())), 1.0)
        rows, cols, vals = train.csr_rows((indptr, indices, data), np.array([1]))
        self.assertEqual(len(rows), indptr[2] - indptr[1])
        self.assertTrue((rows == 0).all())

    def test_gradient_matches_numerical_gradient(self):
        texts, labels = test_support.make_sentences(4, 7)
        docs = [ed.extract_features(t) for t in texts]
        index, idf = train.build_vocabulary(docs, 60, 1)
        matrix = train.vectorize(docs, index, idf)
        label_id = {label: i for i, label in enumerate(LABELS)}
        y = np.array([label_id[label] for label in labels])
        rng = np.random.default_rng(0)
        weights = rng.normal(0, 0.3, (len(index), 6))
        bias = rng.normal(0, 0.3, 6)
        sample_weight = rng.uniform(0.5, 1.5, len(y))
        rows, l2 = np.arange(len(y)), 0.01

        def loss(w, b):
            probs = train.predict_proba(matrix, w, b)
            nll = -np.log(probs[np.arange(len(y)), y])
            return (sample_weight * nll).sum() / len(y) + 0.5 * l2 * (w ** 2).sum()

        grad_w, grad_b = train.batch_gradient(matrix, rows, y, sample_weight,
                                              weights, bias, l2)
        step = 1e-6
        for _ in range(25):
            i, k = rng.integers(len(index)), rng.integers(6)
            plus, minus = weights.copy(), weights.copy()
            plus[i, k] += step
            minus[i, k] -= step
            numeric = (loss(plus, bias) - loss(minus, bias)) / (2 * step)
            self.assertAlmostEqual(grad_w[i, k], numeric, places=6)
        for k in range(6):
            plus, minus = bias.copy(), bias.copy()
            plus[k] += step
            minus[k] -= step
            numeric = (loss(weights, plus) - loss(weights, minus)) / (2 * step)
            self.assertAlmostEqual(grad_b[k], numeric, places=6)

    def test_unknown_validation_label_is_rejected(self):
        texts, labels = test_support.make_sentences(5, 1)
        with self.assertRaises(ValueError):
            train.train_model((texts, labels), (['i feel sad'], ['disgust']),
                              epochs=1, l2_grid=(1e-6,), log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
