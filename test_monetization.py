import hashlib
import hmac
import os
import tempfile
import unittest
from unittest import mock

import test_support

SECRET = 'test_secret_for_hmac'


def _signature(order_id, payment_id, secret=SECRET):
    return hmac.new(secret.encode(), f'{order_id}|{payment_id}'.encode(),
                    hashlib.sha256).hexdigest()


class MonetizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {
            'MODEL_PATH': test_support.tiny_model_path(),
            'DATABASE_PATH': os.path.join(self.tmp.name, 'test.db'),
            'SECRET_KEY': 'test-secret-key',
            'FREE_DAILY_LIMIT': '3',
            'FREE_MAX_CHARS': '50',
            'PRO_MAX_CHARS': '500',
            'FREE_IP_MULTIPLIER': '2',
            'RAZORPAY_KEY_ID': 'rzp_test_key',
            'RAZORPAY_KEY_SECRET': SECRET,
        })
        self.env.start()
        import server
        self.app = server.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def analyze(self, text, client=None):
        return (client or self.client).post('/emotionDetector?format=json',
                                            json={'textToAnalyze': text})

    def buy_pro(self, client=None):
        client = client or self.client
        fake = mock.Mock(status_code=200)
        fake.json.return_value = {'id': 'order_ABC123'}
        with mock.patch('monetization.requests.post', return_value=fake) as post:
            order = client.post('/api/create-order').get_json()
        self.assertEqual(order['order_id'], 'order_ABC123')
        self.assertEqual(order['amount'], 14900)
        self.assertEqual(post.call_args.kwargs['auth'], ('rzp_test_key', SECRET))
        return client.post('/api/verify-payment', json={
            'razorpay_order_id': 'order_ABC123',
            'razorpay_payment_id': 'pay_XYZ789',
            'razorpay_signature': _signature('order_ABC123', 'pay_XYZ789')})

    # -- free tier ---------------------------------------------------------
    def test_original_text_response_is_unchanged(self):
        body = self.client.get('/emotionDetector?textToAnalyze=I am glad').get_data(as_text=True)
        self.assertIn("For the given statement, the system response is", body)
        self.assertIn("'joy': ", body)
        self.assertIn('<strong>joy</strong>', body)

    def test_invalid_text(self):
        self.assertEqual(self.client.get('/emotionDetector?textToAnalyze=').get_data(as_text=True),
                         'Invalid text! Please try again')
        self.assertEqual(self.analyze('   ').status_code, 400)

    def test_free_quota_then_upgrade_prompt(self):
        for _ in range(3):
            self.assertEqual(self.analyze('I am happy').status_code, 200)
        blocked = self.analyze('I am happy')
        self.assertEqual(blocked.status_code, 402)
        self.assertTrue(blocked.get_json()['upgrade'])
        status = self.client.get('/api/status').get_json()
        self.assertEqual(status['remaining'], 0)

    def test_invalid_input_does_not_use_quota(self):
        self.analyze('')
        self.assertEqual(self.client.get('/api/status').get_json()['remaining'], 3)

    def test_clearing_cookies_does_not_reset_quota_forever(self):
        for _ in range(6):
            self.analyze('I am happy', client=self.app.test_client())
        self.assertEqual(self.analyze('I am happy', client=self.app.test_client()).status_code, 402)

    def test_length_limit_by_plan(self):
        response = self.analyze('happy ' * 20)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.get('/api/status').get_json()['remaining'], 3)

    # -- paying ------------------------------------------------------------
    def test_purchase_unlocks_pro(self):
        result = self.buy_pro()
        self.assertEqual(result.status_code, 200)
        key = result.get_json()['license_key']
        self.assertRegex(key, r'^EMO-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$')
        for _ in range(5):
            self.assertEqual(self.analyze('I am happy').status_code, 200)
        self.assertEqual(self.analyze('happy ' * 20).status_code, 200)
        status = self.client.get('/api/status').get_json()
        self.assertEqual(status['tier'], 'pro')
        self.assertIsNone(status['remaining'])

    def test_bad_signature_is_rejected(self):
        fake = mock.Mock(status_code=200)
        fake.json.return_value = {'id': 'order_BAD'}
        with mock.patch('monetization.requests.post', return_value=fake):
            self.client.post('/api/create-order')
        response = self.client.post('/api/verify-payment', json={
            'razorpay_order_id': 'order_BAD', 'razorpay_payment_id': 'pay_1',
            'razorpay_signature': _signature('order_BAD', 'pay_1', secret='wrong')})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get('/api/status').get_json()['tier'], 'free')

    def test_signed_payment_for_unknown_order_is_rejected(self):
        response = self.client.post('/api/verify-payment', json={
            'razorpay_order_id': 'order_NEVER_CREATED', 'razorpay_payment_id': 'pay_1',
            'razorpay_signature': _signature('order_NEVER_CREATED', 'pay_1')})
        self.assertEqual(response.status_code, 404)

    def test_replayed_verification_returns_same_key(self):
        first = self.buy_pro().get_json()['license_key']
        again = self.client.post('/api/verify-payment', json={
            'razorpay_order_id': 'order_ABC123', 'razorpay_payment_id': 'pay_XYZ789',
            'razorpay_signature': _signature('order_ABC123', 'pay_XYZ789')})
        self.assertEqual(again.get_json()['license_key'], first)

    def test_license_key_works_on_another_browser(self):
        key = self.buy_pro().get_json()['license_key']
        other = self.app.test_client()
        self.assertEqual(other.post('/api/activate', json={'license_key': 'EMO-NOPE'}).status_code, 404)
        self.assertEqual(other.post('/api/activate', json={'license_key': key.lower()}).status_code, 200)
        self.assertEqual(other.get('/api/status').get_json()['tier'], 'pro')

    def test_expired_license_stops_working(self):
        self.buy_pro()
        with mock.patch('monetization.time.time', return_value=10 ** 11):
            self.assertEqual(self.client.get('/api/status').get_json()['tier'], 'free')

    def test_payments_disabled_without_keys(self):
        with mock.patch.dict(os.environ, {'RAZORPAY_KEY_ID': '', 'RAZORPAY_KEY_SECRET': ''}):
            self.assertEqual(self.client.post('/api/create-order').status_code, 503)
            self.assertFalse(self.client.get('/api/status').get_json()['payments_enabled'])

    def test_razorpay_failure_is_reported(self):
        fake = mock.Mock(status_code=401)
        fake.json.return_value = {'error': {}}
        with mock.patch('monetization.requests.post', return_value=fake):
            self.assertEqual(self.client.post('/api/create-order').status_code, 502)

    def test_missing_model_gives_a_clear_error_and_keeps_quota(self):
        with mock.patch.dict(os.environ, {'MODEL_PATH': '/nonexistent/model.json'}):
            response = self.analyze('I am happy')
            self.assertEqual(response.status_code, 503)
            self.assertIn('python train.py', response.get_json()['error'])
            self.assertEqual(self.client.get('/api/model').status_code, 503)
        self.assertEqual(self.client.get('/api/status').get_json()['remaining'], 3)

    def test_model_details_endpoint(self):
        info = self.client.get('/api/model').get_json()
        self.assertEqual(info['labels'], ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
        self.assertIn('test_accuracy', info)

    def test_json_scores_cover_all_six_emotions(self):
        data = self.analyze('I am so happy').get_json()
        self.assertEqual(list(data['scores']),
                         ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])
        self.assertNotIn('dominant_emotion', data['scores'])

    def test_index_page_renders(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'NLP - Emotion Detection', page.data)
        self.assertEqual(self.client.get('/mywebscript.js').status_code, 200)


class DemoModeTests(unittest.TestCase):
    """On Vercel there is no persistent disk: payments are off and a small
    cookie-counted trial (5 analyses by default) applies instead."""

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            'VERCEL': '1',
            'MODEL_PATH': test_support.tiny_model_path(),
            'DATABASE_PATH': '/nonexistent-folder/should-never-be-opened.db',
            'RAZORPAY_KEY_ID': 'rzp_test_key',
            'RAZORPAY_KEY_SECRET': SECRET,
        })
        self.env.start()
        os.environ.pop('MONETIZATION_ENABLED', None)
        os.environ.pop('TRIAL_LIMIT', None)
        import server
        self.app = server.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def tearDown(self):
        self.env.stop()

    def analyze(self, text='I am so happy', client=None):
        return (client or self.client).post('/emotionDetector?format=json',
                                            json={'textToAnalyze': text})

    def test_five_free_analyses_then_blocked_without_a_database(self):
        for used in range(1, 6):
            response = self.analyze()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['status']['remaining'], 5 - used)
        blocked = self.analyze()
        self.assertEqual(blocked.status_code, 402)
        self.assertIn('5 free analyses', blocked.get_json()['error'])
        self.assertFalse(blocked.get_json()['upgrade'])
        status = self.client.get('/api/status').get_json()
        self.assertEqual((status['tier'], status['remaining'], status['trial_limit']),
                         ('trial', 0, 5))
        self.assertFalse(status['payments_enabled'])

    def test_a_new_browser_gets_its_own_trial(self):
        for _ in range(5):
            self.analyze()
        self.assertEqual(self.analyze().status_code, 402)
        self.assertEqual(self.analyze(client=self.app.test_client()).status_code, 200)

    def test_invalid_text_does_not_use_the_trial(self):
        self.analyze('')
        self.analyze('   ')
        self.assertEqual(self.client.get('/api/status').get_json()['remaining'], 5)

    def test_original_text_response_also_counts(self):
        for _ in range(5):
            self.client.get('/emotionDetector?textToAnalyze=i feel happy')
        body = self.client.get('/emotionDetector?textToAnalyze=i feel happy').get_data(as_text=True)
        self.assertIn('Thanks for trying the demo', body)

    def test_limit_is_configurable_and_zero_means_unlimited(self):
        with mock.patch.dict(os.environ, {'TRIAL_LIMIT': '2'}):
            client = self.app.test_client()
            self.assertEqual([self.analyze(client=client).status_code for _ in range(3)],
                             [200, 200, 402])
        with mock.patch.dict(os.environ, {'TRIAL_LIMIT': '0'}):
            client = self.app.test_client()
            self.assertTrue(all(self.analyze(client=client).status_code == 200 for _ in range(12)))
            self.assertEqual(client.get('/api/status').get_json()['tier'], 'demo')

    def test_a_garbled_cookie_is_treated_as_unused(self):
        self.client.set_cookie('emo_trial', 'not-a-number')
        self.assertEqual(self.analyze().status_code, 200)
        self.client.set_cookie('emo_trial', '-7')
        self.assertEqual(self.client.get('/api/status').get_json()['remaining'], 5)

    def test_payment_routes_are_off(self):
        self.assertEqual(self.client.post('/api/create-order').status_code, 503)
        self.assertEqual(self.client.post('/api/verify-payment', json={}).status_code, 503)
        self.assertEqual(self.client.post('/api/activate', json={'license_key': 'x'}).status_code, 503)

    def test_can_be_switched_on_explicitly(self):
        with mock.patch.dict(os.environ, {'MONETIZATION_ENABLED': '1'}):
            import monetization
            self.assertTrue(monetization.enabled())

    def test_page_and_script_are_served(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/mywebscript.js').status_code, 200)


if __name__ == "__main__":
    unittest.main()
