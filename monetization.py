'''Monetization for the Emotion Detector.

Free visitors get a daily quota of analyses. A one-time Pro pass (paid through
Razorpay Checkout) removes the quota and raises the text length limit. Paying
creates a license key, which unlocks Pro on the buyer's browser and can be typed
into any other browser through "I have a license key".

Everything is configured with environment variables (see .env.example) and
stored in a small SQLite file, so there is nothing extra to install or host.
'''
import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import requests
from flask import Blueprint, current_app, jsonify, request, session
from werkzeug.middleware.proxy_fix import ProxyFix

billing = Blueprint('billing', __name__)
logger = logging.getLogger(__name__)

RAZORPAY_ORDERS_URL = 'https://api.razorpay.com/v1/orders'
KEY_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'

SCHEMA = '''
CREATE TABLE IF NOT EXISTS usage (
    scope TEXT NOT NULL,
    subject TEXT NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scope, subject, day)
);
CREATE TABLE IF NOT EXISTS licenses (
    key TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    amount INTEGER NOT NULL,
    status TEXT NOT NULL,
    payment_id TEXT,
    license_key TEXT,
    created_at INTEGER NOT NULL
);
'''


def _int_env(name, default):
    '''Read a positive integer setting from the environment.'''
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


def settings():
    '''Return the current monetization settings (read fresh from the env).'''
    return {
        'free_daily_limit': _int_env('FREE_DAILY_LIMIT', 10),
        'free_max_chars': _int_env('FREE_MAX_CHARS', 500),
        'pro_max_chars': _int_env('PRO_MAX_CHARS', 5000),
        'price_inr': _int_env('PRO_PRICE_INR', 149),
        'pro_days': _int_env('PRO_DAYS', 30),
        'ip_multiplier': _int_env('FREE_IP_MULTIPLIER', 5),
        'tz_offset_minutes': _int_env('USAGE_TZ_OFFSET_MINUTES', 330),
    }


def enabled():
    '''Whether quotas and payments are on.

    They are off by default on Vercel, which has no persistent disk for the
    database. Set MONETIZATION_ENABLED=1 or 0 to override either way.
    '''
    flag = os.environ.get('MONETIZATION_ENABLED')
    if flag is not None:
        return flag.strip() == '1'
    return not os.environ.get('VERCEL')


def _razorpay_keys():
    '''Return (key_id, key_secret) or None when payments are not configured.'''
    key_id = os.environ.get('RAZORPAY_KEY_ID', '').strip()
    key_secret = os.environ.get('RAZORPAY_KEY_SECRET', '').strip()
    if key_id and key_secret:
        return key_id, key_secret
    return None


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
def _db_path():
    '''Where the SQLite file lives.'''
    return (os.environ.get('DATABASE_PATH')
            or os.path.join(current_app.root_path, 'emotion_detector.db'))


@contextmanager
def _connect():
    '''Open a connection (autocommit) and make sure the tables exist.'''
    conn = sqlite3.connect(_db_path(), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()


@contextmanager
def _transaction(conn):
    '''Run a block inside one write transaction, so counters never race.'''
    conn.execute('BEGIN IMMEDIATE')
    try:
        yield
    except Exception:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')


def _usage_day():
    '''Today's date for quota purposes (resets at local midnight, IST default).'''
    offset = timedelta(minutes=settings()['tz_offset_minutes'])
    return (datetime.now(timezone.utc) + offset).strftime('%Y-%m-%d')


# --------------------------------------------------------------------------
# Who is asking, and what plan do they have?
# --------------------------------------------------------------------------
def _client_id():
    '''Random id stored in the signed session cookie.'''
    cid = session.get('cid')
    if not cid:
        cid = secrets.token_hex(16)
        session['cid'] = cid
        session.permanent = True
    return cid


def _client_ip():
    '''Caller's IP address (set TRUST_PROXY=1 when running behind a proxy).'''
    return request.remote_addr or 'unknown'


def _active_license(conn):
    '''Return the session's license row if it exists and has not expired.'''
    key = session.get('license')
    if not key:
        return None
    row = conn.execute(
        'SELECT key, expires_at FROM licenses WHERE key = ? AND expires_at > ?',
        (key, int(time.time()))).fetchone()
    if row is None:
        session.pop('license', None)
    return row


def _used_today(conn, scope, subject):
    '''How many free analyses this browser or IP has used today.'''
    row = conn.execute(
        'SELECT count FROM usage WHERE scope = ? AND subject = ? AND day = ?',
        (scope, subject, _usage_day())).fetchone()
    return row['count'] if row else 0


def get_status():
    '''Describe the caller's plan: tier, remaining quota, limits.'''
    conf = settings()
    if not enabled():
        return {'tier': 'demo', 'daily_limit': None, 'remaining': None,
                'max_chars': conf['pro_max_chars'], 'expires_at': None}
    with _connect() as conn:
        lic = _active_license(conn)
        if lic is not None:
            return {'tier': 'pro', 'daily_limit': None, 'remaining': None,
                    'max_chars': conf['pro_max_chars'],
                    'expires_at': lic['expires_at']}
        by_browser = conf['free_daily_limit'] - _used_today(
            conn, 'cid', _client_id())
        by_ip = conf['free_daily_limit'] * conf['ip_multiplier'] - _used_today(
            conn, 'ip', _client_ip())
    return {'tier': 'free', 'daily_limit': conf['free_daily_limit'],
            'remaining': max(0, min(by_browser, by_ip)),
            'max_chars': conf['free_max_chars'], 'expires_at': None}


def consume_analysis():
    '''Spend one analysis. Returns (allowed, status); Pro is never metered.'''
    if not enabled():
        return True, get_status()
    conf = settings()
    with _connect() as conn:
        if _active_license(conn) is not None:
            return True, get_status()
        cid, ip_addr, day = _client_id(), _client_ip(), _usage_day()
        limits = (('cid', cid, conf['free_daily_limit']),
                  ('ip', ip_addr,
                   conf['free_daily_limit'] * conf['ip_multiplier']))
        with _transaction(conn):
            if any(_used_today(conn, scope, subject) >= limit
                   for scope, subject, limit in limits):
                allowed = False
            else:
                allowed = True
                for scope, subject, _limit in limits:
                    conn.execute(
                        'INSERT OR IGNORE INTO usage (scope, subject, day, count) '
                        'VALUES (?, ?, ?, 0)', (scope, subject, day))
                    conn.execute(
                        'UPDATE usage SET count = count + 1 '
                        'WHERE scope = ? AND subject = ? AND day = ?',
                        (scope, subject, day))
    return allowed, get_status()


def init_app(app):
    '''Wire sessions, proxy handling and the billing routes into the Flask app.'''
    secret = os.environ.get('SECRET_KEY')
    if not secret:
        logger.warning('SECRET_KEY is not set: using a temporary key, so '
                       'visitors lose their session whenever the server restarts.')
        secret = secrets.token_hex(32)
    app.secret_key = secret
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=os.environ.get('SESSION_COOKIE_SECURE') == '1',
        PERMANENT_SESSION_LIFETIME=timedelta(days=365),
    )
    if os.environ.get('TRUST_PROXY') == '1':
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.register_blueprint(billing)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@billing.route('/api/status')
def api_status():
    '''Plan, remaining quota and pricing, for the page to display.'''
    conf = settings()
    payload = get_status()
    payload.update({'payments_enabled': enabled() and _razorpay_keys() is not None,
                    'price_inr': conf['price_inr'],
                    'pro_days': conf['pro_days'],
                    'free_max_chars': conf['free_max_chars'],
                    'pro_max_chars': conf['pro_max_chars']})
    return jsonify(payload)


@billing.route('/api/create-order', methods=['POST'])
def api_create_order():
    '''Create a Razorpay order for one Pro pass.'''
    keys = _razorpay_keys()
    if not enabled() or keys is None:
        return jsonify(error='Payments are not enabled on this server yet.'), 503
    conf = settings()
    amount = conf['price_inr'] * 100  # Razorpay amounts are in paise
    try:
        response = requests.post(
            RAZORPAY_ORDERS_URL, auth=keys, timeout=10,
            json={'amount': amount, 'currency': 'INR',
                  'receipt': 'emo_' + secrets.token_hex(6)})
        order_id = response.json()['id'] if response.status_code == 200 else None
    except (requests.RequestException, ValueError, KeyError):
        order_id = None
    if not order_id:
        logger.error('Razorpay order creation failed')
        return jsonify(error='Could not start the payment. Please try again.'), 502
    with _connect() as conn:
        conn.execute(
            'INSERT INTO orders (order_id, amount, status, created_at) '
            "VALUES (?, ?, 'created', ?)", (order_id, amount, int(time.time())))
    return jsonify(order_id=order_id, amount=amount, currency='INR',
                   key_id=keys[0], name='Emotion Detector',
                   description=f"Pro pass, {conf['pro_days']} days")


def _new_license_key():
    '''Generate a key like EMO-7K2P-9QXA-M4TD-H8WZ.'''
    groups = [''.join(secrets.choice(KEY_ALPHABET) for _ in range(4))
              for _ in range(4)]
    return 'EMO-' + '-'.join(groups)


def _fulfil_order(order_id, payment_id):
    '''Turn a paid order into a license. Safe to call twice for one order.'''
    conf = settings()
    with _connect() as conn:
        with _transaction(conn):
            order = conn.execute('SELECT license_key FROM orders WHERE order_id = ?',
                                 (order_id,)).fetchone()
            if order is None:
                return None
            key = order['license_key']
            if not key:
                key = _new_license_key()
                now = int(time.time())
                conn.execute(
                    'INSERT INTO licenses (key, created_at, expires_at) '
                    'VALUES (?, ?, ?)', (key, now, now + conf['pro_days'] * 86400))
                conn.execute(
                    "UPDATE orders SET status = 'paid', payment_id = ?, "
                    'license_key = ? WHERE order_id = ?',
                    (payment_id, key, order_id))
        expires = conn.execute('SELECT expires_at FROM licenses WHERE key = ?',
                               (key,)).fetchone()['expires_at']
    return key, expires


@billing.route('/api/verify-payment', methods=['POST'])
def api_verify_payment():
    '''Check Razorpay's signature, then hand out the license key.'''
    keys = _razorpay_keys()
    if not enabled() or keys is None:
        return jsonify(error='Payments are not enabled on this server yet.'), 503
    data = request.get_json(silent=True) or {}
    order_id = str(data.get('razorpay_order_id') or '')
    payment_id = str(data.get('razorpay_payment_id') or '')
    signature = str(data.get('razorpay_signature') or '')
    if not (order_id and payment_id and signature):
        return jsonify(error='Missing payment details.'), 400
    expected = hmac.new(keys[1].encode(), f'{order_id}|{payment_id}'.encode(),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning('Rejected payment with a bad signature: %s', order_id)
        return jsonify(error='Payment could not be verified.'), 400
    fulfilled = _fulfil_order(order_id, payment_id)
    if fulfilled is None:
        return jsonify(error='Unknown order.'), 404
    key, expires = fulfilled
    session['license'] = key
    session.permanent = True
    return jsonify(license_key=key, expires_at=expires)


@billing.route('/api/activate', methods=['POST'])
def api_activate():
    '''Unlock Pro in this browser with a license key from an earlier payment.'''
    if not enabled():
        return jsonify(error='Licenses are not enabled on this server.'), 503
    data = request.get_json(silent=True) or {}
    key = str(data.get('license_key') or '').strip().upper()
    with _connect() as conn:
        row = conn.execute(
            'SELECT key, expires_at FROM licenses WHERE key = ? AND expires_at > ?',
            (key, int(time.time()))).fetchone()
    if row is None:
        return jsonify(error='That license key is not valid or has expired.'), 404
    session['license'] = row['key']
    session.permanent = True
    return jsonify(tier='pro', expires_at=row['expires_at'])
