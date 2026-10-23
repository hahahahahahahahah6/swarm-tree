"""Isolated storage, auth, HTTP, and CLI tests without provider traffic."""
import base64
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from swarm_tree import cli, config
from swarm_tree.web import app, auth, db


def leaf(name='assistant'):
    return dict(name=name, provider='openai', model='gpt-5', role='Help.')


class IsolatedDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = db.init_db(os.path.join(self.tmp.name, 'web.db'))
        self.secret = patch.dict(os.environ, SWARM_TREE_SECRET='test-secret')
        self.secret.start()

    def tearDown(self):
        self.secret.stop()
        self.conn.close()
        self.tmp.cleanup()


class TestDB(IsolatedDB):
    def test_users(self):
        uid = db.create_user('a@example.com', b'hash', b'salt')
        row = db.get_user_by_email('a@example.com')
        self.assertEqual((row['id'], row['pw_hash'], row['salt']), (uid, b'hash', b'salt'))
        self.assertEqual(db.get_user_by_id(uid)['email'], 'a@example.com')
        with self.assertRaises(sqlite3.IntegrityError):
            db.create_user('a@example.com', b'x', b'y')

    def test_trees(self):
        tid = db.create_tree(1, 'old', '{}')
        self.assertEqual([r['id'] for r in db.list_trees(1)], [tid])
        self.assertIsNone(db.get_tree(tid, 2))
        self.assertEqual(db.list_trees(2), [])
        self.assertEqual(db.update_tree(tid, 2, 'bad', 'bad'), 0)
        self.assertEqual(db.update_tree(tid, 1, 'new', '{"x":1}'), 1)
        self.assertEqual(db.get_tree(tid, 1)['name'], 'new')
        self.assertEqual(db.get_tree(tid, 1)['yaml_text'], '{"x":1}')
        self.assertEqual(db.delete_tree(tid, 2), 0)
        self.assertEqual(db.delete_tree(tid, 1), 1)
        self.assertIsNone(db.get_tree(tid, 1))

    def test_runs(self):
        t1, t2 = db.create_tree(1, 'one', '{}'), db.create_tree(1, 'two', '{}')
        rid = db.create_run(1, t1, 'task')
        other = db.create_run(1, t2, 'other')
        self.assertEqual(db.get_run(rid, 1)['task'], 'task')
        self.assertIsNone(db.get_run(rid, 2))
        self.assertEqual({r['id'] for r in db.list_runs(1)}, {rid, other})
        self.assertEqual([r['id'] for r in db.list_runs(1, tree_id=t1)], [rid])
        self.assertEqual(db.list_runs(2), [])
        self.assertEqual(db.update_run_status(rid, 'ok', '{"answer":"yes"}'), 1)
        row = db.get_run(rid, 1)
        self.assertEqual((row['status'], row['transcript_json']), ('ok', '{"answer":"yes"}'))
        with self.assertRaises(ValueError):
            db.create_run(2, t1, 'forbidden')

    def test_images(self):
        iid = db.create_image(1, 'openai', 'model', 'prompt', Path(self.tmp.name) / 'x.png')
        row = db.get_image(iid, 1)
        self.assertEqual((row['provider'], row['model'], row['prompt']), ('openai', 'model', 'prompt'))
        self.assertEqual(row['png_path'], str(Path(self.tmp.name) / 'x.png'))
        self.assertEqual([r['id'] for r in db.list_images(1)], [iid])
        self.assertIsNone(db.get_image(iid, 2))
        self.assertEqual(db.list_images(2), [])

    def test_keys(self):
        self.assertFalse(any(db.keys_status(1).values()))
        db.set_key(1, 'openai', 'first')
        self.assertEqual(db.get_key(1, 'openai'), 'first')
        self.assertTrue(db.keys_status(1)['openai'])
        self.assertIsNone(db.get_key(2, 'openai'))
        db.set_key(1, 'openai', 'replacement')
        self.assertEqual(db.get_key(1, 'openai'), 'replacement')
        self.assertEqual(db.delete_key(1, 'openai'), 1)
        self.assertIsNone(db.get_key(1, 'openai'))
        self.assertFalse(db.keys_status(1)['openai'])


class TestAuth(IsolatedDB):
    def test_password(self):
        salt, hashed = auth.hash_password('password')
        self.assertTrue(auth.verify_password('password', salt, hashed))
        self.assertFalse(auth.verify_password('wrong', salt, hashed))
        salt2, hashed2 = auth.hash_password('password')
        self.assertNotEqual(salt, salt2)
        self.assertNotEqual(hashed, hashed2)

    def test_session(self):
        session = auth.sign_session(42)
        self.assertEqual(auth.verify_session(session), 42)
        tampered = session[:-1] + ('0' if session[-1] != '0' else '1')
        self.assertIsNone(auth.verify_session(tampered))

    def test_expired_session(self):
        payload = f'42|{int(time.time()) - 60}'
        sig = hmac.new(auth._secret(), payload.encode('ascii'), hashlib.sha256).hexdigest()
        self.assertIsNone(auth.verify_session(f'{payload}|{sig}'))


class TestValidate(IsolatedDB):
    def test_valid(self):
        tree = leaf('root')
        tree['children'] = [leaf('child')]
        node = config.validate(app._parse_tree_text(json.dumps(tree)))
        self.assertEqual(node['children'][0]['name'], 'child')

    def test_unknown_provider(self):
        tree = leaf()
        tree['provider'] = 'unknown'
        with self.assertRaises(config.TreeError):
            config.validate(app._parse_tree_text(json.dumps(tree)))

    def test_malformed(self):
        with self.assertRaises(ValueError):
            app._parse_tree_text('{"unclosed": [')


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dbfile = os.path.join(cls.tmp.name, 'web.db')
        cls.conn = db.init_db(cls.dbfile)
        cls.secret = patch.dict(os.environ, SWARM_TREE_SECRET='http-test-secret')
        cls.secret.start()
        # Image storage is relative to cwd; redirect that root into the fixture.
        cls.cwd_patch = patch.object(app.Path, 'cwd', return_value=Path(cls.tmp.name))
        cls.cwd_patch.start()
        cls.server = app.build_app(cls.dbfile)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        db._conn().close()
        cls.cwd_patch.stop()
        cls.secret.stop()
        cls.tmp.cleanup()

    def setUp(self):
        db.init_db(self.dbfile)
        with auth._fail_lock:
            auth._fails.clear()
        self.cookies = {}
        self.email = self._testMethodName + '@example.com'

    def _req(self, method, path, body=None, headers=None, cookies=None):
        jar = self.cookies if cookies is None else cookies
        headers = dict(headers or {})
        if jar:
            headers['Cookie'] = '; '.join(f'{k}={v}' for k, v in jar.items())
        if body is not None:
            body = urlencode(body)
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            for key, value in response.getheaders():
                if key.lower() == 'set-cookie':
                    name, val = value.split(';', 1)[0].split('=', 1)
                    jar[name] = val
            return response.status, dict(response.getheaders()), data
        finally:
            conn.close()

    def _signup(self):
        status, headers, _ = self._req('POST', '/signup', dict(email=self.email, password='password123'))
        self.assertEqual(status, 303)
        self.assertIn('Set-Cookie', headers)
        return db.get_user_by_email(self.email)['id']

    def test_anonymous_dashboard(self):
        status, headers, _ = self._req('GET', '/dashboard')
        self.assertEqual(status, 303)
        self.assertEqual(headers['Location'], '/login')

    def test_signup_dashboard(self):
        self._signup()
        status, _, body = self._req('GET', '/dashboard')
        self.assertEqual(status, 200)
        self.assertIn(b'Dashboard', body)

    def test_login(self):
        salt, hashed = auth.hash_password('password123')
        db.create_user(self.email, hashed, salt)
        status, _, body = self._req('POST', '/login', dict(email=self.email, password='password123'))
        self.assertEqual(status, 200)
        self.assertIn(b'Dashboard', body)
        self.assertEqual(self._req('GET', '/dashboard')[0], 200)
        status, _, body = self._req('POST', '/login', dict(email=self.email, password='wrong'), cookies={})
        self.assertEqual(status, 200)
        self.assertIn(b'Invalid credentials', body)

    def test_create_validate_tree(self):
        self._signup()
        tree = leaf('root')
        tree['children'] = [leaf('child')]
        status, headers, _ = self._req('POST', '/trees', dict(name='My tree', yaml_text=json.dumps(tree)))
        self.assertEqual(status, 303)
        tid = int(headers['Location'].split('/')[2])
        self.assertIn(b'My tree', self._req('GET', '/dashboard')[2])
        status, _, body = self._req('POST', f'/trees/{tid}/validate', {})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'ok': True})
        tree['provider'] = 'unknown'
        status, _, body = self._req('POST', f'/trees/{tid}/validate', dict(yaml_text=json.dumps(tree)))
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)['ok'])

    def test_chat(self):
        uid = self._signup()
        tid = db.create_tree(uid, 'Single leaf', json.dumps(leaf()))
        db.set_key(uid, 'openai', 'fake-key')

        def fake(provider, model, system, user, kind=None, max_tokens=1024,
                 timeout=120, max_retries=3, key=None):
            return 'mock answer', {'input_tokens': 1, 'output_tokens': 1}, 'stop'

        with patch('swarm_tree.web.app.providers.call', side_effect=fake) as mocked:
            status, headers, _ = self._req('POST', '/chat', dict(tree_id=tid, message='hello'))
            self.assertEqual(status, 303)
            self.assertRegex(headers['Location'], r'^/runs/\d+$')
            deadline = time.monotonic() + 20
            while True:
                status, _, body = self._req('GET', headers['Location'])
                self.assertEqual(status, 200)
                if b'completed' in body or b'mock answer' in body or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            self.assertIn(b'mock answer', body)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(mocked.call_args.kwargs['key'], 'fake-key')

    def test_images(self):
        uid = self._signup()
        db.set_key(uid, 'openai', 'fake-key')
        png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
        with patch('swarm_tree.web.app.images.generate_image', return_value=(png, 'gpt-image-2.5-flare')) as mocked:
            status, headers, _ = self._req('POST', '/images', dict(provider='openai', prompt='x'))
        self.assertEqual(status, 303)
        self.assertEqual(headers['Location'], '/images')
        mocked.assert_called_once_with('openai', 'x', 'fake-key')
        iid = db.list_images(uid)[0]['id']
        status, _, body = self._req('GET', '/images')
        self.assertEqual(status, 200)
        self.assertIn(f'/images/{iid}.png'.encode(), body)
        status, headers, body = self._req('GET', f'/images/{iid}.png')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertEqual(body, png)

    def test_ownership(self):
        owner = db.create_user('owner@example.com', b'hash', b'salt')
        tid = db.create_tree(owner, 'Private', json.dumps(leaf()))
        self._signup()
        self.assertEqual(self._req('GET', f'/trees/{tid}/edit')[0], 404)

    def test_rate_limit(self):
        for _ in range(6):
            status, _, body = self._req('POST', '/login', dict(email=self.email, password='wrong'))
        self.assertEqual(status, 429)
        self.assertIn(b'Too many attempts', body)


class TestWebCLI(unittest.TestCase):
    def test_defaults(self):
        with patch('swarm_tree.web.app.serve_forever') as serve:
            self.assertEqual(cli.main(['web']), 0)
        serve.assert_called_once_with('127.0.0.1', 8080)

    def test_custom_address(self):
        with patch('swarm_tree.web.app.serve_forever') as serve:
            self.assertEqual(cli.main(['web', '--host', '0.0.0.0', '--port', '9000']), 0)
        serve.assert_called_once_with('0.0.0.0', 9000)
