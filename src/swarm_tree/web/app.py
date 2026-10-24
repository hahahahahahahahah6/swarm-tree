"""Local, stdlib HTTP application for owned trees, BYOK chat, and images."""

from functools import wraps
from html import escape
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .. import config, engine, providers
from . import auth, db, images

PROVIDERS = ('openai', 'anthropic', 'gemini', 'xai', 'deepseek')
MAX_BODY = 2 * 1024 * 1024
STUDIO = Path(__file__).resolve().parents[3] / 'studio/swarm-tree-studio.html'


def _parse_tree_text(text):
    """Parse JSON first, with optional PyYAML support."""
    try:
        return json.loads(text)
    except ValueError:
        try:
            import yaml
        except ImportError:
            raise ValueError('install pyyaml or use JSON') from None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(str(exc)) from None


def _node(text):
    return config.validate(_parse_tree_text(text))


def _js(value):
    """Encode data for an inline script without allowing HTML termination."""
    return json.dumps(value).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _page(title, body, user=None):
    nav = ('<a href="/dashboard">Dashboard</a> <a href="/images">Images</a> '
           '<a href="/settings/keys">Keys</a> '
           '<form action="/logout" method="post" style="display:inline"><button>Log out</button></form>') if user else (
               '<a href="/login">Login</a> <a href="/signup">Sign up</a>')
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>{escape(title)}</title><style>
body{{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;background:#fafafa;color:#222}}
nav,a{{margin-right:12px}}input,textarea,select,button{{font:inherit;padding:8px;margin:6px 0}}
label{{display:block}}textarea{{display:block;width:95%;min-height:180px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.card{{padding:16px;background:white;border:1px solid #ddd;margin:12px 0}}img{{max-width:240px}}.error{{color:#a00}}
</style></head><body><nav>{nav}</nav><h1>{escape(title)}</h1>{body}</body></html>'''


def require_login(fn):
    """Redirect anonymous requests before accessing owned resources."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self.user is None:
            return self.redirect('/login')
        return fn(self, *args, **kwargs)
    return wrapped


LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm Tree — grow an agent team, chat with it, ship images</title>
<style>
:root{--bg:#0a0d12;--panel:#11151d;--line:#1f2632;--txt:#e8ecf3;--mut:#9aa4b5;--acc:#4ade80;--acc2:#22d3ee}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6}
a{color:inherit}
.wrap{max-width:1080px;margin:0 auto;padding:0 22px}
nav{position:sticky;top:0;z-index:10;background:rgba(10,13,18,.85);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;height:60px}
.brand{font-weight:800;font-size:19px;letter-spacing:.3px;text-decoration:none}
.brand span{color:var(--acc)}
.btn{display:inline-block;padding:10px 22px;border-radius:10px;font-weight:700;text-decoration:none;font-size:15px;border:1px solid var(--line)}
.btn.pri{background:var(--acc);border-color:var(--acc);color:#06130b}
.btn.ghost{color:var(--txt)}
.btn.pri:hover{filter:brightness(1.08)}
.navlinks{display:flex;gap:10px;align-items:center}
.hero{padding:84px 0 60px;text-align:center}
.kicker{display:inline-block;font-size:13px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--acc);border:1px solid #1d3a2a;background:#0e1a13;padding:6px 14px;border-radius:999px;margin-bottom:22px}
h1{font-size:clamp(34px,6vw,58px);line-height:1.12;font-weight:800;letter-spacing:-1px;margin-bottom:18px}
h1 .g{background:linear-gradient(90deg,var(--acc),var(--acc2));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);font-size:clamp(16px,2.4vw,19px);max-width:640px;margin:0 auto 30px}
.cta{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:54px}
.treebox{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:26px 12px;max-width:720px;margin:0 auto;overflow-x:auto}
.feats{padding:30px 0 10px}
.feats h2,.steps h2{text-align:center;font-size:clamp(24px,4vw,32px);margin-bottom:8px;letter-spacing:-.5px}
.feats p.lead{text-align:center;color:var(--mut);margin-bottom:30px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px}
.card .ic{font-size:26px;margin-bottom:10px}
.card h3{font-size:17px;margin-bottom:8px}
.card p{color:var(--mut);font-size:14.5px}
.steps{padding:56px 0}
.sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;counter-reset:st}
.step{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;position:relative}
.step b.n{display:inline-flex;width:30px;height:30px;border-radius:50%;background:#12331f;color:var(--acc);align-items:center;justify-content:center;font-size:15px;margin-bottom:12px}
.step h3{font-size:16px;margin-bottom:8px}
.step p{color:var(--mut);font-size:14.5px}
.step code{background:#0a0d12;border:1px solid var(--line);border-radius:6px;padding:1px 7px;font-size:13px;color:var(--acc2)}
.cta2{text-align:center;padding:40px 0 70px}
.cta2 h2{font-size:clamp(24px,4vw,34px);margin-bottom:12px}
.cta2 p{color:var(--mut);margin-bottom:24px}
footer{border-top:1px solid var(--line);padding:26px 0;color:var(--mut);font-size:14px}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
footer a{color:var(--mut);text-decoration:none;margin-left:18px}
footer a:hover{color:var(--txt)}
@media(max-width:560px){.hero{padding:56px 0 40px}.navlinks .btn.ghost{display:none}}
</style>
</head>
<body>
<nav><div class="wrap">
<a class="brand" href="/">🌲 <span>Swarm</span> Tree</a>
<div class="navlinks"><a class="btn ghost" href="/login">Log in</a><a class="btn pri" href="/signup">Get started</a></div>
</div></nav>
<div class="wrap">
<section class="hero">
<div class="kicker">v0.5 &mdash; now with a web studio</div>
<h1>Grow an agent tree.<br><span class="g">Chat with it. Ship images.</span></h1>
<p class="sub">Swarm Tree turns a YAML tree of LLM agents into a living team: design it visually, talk to it in the browser, and generate images &mdash; all with your own provider keys.</p>
<div class="cta">
<a class="btn pri" href="/signup">Start free</a>
<a class="btn ghost" href="https://github.com/hahahahahahahahah6/swarm-tree">⭐ GitHub</a>
</div>
<div class="treebox">
<svg viewBox="0 0 640 300" width="100%" style="max-width:640px" role="img" aria-label="agent tree diagram">
<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#3a4456"/></marker></defs>
<g font-family="inherit" font-size="13" text-anchor="middle">
<line x1="320" y1="46" x2="180" y2="120" stroke="#3a4456" stroke-width="2" marker-end="url(#ar)"/>
<line x1="320" y1="46" x2="460" y2="120" stroke="#3a4456" stroke-width="2" marker-end="url(#ar)"/>
<line x1="180" y1="152" x2="110" y2="226" stroke="#3a4456" stroke-width="2" marker-end="url(#ar)"/>
<line x1="180" y1="152" x2="250" y2="226" stroke="#3a4456" stroke-width="2" marker-end="url(#ar)"/>
<line x1="460" y1="152" x2="460" y2="226" stroke="#3a4456" stroke-width="2" marker-end="url(#ar)"/>
<rect x="248" y="14" width="144" height="34" rx="10" fill="#12331f" stroke="#4ade80"/>
<text x="320" y="36" fill="#4ade80" font-weight="700">🎯 coordinator</text>
<rect x="108" y="120" width="144" height="34" rx="10" fill="#11151d" stroke="#3a4456"/>
<text x="180" y="142" fill="#e8ecf3">🔍 researcher</text>
<rect x="388" y="120" width="144" height="34" rx="10" fill="#11151d" stroke="#3a4456"/>
<text x="460" y="142" fill="#e8ecf3">🎨 designer</text>
<rect x="38" y="226" width="144" height="34" rx="10" fill="#11151d" stroke="#3a4456"/>
<text x="110" y="248" fill="#9aa4b5">📄 summarizer</text>
<rect x="178" y="226" width="144" height="34" rx="10" fill="#11151d" stroke="#3a4456"/>
<text x="250" y="248" fill="#9aa4b5">✅ fact-checker</text>
<rect x="388" y="226" width="144" height="34" rx="10" fill="#0e2a33" stroke="#22d3ee"/>
<text x="460" y="248" fill="#22d3ee">🖼️ image-gen</text>
</g></svg>
</div>
</section>
<section class="feats">
<h2>One tree, every superpower</h2>
<p class="lead">Design once in the visual studio. Run anywhere &mdash; CLI or web.</p>
<div class="grid">
<div class="card"><div class="ic">🌳</div><h3>Visual tree editor</h3><p>Drag nodes, set prompts, providers and budgets per node. Export validated YAML in one click.</p></div>
<div class="card"><div class="ic">💬</div><h3>Streaming chat</h3><p>Talk to any tree in the browser. Watch each agent work live, keep full run history.</p></div>
<div class="card"><div class="ic">🖼️</div><h3>Image generation</h3><p>Generate images from chat or a dedicated studio. Latest OpenAI &amp; Gemini image models.</p></div>
<div class="card"><div class="ic">⏰</div><h3>Schedules &amp; triggers</h3><p>Nodes run on cron, on file changes, or on shell triggers. Skip the LLM call when nothing changed.</p></div>
</div>
</section>
<section class="steps">
<h2>Live in three steps</h2>
<div class="sgrid" style="margin-top:26px">
<div class="step"><b class="n">1</b><h3>Create an account</h3><p>Thirty seconds. No credit card &mdash; you bring your own provider keys.</p></div>
<div class="step"><b class="n">2</b><h3>Design your tree</h3><p>Use the visual studio or paste YAML. <code>swarm-tree validate</code> checks it instantly.</p></div>
<div class="step"><b class="n">3</b><h3>Chat &amp; create</h3><p>Talk to your team, generate images, set schedules. Watch the tree work.</p></div>
</div>
</section>
<section class="cta2">
<h2>Your keys, your agents, your images.</h2>
<p>Bring-your-own-key: keys are stored encrypted-ish (0600 SQLite), never logged, never echoed back.</p>
<a class="btn pri" href="/signup" style="font-size:17px;padding:13px 34px">Get started — it's free</a>
</section>
</div>
<footer><div class="wrap">
<span>🌲 Swarm Tree v0.5.0 — stdlib-only, MIT</span>
<span><a href="https://github.com/hahahahahahahahah6/swarm-tree">GitHub</a><a href="https://dev.to/haoli">Blog</a><a href="/login">Log in</a></span>
</div></footer>
</body>
</html>
"""


class WebApp:
    """Application state shared by request threads and background workers."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.RUNS = {}
        self.lock = threading.Lock()

    def _chat_worker(self, run_id, user_id, tree_id, node, task, keymap):
        """Execute an already-authorized run; redact keys before persistence."""
        def scrub(text):
            for key in sorted(filter(None, keymap.values()), key=len, reverse=True):
                text = text.replace(key, '[redacted]')
            return text

        def scrub_json(value):
            # Redact strings before encoding so unusual keys cannot break JSON.
            def clean(item):
                if isinstance(item, str):
                    return scrub(item)
                if isinstance(item, dict):
                    return {scrub(k): clean(v) for k, v in item.items()}
                if isinstance(item, list):
                    return [clean(v) for v in item]
                return item
            return json.dumps(clean(value))

        def provider_fn(provider, model, system, user, kind, max_tokens):
            with self.lock:
                self.RUNS[run_id]['events'].append(
                    {'type': 'call', 'provider': provider, 'model': model, 'kind': kind})
            try:
                return providers.call(provider, model, system, user,
                                      max_tokens=max_tokens, key=keymap[provider],
                                      timeout=node.get('timeout', config.DEFAULT_TIMEOUT))
            except Exception as exc:
                raise RuntimeError(scrub(str(exc))) from None

        try:
            # Scope the read even though this ID was authorized before spawning.
            if db.get_run(run_id, user_id) is None:
                raise ValueError('run not found')
            record = engine.execute_tree(
                node, task, provider_fn, max_tokens=1024,
                summary_mode=node.get('summary_mode', True),
                max_concurrency=node.get('max_concurrency', config.DEFAULT_MAX_CONCURRENCY),
                timeout=node.get('timeout', config.DEFAULT_TIMEOUT))
            record = json.loads(scrub_json(record))
            status = 'ok' if record.get('status') == 'ok' else 'error'
        except Exception as exc:
            record, status = scrub(str(exc)), 'error'
        try:
            db.update_run_status(run_id, status, json.dumps({'task': task, 'root': record}))
        finally:
            with self.lock:
                self.RUNS[run_id].update(status='done' if status == 'ok' else 'error', record=record)
                self.RUNS[run_id]['events'].append({'type': 'done'})


class Handler(BaseHTTPRequestHandler):
    """Dispatch requests without ever logging request bodies or credentials."""

    def log_message(self, format, *args):
        # Even URLs may accidentally contain sensitive input.
        pass

    @property
    def app(self):
        return self.server.app

    def respond(self, body, status=200, content_type='text/html; charset=utf-8', headers=None):
        data = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def page(self, title, body, status=200, headers=None):
        self.respond(_page(title, body, self.user), status, headers=headers)

    def redirect(self, target, cookie=None):
        headers = {'Location': target}
        if cookie:
            headers['Set-Cookie'] = cookie
        self.respond(b'', 303, headers=headers)

    def json(self, value, status=200):
        self.respond(json.dumps(value), status, 'application/json; charset=utf-8')

    def form(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise ValueError('Invalid Content-Length') from None
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            raise OverflowError('Form body exceeds 2MB')
        return {k: v[-1] for k, v in parse_qs(
            self.rfile.read(length).decode('utf-8'), keep_blank_values=True).items()}

    def session_cookie(self, uid=None):
        cookie = SimpleCookie()
        cookie[auth.SESSION_COOKIE] = auth.sign_session(uid) if uid is not None else ''
        morsel = cookie[auth.SESSION_COOKIE]
        morsel['httponly'] = True
        morsel['samesite'] = 'Lax'
        morsel['path'] = '/'
        morsel['max-age'] = auth.SESSION_MAX_AGE if uid is not None else 0
        return morsel.OutputString()

    def do_GET(self):
        self.dispatch('GET')

    def do_POST(self):
        self.dispatch('POST')

    def dispatch(self, method):
        self.user = None
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
            session = cookie.get(auth.SESSION_COOKIE)
            uid = auth.verify_session(session.value) if session else None
            self.user = db.get_user_by_id(uid) if uid is not None else None
        except CookieError:
            pass
        url = urlsplit(self.path)
        self.query = parse_qs(url.query)
        path = url.path
        try:
            self.fields = self.form() if method == 'POST' else {}
            routes = {
                ('GET', '/'): self.home,
                ('GET', '/login'): self.auth_form, ('GET', '/signup'): self.auth_form,
                ('POST', '/login'): self.authenticate, ('POST', '/signup'): self.authenticate,
                ('POST', '/logout'): self.logout, ('GET', '/dashboard'): self.dashboard,
                ('GET', '/trees/new'): self.new_tree, ('POST', '/trees'): self.create_tree,
                ('GET', '/chat'): self.chat, ('POST', '/chat'): self.start_chat,
                ('GET', '/images'): self.image_page, ('POST', '/images'): self.generate_image,
                ('GET', '/settings/keys'): self.keys, ('POST', '/settings/keys'): self.keys,
            }
            self.route_path = path
            if (method, path) in routes:
                return routes[method, path]()
            match = re.fullmatch(r'/trees/(\d+)(?:/(edit|delete|validate))?', path)
            if match and (method, match[2]) in {('GET', 'edit'), ('POST', None), ('POST', 'delete'), ('POST', 'validate')}:
                return self.tree(int(match[1]), match[2])
            match = re.fullmatch(r'/runs/(\d+)(/events)?', path)
            if method == 'GET' and match:
                return self.run(int(match[1]), bool(match[2]))
            match = re.fullmatch(r'/images/(\d+)\.png', path)
            if method == 'GET' and match:
                return self.image_file(int(match[1]))
            self.page('Not found', 'Not found', 404)
        except OverflowError as exc:
            self.page('Request too large', escape(str(exc)), 413)
        except (ValueError, UnicodeError) as exc:
            self.page('Invalid request', escape(str(exc)), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.page('Server error', 'The request could not be completed.', 500)

    def home(self):
        if self.user:
            return self.redirect('/dashboard')
        self.respond(LANDING_HTML)

    def auth_form(self, error='', status=200):
        action = self.route_path
        self.page('Sign up' if action == '/signup' else 'Login',
                  f'<p class="error">{escape(error)}</p><form method="post" action="{action}">'
                  '<label>Email <input type="email" name="email" required></label>'
                  '<label>Password <input type="password" name="password" required></label><button>Continue</button></form>', status)

    def authenticate(self):
        ip = self.client_address[0]
        # auth exposes failure recording, but no read-only limiter function.
        with auth._fail_lock:
            limited = sum(t > time.monotonic() - 300 for t in auth._fails.get(ip, [])) > 5
        if limited:
            return self.auth_form('Too many attempts. Try again later.', 429)
        email = self.fields.get('email', '').strip().lower()
        password = self.fields.get('password', '')
        user = db.get_user_by_email(email)
        valid = False
        if self.route_path == '/signup':
            if email and '@' in email and len(password) >= 8 and user is None:
                salt, hashed = auth.hash_password(password)
                try:
                    uid = db.create_user(email, hashed, salt)
                    valid = True
                except sqlite3.IntegrityError:
                    pass
        elif user and auth.verify_password(password, user['salt'], user['pw_hash']):
            uid, valid = user['id'], True
        if not valid:
            limited = auth.record_fail(ip)
            return self.auth_form('Too many attempts. Try again later.' if limited else
                                  'Invalid credentials or signup details (password minimum 8 characters).',
                                  429 if limited else 200)
        auth.clear_fails(ip)
        if self.route_path == '/login':
            self.user = db.get_user_by_id(uid)
            return self.dashboard(self.session_cookie(uid))
        self.redirect('/dashboard', self.session_cookie(uid))

    def logout(self):
        self.redirect('/login', self.session_cookie())

    def run_list(self, rows):
        return '<ul>' + ''.join(f'<li><a href="/runs/{r["id"]}">{escape(r["task"])}</a> — {escape(r["status"])}</li>' for r in rows) + '</ul>'

    def image_grid(self, rows):
        return ''.join(f'<div class="card"><img src="/images/{r["id"]}.png" alt="{escape(r["prompt"])}"><p>{escape(r["prompt"])}</p></div>' for r in rows)

    @require_login
    def dashboard(self, cookie=None):
        uid = self.user['id']
        trees = ''.join(f'<li>{escape(t["name"])} <a href="/trees/{t["id"]}/edit">Edit</a> '
                        f'<a href="/chat?tree={t["id"]}">Chat</a><form method="post" action="/trees/{t["id"]}/delete"><button>Delete</button></form></li>' for t in db.list_trees(uid))
        self.page('Dashboard', '<a href="/trees/new">New tree</a><ul>' + trees + '</ul><h2>Recent runs</h2>' +
                  self.run_list(db.list_runs(uid, limit=10)) + '<h2>Recent images</h2>' + self.image_grid(db.list_images(uid, limit=8)),
                  headers={'Set-Cookie': cookie} if cookie else None)

    @require_login
    def new_tree(self):
        example = json.dumps({'name': 'assistant', 'provider': 'openai', 'model': 'gpt-5', 'role': 'You are a helpful assistant.'}, indent=2)
        self.page('New tree', '<form method="post" action="/trees"><label>Name <input name="name" required></label>'
                  f'<textarea name="yaml_text">{escape(example)}</textarea><button>Create</button></form>')

    @require_login
    def create_tree(self):
        text, name = self.fields.get('yaml_text', ''), self.fields.get('name', '').strip()
        _node(text)
        if not name:
            raise ValueError('Name is required')
        tree_id = db.create_tree(self.user['id'], name, text)
        self.redirect(f'/trees/{tree_id}/edit')

    @require_login
    def tree(self, tree_id, action):
        row = db.get_tree(tree_id, self.user['id'])
        if row is None:
            return self.page('Not found', 'Tree not found', 404)
        if action == 'delete':
            db.delete_tree(tree_id, self.user['id'])
            return self.redirect('/dashboard')
        text = self.fields.get('yaml_text', row['yaml_text'])
        error = ''
        if self.command == 'POST':
            try:
                _node(text)
            except (config.TreeError, ValueError) as exc:
                error = str(exc)
            if action == 'validate':
                return self.json({'ok': False, 'errors': [error]} if error else {'ok': True})
            if not error:
                db.update_tree(tree_id, self.user['id'], row['name'], text)
                if self.headers.get('Accept') == 'application/json':
                    return self.json({'ok': True})
                return self.redirect(f'/trees/{tree_id}/edit')
            if self.headers.get('Accept') == 'application/json':
                return self.json({'ok': False, 'errors': [error]}, 400)
        self.editor(tree_id, text, error)

    def editor(self, tree_id, text, error=''):
        studio = STUDIO.read_text(encoding='utf-8')
        injected = '''<div style="position:fixed;bottom:16px;right:20px;z-index:10000;background:#222;padding:12px">
<button id="web-save">Save</button> <span id="web-save-status" role="status"></span> <a href="/dashboard">Dashboard</a></div>
<script>
window.addEventListener('load',()=>{
const status=document.getElementById('web-save-status');
status.textContent=INITIAL_ERROR;
try {document.getElementById('imp').value=INITIAL_TEXT;document.getElementById('impgo').click();
const err=document.getElementById('imperr').textContent;if(err)status.textContent=err;
} catch(e) {status.textContent='Import failed: '+e.message;}
document.getElementById('web-save').onclick=async()=>{
status.textContent='Saving…';
try {const response=await fetch(SAVE_URL,{method:'POST',headers:{'Accept':'application/json'},
body:new URLSearchParams({yaml_text:toYaml()})});
const result=await response.json();
status.textContent=response.ok&&result.ok?'Saved':(result.errors||['Save failed']).join('; ');
} catch(e) {status.textContent='Save failed: '+e.message;}
};
});
</script>'''
        values = {'INITIAL_ERROR': error, 'INITIAL_TEXT': text, 'SAVE_URL': f'/trees/{tree_id}'}
        injected = re.sub(r'INITIAL_ERROR|INITIAL_TEXT|SAVE_URL',
                          lambda match: _js(values[match[0]]), injected)
        self.respond(studio.replace('</body>', injected + '\n</body>'), 400 if error else 200)

    @require_login
    def chat(self):
        tree_id = int(self.query.get('tree', ['0'])[0])
        tree = db.get_tree(tree_id, self.user['id'])
        if tree is None:
            return self.page('Not found', 'Tree not found', 404)
        self.page('Chat: ' + tree['name'], f'<form method="post" action="/chat"><input type="hidden" name="tree_id" value="{tree_id}">'
                  '<textarea name="message" required></textarea><button>Send</button></form><h2>Prior runs</h2>' +
                  self.run_list(db.list_runs(self.user['id'], tree_id=tree_id)))

    @require_login
    def start_chat(self):
        uid = self.user['id']
        tree_id = int(self.fields.get('tree_id', '0'))
        tree = db.get_tree(tree_id, uid)
        if tree is None:
            return self.page('Not found', 'Tree not found', 404)
        task = self.fields.get('message', '').strip()
        if not task:
            raise ValueError('Message is required')
        node = _node(tree['yaml_text'])
        needed = set(config.providers_used(node))
        def summarizers(n):
            if n.get('summarizer'):
                needed.add(n['summarizer']['provider'])
            for child in n['children']:
                summarizers(child)
        summarizers(node)
        keymap = {p: db.get_key(uid, p) for p in needed}
        missing = sorted(p for p, key in keymap.items() if not key)
        if missing:
            return self.page('Missing keys', '<p>Missing providers: ' + escape(', '.join(missing)) +
                             '</p><a href="/settings/keys">Set provider keys</a>', 400)
        run_id = db.create_run(uid, tree_id, task, status='running')
        with self.app.lock:
            self.app.RUNS[run_id] = {'status': 'running', 'events': [], 'record': None}
        threading.Thread(target=self.app._chat_worker,
                         args=(run_id, uid, tree_id, node, task, keymap), daemon=True).start()
        self.redirect(f'/runs/{run_id}')

    @require_login
    def run(self, run_id, events):
        row = db.get_run(run_id, self.user['id'])
        if row is None:
            return self.page('Not found', 'Run not found', 404)
        if events:
            return self.events(run_id)
        body = f'<p>Status: {escape(row["status"])}</p><div id="transcript"><pre>{escape(row["transcript_json"] or "Running…")}</pre></div>'
        if row['status'] == 'running':
            body += f'''<script>const stream=new EventSource('/runs/{run_id}/events');
stream.addEventListener('done',async()=>{{stream.close();const r=await fetch('/runs/{run_id}');
const doc=new DOMParser().parseFromString(await r.text(),'text/html');
document.getElementById('transcript').replaceWith(doc.getElementById('transcript'));
document.querySelector('h1+p').textContent=doc.querySelector('h1+p').textContent;}});
stream.onerror=()=>{{stream.close();setTimeout(()=>location.reload(),1500)}};</script>'''
        self.page(f'Run {run_id}', body)

    def events(self, run_id):
        with self.app.lock:
            known = run_id in self.app.RUNS
        if not known:
            return self.page('Not found', 'Live run not found', 404)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        index, deadline = 0, time.monotonic() + 600
        while time.monotonic() < deadline:
            with self.app.lock:
                state = self.app.RUNS[run_id]
                batch = state['events'][index:]
                finished = state['status'] != 'running'
            for event in batch:
                self.wfile.write(('event: node\ndata: ' + json.dumps(event) + '\n\n').encode())
            index += len(batch)
            if finished:
                self.wfile.write(b'event: done\ndata: {}\n\n')
                self.wfile.flush()
                time.sleep(1)
                return
            self.wfile.flush()
            time.sleep(0.4)

    @require_login
    def keys(self):
        uid = self.user['id']
        if self.command == 'POST':
            for provider in PROVIDERS:
                if self.fields.get('delete_' + provider) == '1':
                    db.delete_key(uid, provider)
                elif self.fields.get(provider, '').strip():
                    db.set_key(uid, provider, self.fields[provider].strip())
            return self.redirect('/settings/keys')
        statuses = db.keys_status(uid)
        body = '<form method="post">' + ''.join(
            f'<label>{p} ({"set" if statuses.get(p) else "not set"}) '
            f'<input type="password" name="{p}" autocomplete="new-password"> '
            f'<button name="delete_{p}" value="1">Delete</button></label>' for p in PROVIDERS)
        self.page('Provider keys', body + '<button>Save keys</button></form>')

    @require_login
    def image_page(self):
        self.page('Images', '<form method="post"><select name="provider"><option>openai</option><option>gemini</option></select>'
                  '<textarea name="prompt" required></textarea><button>Generate</button></form>' +
                  self.image_grid(db.list_images(self.user['id'], limit=100)))

    @require_login
    def generate_image(self):
        uid = self.user['id']
        provider, prompt = self.fields.get('provider'), self.fields.get('prompt', '').strip()
        if provider not in ('openai', 'gemini') or not prompt:
            raise ValueError('Choose openai or gemini and enter a prompt')
        key = db.get_key(uid, provider)
        if not key:
            return self.page('Missing key', '<a href="/settings/keys">Set the provider key</a>', 400)
        image_id, path = None, None
        try:
            png, model = images.generate_image(provider, prompt, key)
            image_id = db.create_image(uid, provider, model, prompt, '')
            path = Path.cwd() / 'serve-data/images' / str(uid) / f'{image_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png)
            # db has no image mutation helpers; keep both mutations ownership-scoped.
            db._write('UPDATE images SET png_path=? WHERE id=? AND user_id=?',
                      (str(path), image_id, uid))
        except Exception:
            if image_id is not None:
                db._write('DELETE FROM images WHERE id=? AND user_id=?', (image_id, uid))
            if path is not None:
                path.unlink(missing_ok=True)
            return self.page('Image failed', 'Image generation or storage failed.', 400)
        self.redirect('/images')

    @require_login
    def image_file(self, image_id):
        row = db.get_image(image_id, self.user['id'])
        if row is None:
            return self.page('Not found', 'Image not found', 404)
        try:
            png = Path(row['png_path']).read_bytes()
        except OSError:
            return self.page('Not found', 'Image not found', 404)
        self.respond(png, content_type='image/png')


def build_app(db_path):
    """Initialize storage and return a localhost server on an ephemeral port."""
    db.init_db(db_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.app = WebApp(db_path)
    return server


def serve_forever(host='127.0.0.1', port=8080):
    """Run the local web application until interrupted."""
    db_path = Path.cwd() / 'serve-data/web.db'
    db.init_db(db_path)
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = WebApp(db_path)
    print(f'http://{host}:{server.server_port}')
    try:
        server.serve_forever()
    finally:
        server.server_close()
