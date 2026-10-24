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


APP_CSS = """
:root{--paper:#f3f1ea;--paper2:#ece9df;--ink:#171715;--ink2:#3a3934;--mute:#77736a;
--line:rgba(23,23,21,.13);--line2:rgba(23,23,21,.26);--accent:#c23a22;
--sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Hiragino Sans","Apple SD Gothic Neo","Segoe UI","Noto Sans",Arial,sans-serif;
--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%}
body{background:var(--paper);color:var(--ink);font:16px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:inherit}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
nav{position:sticky;top:0;z-index:10;background:rgba(243,241,234,.9);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;height:60px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px;letter-spacing:-.01em}
.brand svg{width:20px;height:20px}
.links{display:flex;gap:22px;align-items:center;font-size:13.5px;color:var(--ink2)}
.links a{position:relative;text-decoration:none}
.links a:hover{color:var(--ink)}
main.wrap{padding-top:52px;padding-bottom:110px;min-height:72vh}
h1{font-size:clamp(32px,5vw,50px);line-height:1.05;letter-spacing:-.04em;font-weight:600;margin-bottom:10px}
.eyebrow{font:12px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--mute);display:flex;align-items:center;gap:12px;margin-bottom:26px}
.eyebrow b{width:8px;height:8px;background:var(--accent);display:inline-block}
h2{font-size:23px;letter-spacing:-.02em;font-weight:600;margin:60px 0 6px}
label{display:block;font:11.5px/1.6 var(--mono);letter-spacing:.09em;text-transform:uppercase;color:var(--mute);margin:22px 0 8px}
input[type=email],input[type=password],input[type=text],textarea,select{font:15px/1.5 var(--sans);background:transparent;border:1px solid var(--line2);color:var(--ink);padding:12px 14px;width:100%;max-width:560px;border-radius:0}
input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent)}
textarea{min-height:150px;font-family:var(--mono);font-size:13px;line-height:1.7;resize:vertical}
select{max-width:280px}
.btn{display:inline-block;background:var(--ink);color:var(--paper);border:0;padding:13px 24px;font-size:14px;font-weight:500;cursor:pointer;margin-top:26px;text-decoration:none;transition:background .25s}
.btn:hover{background:var(--accent)}
.link-btn{background:none;border:0;color:var(--mute);font-size:13.5px;cursor:pointer;text-decoration:underline;text-underline-offset:3px;padding:0}
.link-btn:hover{color:var(--accent)}
.error{color:var(--accent);font-size:14px;margin:14px 0}
.mute{color:var(--mute)}
.card{border:1px solid var(--line2);padding:20px;background:rgba(255,255,255,.35);margin:18px 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:15px 0;border-bottom:1px solid var(--line)}
.row:first-child{border-top:1px solid var(--line)}
.task{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ops{display:flex;gap:18px;align-items:center;font-size:14px;flex:none}
.ops a{text-decoration:underline;text-underline-offset:3px;color:var(--ink2)}
.ops a:hover{color:var(--accent)}
form.inline,form.logout{display:inline}
.pill{font:10.5px/1 var(--mono);letter-spacing:.07em;text-transform:uppercase;border:1px solid var(--line2);padding:6px 11px;color:var(--ink2);white-space:nowrap}
.pill.running,.pill.on{color:var(--accent);border-color:var(--accent)}
.grid-img{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px;margin-top:20px}
.grid-img .card{margin:0;padding:14px}
.grid-img img{width:100%;display:block;border:1px solid var(--line)}
.grid-img figcaption{font-size:12.5px;color:var(--ink2);margin-top:10px;line-height:1.5}
pre{background:var(--paper2);border:1px solid var(--line);padding:18px;font:12.5px/1.7 var(--mono);overflow-x:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin-top:16px;color:var(--ink2)}
.status-line{color:var(--ink2);display:flex;align-items:center;gap:12px;margin-bottom:4px}
.auth{max-width:430px;margin:30px auto 0}
.switch{margin-top:26px;font-size:14px;color:var(--ink2)}
.switch a{text-decoration:underline;text-underline-offset:3px}
.lede-sm{color:var(--ink2);max-width:600px;margin:6px 0 4px;font-size:15px}
.key-row{display:flex;justify-content:space-between;align-items:center;gap:18px}
.prov-name{font:12.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px}
.key-form{display:flex;gap:14px;align-items:center}
.key-form input{max-width:300px;margin:0}
footer{border-top:1px solid var(--line);padding:34px 0 44px;color:var(--mute);font-size:13px}
footer .wrap{display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
footer a{color:var(--ink2);text-decoration:none}
@media (max-width:640px){
:root{--g:16px}
.wrap{padding:0 16px}
.links{gap:14px;font-size:12.5px}
.row{flex-direction:column;align-items:flex-start;gap:8px}
.key-row{flex-direction:column;align-items:stretch}
.key-form input{max-width:none}
h2{margin-top:44px}
}
"""


def _page(title, body, user=None):
    logo = ('<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v5M10 10H4.5v4.5'
            'M10 10h5.5v4.5" fill="none" stroke="#171715" stroke-width="1.2"/>'
            '<circle cx="10" cy="4" r="2.4" fill="#c23a22"/>'
            '<circle cx="4.5" cy="16" r="2" fill="#171715"/>'
            '<circle cx="15.5" cy="16" r="2" fill="#171715"/></svg>')
    links = (f'<a href="/dashboard">Dashboard</a><a href="/images">Images</a>'
             '<a href="/settings/keys">Keys</a>'
             '<form action="/logout" method="post" class="logout"><button class="link-btn">Log out</button></form>') if user else (
                 '<a href="/login">Log in</a><a href="/signup">Sign up</a>')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} &middot; Swarm Tree</title><link rel="icon" href="data:,">
<style>{APP_CSS}</style></head>
<body><nav><div class="wrap"><a class="brand" href="/">{logo}Swarm Tree</a>
<div class="links">{links}</div></div></nav>
<main class="wrap"><h1>{escape(title)}</h1>{body}</main>
<footer><div class="wrap"><span>tree.haocoach.com</span>
<a href="https://github.com/hahahahahahahahah6/swarm-tree">GitHub</a></div></div>
</body></html>'''


def require_login(fn):
    """Redirect anonymous requests before accessing owned resources."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self.user is None:
            return self.redirect('/login')
        return fn(self, *args, **kwargs)
    return wrapped


LANDING_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Swarm Tree</title><meta name="description" content="Swarm Tree is a hierarchical multi-agent orchestration tool. A coordinator delegates to child agents, each on the LLM provider you choose, with your own keys."><meta name="theme-color" content="#f3f1ea"><link rel="icon" href="data:,"><style>
:root{--paper:#f3f1ea;--paper2:#ece9df;--ink:#171715;--ink2:#3a3934;--mute:#77736a;--line:rgba(23,23,21,.13);--line2:rgba(23,23,21,.26);--accent:#c23a22;--sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Hiragino Sans","Apple SD Gothic Neo","Segoe UI","Noto Sans",Arial,sans-serif;--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;--w:1120px;--g:28px;}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{background:var(--paper);color:var(--ink);font:16px/1.6 var(--sans);-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:inherit;text-decoration:none}
::selection{background:var(--accent);color:var(--paper)}
.wrap{max-width:var(--w);margin:0 auto;padding:0 var(--g)}
.mono{font-family:var(--mono)}
nav{position:sticky;top:0;z-index:10;background:rgba(243,241,234,.88);-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;height:60px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.01em;font-size:15px}
.brand svg{width:20px;height:20px}
.links{display:flex;gap:28px;font-size:13.5px;color:var(--ink2)}
.ul{position:relative}
.ul::after{content:"";position:absolute;left:0;right:0;bottom:-3px;height:1px;background:currentColor;transform:scaleX(0);transform-origin:right;transition:transform .35s cubic-bezier(.2,.7,.2,1)}
.ul:hover::after{transform:scaleX(1);transform-origin:left}
.hero{padding:88px 0 96px;border-bottom:1px solid var(--line)}
.hero .wrap{display:grid;grid-template-columns:1.08fr .92fr;gap:64px;align-items:center}
.eyebrow{font:12px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--mute);display:flex;align-items:center;gap:12px;margin-bottom:32px}
.eyebrow b{width:8px;height:8px;background:var(--accent);display:inline-block}
h1{font-size:clamp(42px,6.6vw,78px);line-height:1.02;letter-spacing:-.045em;font-weight:600}
h1 em{font-style:normal;color:var(--accent)}
.lede{margin-top:28px;max-width:500px;font-size:17.5px;color:var(--ink2);line-height:1.65}
.term{margin-top:36px;border-top:1px solid var(--line2);border-bottom:1px solid var(--line);padding:14px 0;font:13.5px/1.5 var(--mono);color:var(--ink2);overflow:hidden;white-space:nowrap;text-overflow:clip;max-width:560px}
.term .p{color:var(--accent);margin-right:8px}
.term .q{color:var(--ink)}
.caret{display:inline-block;width:7px;height:15px;background:var(--ink);vertical-align:-2px;margin-left:1px;animation:blink 1.05s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.acts{margin-top:36px;display:flex;gap:28px;align-items:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:10px;background:var(--ink);color:var(--paper);padding:13px 20px;font-size:14px;font-weight:500;letter-spacing:-.005em;transition:background .25s,transform .25s}
.btn:hover{background:var(--accent)}
.btn svg{width:16px;height:16px;transition:transform .3s}
.btn:hover svg{transform:translateX(3px)}
.txt{font-size:14px;color:var(--ink2);display:inline-flex;gap:8px;align-items:center}
figure{position:relative;border:1px solid var(--line2);background:linear-gradient(var(--paper),var(--paper2));padding:22px 22px 0}
.fig-top{display:flex;justify-content:space-between;font:11px/1 var(--mono);color:var(--mute);letter-spacing:.06em;text-transform:uppercase;padding-bottom:14px;border-bottom:1px solid var(--line)}
.fig-top .st{display:flex;align-items:center;gap:7px}
.fig-top .st i{width:6px;height:6px;border-radius:50%;background:var(--mute);transition:background .3s}
.fig-top .st.live i{background:var(--accent)}
#tree{display:block;width:100%;height:auto;margin:10px 0 4px}
#tree .edge{fill:none;stroke:var(--line2);stroke-width:1}
#tree .edge.hot{stroke:var(--ink2)}
#tree .core{fill:var(--paper);stroke:var(--ink);stroke-width:1.2;transition:fill .35s}
#tree .halo{fill:none;stroke:var(--accent);stroke-width:1;opacity:0;transform-box:fill-box;transform-origin:center}
#tree .on .core{fill:var(--accent);stroke:var(--accent)}
#tree .on .halo{animation:ring 1.6s ease-out infinite}
#tree .done .core{fill:var(--ink)}
#tree .lbl{font:500 11.5px var(--mono);fill:var(--ink);text-anchor:middle}
#tree .prov{font:10px var(--mono);fill:var(--mute);text-anchor:middle;letter-spacing:.04em}
@keyframes ring{0%{transform:scale(.6);opacity:.7}100%{transform:scale(2.1);opacity:0}}
.log{border-top:1px solid var(--line);margin:0 -22px;padding:12px 22px 14px;height:118px;font:11.5px/1.75 var(--mono);color:var(--ink2);overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end}
.log div{display:grid;grid-template-columns:52px 96px 1fr;white-space:nowrap;animation:lin .4s ease-out}
.log div span:first-child{color:var(--mute)}
.log div span:nth-child(2){color:var(--ink)}
.log div:nth-last-child(n+4){opacity:.38}
@keyframes lin{from{opacity:0;transform:translateY(6px)}}
figcaption{position:absolute;right:-30px;top:0;writing-mode:vertical-rl;font:10.5px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--mute)}
section{padding:112px 0;border-bottom:1px solid var(--line)}
.sh{display:flex;align-items:center;gap:16px;font:12px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin-bottom:56px}
.sh .no{color:var(--accent)}
.sh i{flex:1;height:1px;background:var(--line)}
h2{font-size:clamp(30px,4.2vw,48px);line-height:1.08;letter-spacing:-.035em;font-weight:600;max-width:720px}
.sub{margin-top:18px;color:var(--ink2);max-width:560px;font-size:16.5px}
.grid{margin-top:64px;display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line2);border-left:1px solid var(--line)}
.cell{position:relative;border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:36px 34px 40px;transition:background .35s}
.cell:hover{background:rgba(255,255,255,.45)}
.cell .hd{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:40px}
.cell svg{width:30px;height:30px;stroke:var(--ink);fill:none;stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round;transition:stroke .3s,transform .5s cubic-bezier(.2,.7,.2,1)}
.cell:hover svg{stroke:var(--accent);transform:translateY(-2px)}
.cell .n{font:11.5px var(--mono);color:var(--mute)}
.cell h3{font-size:21px;letter-spacing:-.02em;font-weight:600;margin-bottom:10px}
.cell p{color:var(--ink2);font-size:15px;max-width:420px}
.specs{list-style:none;margin-top:22px;border-top:1px solid var(--line)}
.specs li{display:grid;grid-template-columns:110px 1fr;gap:12px;padding:10px 0;border-bottom:1px solid var(--line);font-size:14px;color:var(--ink2)}
.specs li span{font:11.5px/1.9 var(--mono);color:var(--ink);text-transform:lowercase}
.prov-row{margin-top:64px;display:grid;grid-template-columns:200px 1fr;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.prov-row .k{padding:20px 0;font:12px/1.5 var(--mono);color:var(--mute);letter-spacing:.06em;text-transform:uppercase}
.prov-row ul{list-style:none;display:flex;flex-wrap:wrap}
.prov-row li{padding:18px 26px;border-left:1px solid var(--line);font:14px var(--mono);color:var(--ink);transition:color .25s}
.prov-row li:hover{color:var(--accent)}
.steps{margin-top:64px;display:grid;grid-template-columns:repeat(3,1fr);gap:0;border-top:1px solid var(--line2)}
.step{padding:34px 30px 0 0;border-right:1px solid var(--line);margin-right:30px}
.step:last-child{border-right:0;margin-right:0}
.step .big{font:300 56px/1 var(--sans);letter-spacing:-.05em;color:var(--line2);transition:color .4s}
.step:hover .big{color:var(--accent)}
.step h3{margin:22px 0 8px;font-size:19px;letter-spacing:-.02em;font-weight:600}
.step p{color:var(--ink2);font-size:15px;min-height:72px}
pre{margin-top:22px;font:12.5px/1.75 var(--mono);color:var(--ink2);border-top:1px solid var(--line);padding:16px 0 32px;overflow-x:auto;white-space:pre}
pre .k{color:var(--ink)}pre .a{color:var(--accent)}pre .c{color:var(--mute)}
.cta{padding:136px 0;text-align:left}
.cta h2{font-size:clamp(36px,5.6vw,68px);max-width:none;letter-spacing:-.045em}
.cta h2 em{font-style:normal;color:var(--accent)}
.clone{margin-top:44px;display:flex;align-items:stretch;max-width:640px;border:1px solid var(--line2)}
.clone code{flex:1;font:13px/1 var(--mono);padding:17px 18px;color:var(--ink2);overflow-x:auto;white-space:nowrap}
.clone code b{color:var(--accent);font-weight:400;margin-right:8px}
.clone button{font:12px var(--mono);letter-spacing:.06em;text-transform:uppercase;border:0;border-left:1px solid var(--line2);background:transparent;color:var(--ink);padding:0 20px;cursor:pointer;transition:background .25s,color .25s;min-width:88px}
.clone button:hover{background:var(--ink);color:var(--paper)}
.cta .acts{margin-top:28px}
footer{padding:40px 0 48px}
footer .wrap{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--mute)}
footer .r{display:flex;gap:26px;font-family:var(--mono);font-size:12.5px}
footer a{color:var(--ink2)}
.rv{opacity:0;transform:translateY(16px);transition:opacity .8s cubic-bezier(.2,.7,.2,1),transform .8s cubic-bezier(.2,.7,.2,1);transition-delay:var(--d,0ms)}
.rv.in{opacity:1;transform:none}
.hero .rv{transition-duration:1s}
@media (max-width:920px){
.hero{padding:56px 0 72px}
.hero .wrap{grid-template-columns:1fr;gap:56px}
figcaption{display:none}
.steps{grid-template-columns:1fr}
.step{border-right:0;margin-right:0;padding-right:0}
.step p{min-height:0}
}
@media (max-width:680px){
:root{--g:16px}
.links .hide{display:none}
.links{gap:20px}
section{padding:80px 0}
.sh{margin-bottom:40px}
.grid{grid-template-columns:1fr;margin-top:44px}
.cell{padding:28px 22px 32px}
.cell .hd{margin-bottom:28px}
.specs li{grid-template-columns:92px 1fr}
.prov-row{grid-template-columns:1fr;margin-top:44px}
.prov-row .k{padding:14px 0 6px}
.prov-row ul{border-top:1px solid var(--line)}
.prov-row li{padding:13px 16px 13px 0;border-left:0;margin-right:4px}
.steps{margin-top:44px}
.lede{font-size:16.5px}
figure{padding:16px 16px 0}
.log{margin:0 -16px;padding:10px 16px 12px;font-size:10.5px}
.log div{grid-template-columns:44px 82px 1fr}
.cta{padding:96px 0}
.clone code{font-size:11.5px;padding:15px 12px}
.clone button{min-width:70px;padding:0 12px}
}
@media (prefers-reduced-motion:reduce){
::before,*::after{animation:none!important;transition:none!important}
.rv{opacity:1;transform:none}
}
</style></head><body><nav><div class="wrap"><a class="brand" href="#top" aria-label="Swarm Tree home"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v5M10 10H4.5v4.5M10 10h5.5v4.5" fill="none" stroke="#171715" stroke-width="1.2"/><circle cx="10" cy="4" r="2.4" fill="#c23a22"/><circle cx="4.5" cy="16" r="2" fill="#171715"/><circle cx="15.5" cy="16" r="2" fill="#171715"/></svg>Swarm Tree</a><div class="links"><a class="ul hide" href="#features">Features</a><a class="ul hide" href="#how">How it works</a><a class="ul" href="/login">Log in</a><a class="ul" href="https://github.com/hahahahahahahahah6/swarm-tree">GitHub</a></div></div></nav>
<header class="hero" id="top"><div class="wrap"><div><div class="eyebrow rv"><b></b>Hierarchical multi-agent orchestration</div><h1 class="rv" style="--d:80ms">Hand one task<br>to a tree of<br>agents<em>.</em></h1><p class="lede rv" style="--d:180ms">A coordinator splits the work and delegates it down to specialists: a researcher, a designer, an image node. Each runs on the model you pick, with your own keys. Write the tree in YAML or draw it in the studio.</p><div class="term rv" style="--d:260ms" aria-label="Example command"><span class="p">$</span>swarm-tree run --tree tree.yaml --task "<span class="q" id="typed"></span><span class="caret"></span>"</div><div class="acts rv" style="--d:340ms"><a class="btn" href="/signup">Start free <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 8h10M9 4l4 4-4 4"/></svg></a><a class="txt ul" href="#how">How it works</a></div></div>
<figure class="rv" style="--d:200ms">
<div class="fig-top"><span>tree.yaml</span><span class="st" id="st"><i></i><span id="stt">idle</span></span></div>
<svg id="tree" viewBox="0 0 400 320" role="img" aria-label="Animated diagram: a coordinator delegates to researcher, designer and writer agents; the designer delegates to an image-gen agent; results flow back up."></svg>
<div class="log" id="log" aria-hidden="true"></div>
<figcaption>fig. 1 &nbsp;one task through the tree</figcaption>
</figure>
</div></header><section id="features"><div class="wrap"><div class="sh rv"><span class="no">01</span><span>Features</span><i></i></div><h2 class="rv">Everything the tree needs, and nothing it doesn't.</h2><p class="sub rv">A CLI for scripts, a studio for people, and a serve mode for trees that should keep working after you close the tab.</p>
<div class="grid">
<div class="cell rv">
<div class="hd"><svg viewBox="0 0 30 30" aria-hidden="true"><circle cx="15" cy="6" r="2.6"/><circle cx="6" cy="23" r="2.6"/><circle cx="15" cy="23" r="2.6"/><circle cx="24" cy="23" r="2.6"/><path d="M15 8.6V15M15 15H6v5.4M15 15v5.4M15 15h9v5.4"/></svg><span class="n">01 / 04</span></div>
<h3>Visual editor</h3>
<p>Build trees on a canvas. Add a node, nest it under a parent, give it a role, a prompt and a provider. See the whole hierarchy at a glance instead of reading indentation.</p>
</div>
<div class="cell rv" style="--d:90ms">
<div class="hd"><svg viewBox="0 0 30 30" aria-hidden="true"><path d="M5 7h20v13H12l-7 5z"/><path d="M10 12h10M10 16h6"/></svg><span class="n">02 / 04</span></div>
<h3>Streaming chat</h3>
<p>Talk to any tree. Output streams in as each node works, so you can see which agent is speaking, what it was asked, and what it handed back up.</p>
</div>
<div class="cell rv">
<div class="hd"><svg viewBox="0 0 30 30" aria-hidden="true"><rect x="5" y="6" width="20" height="18"/><circle cx="11.5" cy="12.5" r="2"/><path d="M5 21l6-5 4.5 3.5 3.5-2.5 6 4.5"/></svg><span class="n">03 / 04</span></div>
<h3>Image generation</h3>
<p>Image nodes call the latest OpenAI and Gemini image models. A designer agent writes the brief, the image node renders it, and the result lands in the same thread.</p>
</div>
<div class="cell rv" style="--d:90ms">
<div class="hd"><svg viewBox="0 0 30 30" aria-hidden="true"><circle cx="15" cy="15" r="10"/><path d="M15 9v6l4 2.5"/></svg><span class="n">04 / 04</span></div>
<h3>Schedules &amp; triggers</h3>
<p>Serve mode keeps trees running on their own, and keeps them cheap.</p>
<ul class="specs">
<li><span>schedule</span>Each node runs on its own cadence.</li>
<li><span>trigger</span>A shell check runs first. Nothing changed, no LLM call.</li>
<li><span>budget</span>Spend caps set per node, not per tree.</li>
<li><span>queue</span>Actions that need you wait in an owner queue.</li>
</ul>
</div>
</div>
<div class="prov-row rv">
<div class="k">Bring your own key</div>
<ul><li>OpenAI</li><li>Anthropic</li><li>Gemini</li><li>xAI</li><li>DeepSeek</li></ul>
</div>
</div></section><section id="how"><div class="wrap"><div class="sh rv"><span class="no">02</span><span>How it works</span><i></i></div><h2 class="rv">Three steps. Define, run, then leave it running.</h2>
<div class="steps">
<div class="step rv">
<div class="big">01</div>
<h3>Define the tree</h3>
<p>One coordinator at the root. Children for each kind of work. A provider on every node.</p>
<pre><span class="c"># tree.yaml (sketch)</span>
<span class="k">root</span>: <span class="a">coordinator</span>  <span class="k">provider</span>: anthropic
<span class="k">children</span>:
  - <span class="a">researcher</span>  <span class="k">provider</span>: openai
  - <span class="a">designer</span>    <span class="k">provider</span>: gemini
  - <span class="a">image-gen</span>   <span class="k">provider</span>: openai</pre>
</div>
<div class="step rv" style="--d:110ms">
<div class="big">02</div>
<h3>Run a task</h3>
<p>From the terminal, or open the studio and chat with the tree while it streams.</p>
<pre><span class="a">$</span> <span class="k">swarm-tree run</span> \
  --tree tree.yaml \
  --task "Write a launch brief"</pre>
</div>
<div class="step rv" style="--d:220ms">
<div class="big">03</div>
<h3>Serve it</h3>
<p>Nodes wake on schedule or when a trigger finds real change. Budgets hold. You approve what matters.</p>
<pre><span class="k">digest</span>   schedule mon 09:00
<span class="k">watcher</span>  trigger <span class="c">no change, skipped</span>
<span class="k">writer</span>   budget $0.42 / $2.00
<span class="k">owner</span>    queue <span class="a">1 waiting</span></pre>
</div>
</div>
</div></section><section class="cta"><div class="wrap"><div class="sh rv"><span class="no">03</span><span>Start</span><i></i></div><h2 class="rv">Start with one coordinator<br>and two children<em>.</em> Grow from there.</h2><div class="acts rv" style="--d:120ms"><a class="btn" href="/signup">Create free account <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 8h10M9 4l4 4-4 4"/></svg></a><a class="txt ul" href="https://github.com/hahahahahahahahah6/swarm-tree">or read the docs on GitHub</a></div><div class="clone rv" style="--d:200ms"><code><b>$</b>git clone https://github.com/hahahahahahahahah6/swarm-tree</code><button type="button" id="copy">Copy</button></div></div></section><footer><div class="wrap"><span class="brand"><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v5M10 10H4.5v4.5M10 10h5.5v4.5" fill="none" stroke="#171715" stroke-width="1.2"/><circle cx="10" cy="4" r="2.4" fill="#c23a22"/><circle cx="4.5" cy="16" r="2" fill="#171715"/><circle cx="15.5" cy="16" r="2" fill="#171715"/></svg>Swarm Tree</span><div class="r"><span>tree.haocoach.com</span><a class="ul" href="/login">Log in</a><a class="ul" href="https://github.com/hahahahahahahahah6/swarm-tree">GitHub</a><span>&copy; 2026</span></div></div></footer>
<script>
(function(){
var RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
var els = document.querySelectorAll('.rv');
if ('IntersectionObserver' in window && !RM) {
var io = new IntersectionObserver(function(es){
es.forEach(function(e){ if (e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); } });
}, {threshold: .12, rootMargin: '0px 0px -40px 0px'});
els.forEach(function(el){ io.observe(el); });
} else els.forEach(function(el){ el.classList.add('in'); });
var tasks = ['Write a launch brief with a hero image', 'Compare pricing across five competitors', "Summarize this week's issues into a digest", 'Draft three logo directions and render them'];
var t = document.getElementById('typed'), ti = 0, ci = 0, del = false;
function type(){
var s = tasks[ti];
if (!del) { ci++; t.textContent = s.slice(0, ci); if (ci === s.length){ del = true; return setTimeout(type, 2400); } setTimeout(type, 38 + Math.random()*40); }
else { ci--; t.textContent = s.slice(0, ci); if (ci === 0){ del = false; ti = (ti+1) % tasks.length; return setTimeout(type, 380); } setTimeout(type, 16); }
}
if (RM) t.textContent = tasks[0]; else setTimeout(type, 900);
var NS = 'http://www.w3.org/2000/svg', svg = document.getElementById('tree');
var N = { root:{x:200,y:34,l:'coordinator',p:'anthropic'}, res:{x:70,y:150,l:'researcher',p:'openai'}, des:{x:200,y:150,l:'designer',p:'gemini'}, wri:{x:330,y:150,l:'writer',p:'xai'}, img:{x:200,y:262,l:'image-gen',p:'openai'} };
var E = {}, G = {};
function mk(tag, a, parent){ var el = document.createElementNS(NS, tag); for (var k in a) el.setAttribute(k, a[k]); (parent||svg).appendChild(el); return el; }
function edge(a, b){ var A = N[a], B = N[b], y1 = A.y + 9, y2 = B.y - 9, m = (y1 + y2) / 2 + (a === 'root' ? 8 : 0); E[a+b] = mk('path', {'class':'edge', d:'M'+A.x+' '+(y1+30)+' C'+A.x+' '+(m+20)+' '+B.x+' '+(m-10)+' '+B.x+' '+y2}); }
edge('root','res'); edge('root','des'); edge('root','wri'); edge('des','img');
E.desimg.setAttribute('d', 'M200 189 L200 253');
Object.keys(N).forEach(function(k){ var n = N[k], g = mk('g', {'class':'nd', transform:'translate('+n.x+' '+n.y+')'}); mk('circle', {'class':'halo', r:9}, g); mk('circle', {'class':'core', r:6}, g); mk('text', {'class':'lbl', y:24}, g).textContent = n.l; mk('text', {'class':'prov', y:37}, g).textContent = n.p; G[k] = g; });
var log = document.getElementById('log'), st = document.getElementById('st'), stt = document.getElementById('stt'), t0 = 0;
function L(who, msg){ var s = ((performance.now() - t0) / 1000).toFixed(1), d = document.createElement('div'); d.innerHTML = '<span>' + (s < 10 ? '0' : '') + s + 's</span><span>' + who + '</span><span>' + msg + '</span>'; log.appendChild(d); while (log.children.length > 6) log.removeChild(log.firstChild); }
function on(k){ G[k].setAttribute('class','nd on'); }
function done(k){ G[k].setAttribute('class','nd done'); }
function wait(ms){ return new Promise(function(r){ setTimeout(r, ms); }); }
function pulse(key, back, color){ return new Promise(function(res){ var p = E[key], len = p.getTotalLength(), dot = mk('circle', {r:3, fill: color || '#c23a22'}), s = null, dur = 780; p.classList.add('hot'); function f(ts){ if (!s) s = ts; var k = Math.min(1, (ts - s) / dur), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2; var pt = p.getPointAtLength(len * (back ? 1 - e : e)); dot.setAttribute('cx', pt.x); dot.setAttribute('cy', pt.y); if (k < 1) requestAnimationFrame(f); else { dot.remove(); p.classList.remove('hot'); res(); } } requestAnimationFrame(f); }); }
function reset(){ Object.keys(G).forEach(function(k){ G[k].setAttribute('class','nd'); }); log.innerHTML = ''; }
async function run(){ for (;;) { reset(); t0 = performance.now(); st.classList.add('live'); stt.textContent = 'running'; on('root'); L('coordinator', 'plan: 3 subtasks'); await wait(800); await Promise.all([ pulse('rootres').then(function(){ on('res'); L('researcher', 'reading 12 sources'); }), wait(160).then(function(){ return pulse('rootdes'); }).then(function(){ on('des'); L('designer', 'writing image brief'); }), wait(320).then(function(){ return pulse('rootwri'); }).then(function(){ on('wri'); L('writer', 'outlining copy'); }) ]); await wait(700); await pulse('desimg'); on('img'); L('image-gen', 'rendering 1 of 1'); await wait(900); done('res'); L('researcher', 'notes returned'); await pulse('rootres', true, '#171715'); await wait(300); done('img'); L('image-gen', 'image returned'); await pulse('desimg', true, '#171715'); done('des'); await pulse('rootdes', true, '#171715'); done('wri'); L('writer', 'draft returned'); await pulse('rootwri', true, '#171715'); done('root'); L('coordinator', 'merged: final answer'); st.classList.remove('live'); stt.textContent = 'done'; await wait(3200); } }
if (RM) { Object.keys(G).forEach(done); stt.textContent = 'done'; L('coordinator','merged: final answer'); } else setTimeout(run, 1200);
var b = document.getElementById('copy');
b.addEventListener('click', function(){ var s = 'git clone https://github.com/hahahahahahahahah6/swarm-tree'; function ok(){ b.textContent = 'Copied'; setTimeout(function(){ b.textContent = 'Copy'; }, 1600); } if (navigator.clipboard) navigator.clipboard.writeText(s).then(ok, function(){}); else { var ta = document.createElement('textarea'); ta.value = s; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); ok(); } catch(e){} ta.remove(); } });
})();
</script></body></html>
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
        is_signup = action == '/signup'
        title = 'Sign up' if is_signup else 'Login'
        other_href, other_text = ('/login', 'Log in') if is_signup else ('/signup', 'Sign up')
        other_line = ('Already have an account' if is_signup else 'New here')
        err = f'<p class="error">{escape(error)}</p>' if error else ''
        self.page(title,
                  '<div class="auth"><div class="eyebrow"><b></b>Account</div>' + err +
                  f'<form method="post" action="{action}">'
                  '<label>Email</label><input type="email" name="email" required autocomplete="email">'
                  '<label>Password</label><input type="password" name="password" required autocomplete="current-password">'
                  '<button class="btn">Continue</button></form>'
                  f'<p class="switch">{other_line}? <a href="{other_href}">{other_text}</a></p></div>', status)

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
        items = ''.join(
            f'<div class="row"><a class="task" href="/runs/{r["id"]}">{escape(r["task"])}</a>'
            f'<span class="pill {escape(r["status"])}">{escape(r["status"])}</span></div>' for r in rows)
        return f'<div class="runs">{items or "<p class=\'mute\'>No runs yet.</p>"}</div>'

    def image_grid(self, rows):
        return '<div class="grid-img">' + ''.join(
            f'<figure class="card"><img src="/images/{r["id"]}.png" alt="{escape(r["prompt"])}" loading="lazy">'
            f'<figcaption>{escape(r["prompt"])}</figcaption></figure>' for r in rows) + '</div>'

    @require_login
    def dashboard(self, cookie=None):
        uid = self.user['id']
        trees = ''.join(
            f'<div class="row"><span class="task">{escape(t["name"])}</span><span class="ops">'
            f'<a href="/trees/{t["id"]}/edit">Edit</a><a href="/chat?tree={t["id"]}">Chat</a>'
            f'<form method="post" action="/trees/{t["id"]}/delete" class="inline">'
            '<button class="link-btn">Delete</button></form></span></div>' for t in db.list_trees(uid))
        self.page('Dashboard',
                  '<div class="eyebrow"><b></b>Your agent trees</div>'
                  '<a class="btn" href="/trees/new" style="margin-top:0">New tree</a>'
                  f'<div class="runs" style="margin-top:22px">{trees or "<p class=\'mute\'>No trees yet. Create one to get started.</p>"}</div>'
                  '<h2>Recent runs</h2>' + self.run_list(db.list_runs(uid, limit=10)) +
                  '<h2>Recent images</h2>' + self.image_grid(db.list_images(uid, limit=8)),
                  headers={'Set-Cookie': cookie} if cookie else None)

    @require_login
    def new_tree(self):
        example = json.dumps({'name': 'assistant', 'provider': 'openai', 'model': 'gpt-5', 'role': 'You are a helpful assistant.'}, indent=2)
        self.page('New tree',
                  '<div class="eyebrow"><b></b>Define in JSON or YAML</div>'
                  '<form method="post" action="/trees"><label>Name</label>'
                  f'<input type="text" name="name" required placeholder="my-assistant">'
                  f'<label>Tree definition</label><textarea name="yaml_text">{escape(example)}</textarea>'
                  '<button class="btn">Create tree</button></form>')

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
        injected = '''<div style="position:fixed;bottom:16px;right:20px;z-index:10000;background:#f3f1ea;border:1px solid rgba(23,23,21,.26);padding:10px 14px;font:13px -apple-system,system-ui,sans-serif;color:#171715">
<button id="web-save" style="background:#171715;color:#f3f1ea;border:0;padding:8px 16px;font:inherit;cursor:pointer">Save</button> <span id="web-save-status" role="status" style="margin:0 10px"></span> <a href="/dashboard" style="color:#171715">Dashboard</a></div>
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
        self.page('Chat · ' + tree['name'],
                  '<div class="eyebrow"><b></b>Talk to the tree</div>'
                  f'<form method="post" action="/chat"><input type="hidden" name="tree_id" value="{tree_id}">'
                  '<label>Message</label><textarea name="message" required placeholder="Ask the tree anything&hellip;"></textarea>'
                  '<button class="btn">Send</button></form><h2>Prior runs</h2>' +
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
        body = (f'<p class="status-line">Status <span class="pill {escape(row["status"])}">'
                f'{escape(row["status"])}</span></p>'
                f'<div id="transcript"><pre>{escape(row["transcript_json"] or "Running…")}</pre></div>')
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
        rows = ''.join(
            f'<div class="card key-row"><div><div class="prov-name">{p}</div>'
            f'<span class="pill {"on" if statuses.get(p) else ""}">{"set" if statuses.get(p) else "not set"}</span></div>'
            f'<div class="key-form"><input type="password" name="{p}" autocomplete="new-password" placeholder="Paste new key">'
            f'<button class="link-btn" name="delete_{p}" value="1">Delete</button></div></div>' for p in PROVIDERS)
        self.page('Provider keys',
                  '<div class="eyebrow"><b></b>Bring your own key</div>'
                  '<p class="lede-sm">Keys are stored for your account only and never shown again. Paste a key to set or replace it.</p>'
                  '<form method="post">' + rows + '<button class="btn">Save keys</button></form>')

    @require_login
    def image_page(self):
        self.page('Images',
                  '<div class="eyebrow"><b></b>Generate</div>'
                  '<form method="post"><label>Provider</label><select name="provider"><option>openai</option><option>gemini</option></select>'
                  '<label>Prompt</label><textarea name="prompt" required placeholder="A quiet studio portrait&hellip;"></textarea>'
                  '<button class="btn">Generate</button></form><h2>Gallery</h2>' +
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
