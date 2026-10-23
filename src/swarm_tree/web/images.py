"""Minimal stdlib image generation clients; credentials never enter errors."""

import base64
import binascii
import json
import time
import urllib.error
import urllib.request

_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
_MODELS = {'openai': 'gpt-image-2.5-flare', 'gemini': 'gemini-3.1-flash-image'}


class ImageError(RuntimeError):
    """Provider-prefixed image request or response failure."""


def generate_image(provider, prompt, api_key, timeout=120):
    """Return (PNG bytes, model); retry HTTP 429/5xx once after one second.

    Error bodies and underlying exception text are deliberately omitted:
    either may contain credentials echoed by a remote server or transport.
    """
    if provider not in _MODELS:
        raise ImageError(f'{provider}: unsupported image provider')
    model = _MODELS[provider]
    try:
        headers = {'Content-Type': 'application/json'}
        if provider == 'openai':
            url = 'https://api.openai.com/v1/images/generations'
            headers['Authorization'] = f'Bearer {api_key}'
            payload = {'model': model, 'prompt': prompt, 'size': '1024x1024',
                       'output_format': 'png', 'n': 1}
        else:
            url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
            headers['x-goog-api-key'] = api_key
            payload = {'contents': [{'parts': [{'text': prompt}]}],
                       'generationConfig': {'responseModalities': ['TEXT', 'IMAGE']}}
        request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                         headers=headers, method='POST')
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                status = exc.code
                exc.close()
                if attempt == 0 and (status == 429 or 500 <= status <= 599):
                    time.sleep(1)
                    continue
                raise ImageError(f'{provider}: API error (HTTP {status})') from None
        data = json.loads(raw)
        if provider == 'openai':
            encoded = data['data'][0]['b64_json']
        else:
            parts = data['candidates'][0]['content']['parts']
            encoded = next(part['inlineData']['data'] for part in parts
                           if isinstance(part, dict)
                           and isinstance(part.get('inlineData'), dict)
                           and 'data' in part['inlineData'])
        png = base64.b64decode(encoded, validate=True)
        if not png.startswith(_PNG_MAGIC):
            raise ImageError(f'{provider}: response image is not PNG')
        return png, model
    except ImageError:
        raise
    except (urllib.error.URLError, OSError):
        raise ImageError(f'{provider}: network request failed') from None
    except (KeyError, IndexError, TypeError, ValueError, AttributeError,
            StopIteration, binascii.Error):
        raise ImageError(f'{provider}: malformed image response or request') from None
