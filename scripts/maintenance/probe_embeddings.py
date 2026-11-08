"""Explicit, source-independent Ollama diagnostic; never accepts zero vectors."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
    model = os.getenv('EMBEDDING_MODEL', 'bge-m3').strip()
    if ':' not in model.rsplit('/', 1)[-1]:
        model += ':latest'

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode('utf-8')
        with urlopen(Request(base + path, data=data, headers={'Content-Type': 'application/json'}), timeout=180) as response:
            return json.load(response)

    report = {'python': sys.version.split()[0], 'requested_model': model,
              'version': request('/api/version')}
    installed = request('/api/tags').get('models', [])
    report['installed_identity'] = next(({'name': m.get('name'), 'digest': m.get('digest')}
                                         for m in installed if m.get('name') == model), None)
    report['probes'] = []
    valid = report['installed_identity'] is not None
    for texts in [['Source document test'],
                  ['Source document test', '\u0627\u062e\u062a\u0628\u0627\u0631 \u0645\u0635\u062f\u0631 \u0639\u0631\u0628\u064a English']]:
        response = request('/api/embed', {'model': model, 'input': texts, 'truncate': False})
        rows = response.pop('embeddings', [])
        stats = []
        for row in rows:
            numeric = isinstance(row, list) and all(type(x) in (int, float) and math.isfinite(x) for x in row)
            norm = math.sqrt(sum(x*x for x in row)) if numeric else None
            stats.append({'dimension': len(row) if isinstance(row, list) else None,
                          'nonzero': sum(x != 0 for x in row) if numeric else None,
                          'finite': numeric, 'norm': norm})
        ok = (len(rows) == len(texts) and response.get('model') == model
              and all(s['finite'] and s['dimension'] == 1024 and s['norm'] > 0 for s in stats))
        valid = valid and ok
        report['probes'].append({'inputs': texts, 'statistics': stats, 'valid_bge_m3': ok,
                                 'response_metadata': response})
    report['ps_after'] = request('/api/ps')
    report['valid'] = valid
    rendered = json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)
    if not valid:
        raise SystemExit('Embedding diagnostic FAILED. Do not rebuild the index with this runtime.')


if __name__ == '__main__':
    main()
