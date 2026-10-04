"""Decode tool diagnostics only; retain raw bytes and their identity."""
from pathlib import Path
import hashlib,json
def diagnostic_text(path):
    path=Path(path);raw=path.read_bytes()
    try:text=raw.decode('utf-8');replacement=False
    except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');replacement=True
    path.with_suffix(path.suffix+'.decode.json').write_text(json.dumps({'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'strict_utf8':not replacement,'diagnostic_replacement_used':replacement,'raw_bytes_preserved':True},indent=2),encoding='utf-8')
    return text
