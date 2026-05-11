#!/usr/bin/env python3
"""Fetch bytes from a remote GGUF (HTTP Range) until `general.architecture` KV is readable."""

from __future__ import annotations

import struct
import sys
import urllib.request


def u32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def u64(b: bytes, o: int) -> int:
    return struct.unpack_from("<Q", b, o)[0]


def read_string(b: bytes, o: int) -> tuple[str, int]:
    ln = u64(b, o)
    o += 8
    return b[o : o + ln].decode("utf-8"), o + ln


def skip_value(b: bytes, o: int, typ: int) -> int:
    if typ == 8:
        _, end = read_string(b, o)
        return end
    if typ == 9:
        et = u32(b, o)
        ln = u64(b, o + 4)
        o += 12
        for _ in range(int(ln)):
            o = skip_value(b, o, et)
        return o
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if typ in sizes:
        return o + sizes[typ]
    raise ValueError(f"unsupported GGUF metadata type {typ}")


def peek_architecture(url: str, initial: int = 262_144, cap: int = 16_777_216) -> str | None:
    buf = b""
    step = initial
    while len(buf) < cap:
        req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={len(buf)}-{len(buf) + step - 1}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            chunk = resp.read()
        if not chunk:
            break
        buf += chunk
        if len(buf) < 24 or buf[:4] != b"GGUF":
            continue
        kv_count = u64(buf, 16)
        off = 24
        try:
            for _ in range(kv_count):
                key, off = read_string(buf, off)
                typ = u32(buf, off)
                off += 4
                if typ == 8:
                    val, off = read_string(buf, off)
                    if key == "general.architecture":
                        return val
                else:
                    off = skip_value(buf, off, typ)
            return None
        except (IndexError, struct.error, UnicodeDecodeError, ValueError):
            step = min(step * 2, cap - len(buf))
            if step <= 0:
                break
            continue
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: gguf_peek_architecture.py <gguf-url>", file=sys.stderr)
        return 2
    arch = peek_architecture(sys.argv[1])
    print(arch if arch else "(general.architecture not found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
