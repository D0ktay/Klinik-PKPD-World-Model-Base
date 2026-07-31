"""Satir-bazli flush'li, zaman damgali log yazici -- surec beklenmedik
sekilde olurse kismi log KAYBOLMASIN diye."""
import os
import sys
import time

# Windows konsolu kod sayfasi (ornegin cp1254) UTF-8 karakterlerin hepsini
# encode edemiyor -- stdout'u once yeniden yapilandirmayi dene, olmuyorsa
# print() cagrilarini hatasiz gecmesi icin errors='replace' ile sar.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class LineFlushLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8", buffering=1)

    def log(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
        self._f.write(line + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        self._f.close()
