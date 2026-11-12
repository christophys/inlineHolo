# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Minimal reader for Gatan Digital Micrograph files (DM3 and DM4), pure NumPy.

Used by :mod:`inlineholo.io_formats` when RosettaSciIO is not installed (and as a fast path
for large stacks: the image data are memory-mapped instead of copied tag by tag).

The file is a tree of tag groups; leaves are typed scalars, strings, structs or arrays.
``read_dm(path)`` returns ``(images, tags)`` with ``tags`` the complete tag tree (nested dicts,
lists for TagGroup lists) and ``images`` a list of dicts with ``data`` (NumPy array, shape
(nz, ny, nx) or (ny, nx)), ``scales``, ``origins``, ``units`` (per axis, in array order),
``name`` and ``tags`` (the ImageTags of that image).
"""
from __future__ import annotations

import struct

import numpy as np

_SIMPLE = {2: "h", 3: "i", 4: "H", 5: "I", 6: "f", 7: "d", 8: "?", 9: "b", 10: "b", 11: "q", 12: "Q"}
_NP = {2: np.int16, 3: np.int32, 4: np.uint16, 5: np.uint32, 6: np.float32, 7: np.float64, 8: np.bool_,
       9: np.int8, 10: np.int8, 11: np.int64, 12: np.uint64}
_LAZY_BYTES = 1 << 16        # arrays larger than this are memory-mapped, not decoded


class _Lazy:
    """Placeholder for a large array in the file (offset, dtype, count)."""

    def __init__(self, path, offset, dtype, count):
        self.path, self.offset, self.dtype, self.count = path, offset, dtype, count

    def load(self):
        return np.memmap(self.path, dtype=self.dtype, mode="r", offset=self.offset, shape=(self.count,))

    def __repr__(self):
        return f"<array {self.dtype} x {self.count}>"


class _Reader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.version = struct.unpack(">i", self.f.read(4))[0]
        if self.version not in (3, 4):
            raise ValueError(f"not a DM3/DM4 file (version {self.version})")
        self.f.read(8 if self.version == 4 else 4)                      # file size
        self.le = struct.unpack(">i", self.f.read(4))[0] == 1
        self.e = "<" if self.le else ">"
        self.isize = 8 if self.version == 4 else 4
        self.ifmt = ">Q" if self.version == 4 else ">I"

    def _int(self):
        return struct.unpack(self.ifmt, self.f.read(self.isize))[0]

    def _group(self):
        self.f.read(2)                                                  # sorted, open
        n = self._int()
        out = {}
        unnamed = 0
        for _ in range(n):
            kind = self.f.read(1)[0]
            ln = struct.unpack(">H", self.f.read(2))[0]
            name = self.f.read(ln).decode("latin-1") if ln else None
            if self.version == 4:
                self.f.read(8)                                          # tag size
            if kind == 20:
                val = self._group()
            elif kind == 21:
                val = self._data()
            else:
                raise ValueError(f"bad tag type {kind} at {self.f.tell()}")
            if name is None:
                name = f"TagGroup{unnamed}"
                unnamed += 1
            out[name] = val
        return out

    def _data(self):
        if self.f.read(4) != b"%%%%":
            raise ValueError(f"bad data tag at {self.f.tell()}")
        n = self._int()
        info = [self._int() for _ in range(n)]
        return self._value(info)

    def _scalar(self, t):
        fmt = _SIMPLE[t]
        v = struct.unpack(self.e + fmt, self.f.read(struct.calcsize(fmt)))[0]
        return v

    def _value(self, info):
        t = info[0]
        if t in _SIMPLE:
            return self._scalar(t)
        if t == 18:                                                     # string (length in info)
            n = info[1]
            return self.f.read(n).decode("latin-1", "replace")
        if t == 15:                                                     # struct
            nf = info[2]
            types = [info[4 + 2 * i] for i in range(nf)]
            return tuple(self._scalar(tt) for tt in types)
        if t == 20:                                                     # array
            et = info[1]
            if et == 15:
                nf = info[3]
                types = [info[5 + 2 * i] for i in range(nf)]
                count = info[4 + 2 * nf]
                return [tuple(self._scalar(tt) for tt in types) for _ in range(count)]
            count = info[2]
            dt = np.dtype(_NP[et]).newbyteorder(self.e)
            nbytes = dt.itemsize * count
            if nbytes > _LAZY_BYTES:
                off = self.f.tell()
                self.f.seek(nbytes, 1)
                return _Lazy(self.path, off, dt, count)
            a = np.frombuffer(self.f.read(nbytes), dtype=dt)
            if et == 4 and count < 4096:                                # UTF-16 text (units, names)
                try:
                    return a.astype(np.uint16).tobytes().decode("utf-16-le")
                except Exception:
                    return a.copy()
            return a.copy()
        raise ValueError(f"unknown DM data type {t}")

    def read(self):
        try:
            return self._group()
        finally:
            self.f.close()


def _as_list(group):
    if isinstance(group, dict):
        return [group[k] for k in sorted(group, key=lambda s: int(s[8:]) if s.startswith("TagGroup") and s[8:].isdigit() else 0)]
    return list(group or [])


def _plain(v):
    if isinstance(v, _Lazy):
        return repr(v)
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 16 else f"array{v.shape}"
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def read_dm(path):
    """Read a DM3/DM4 file -> (images, tags)."""
    tags = _Reader(path).read()
    images = []
    for img in _as_list(tags.get("ImageList", {})):
        d = img.get("ImageData", {})
        raw = d.get("Data")
        dims = [int(v) for v in _as_list(d.get("Dimensions", {}))] if isinstance(d.get("Dimensions"), dict) else []
        if raw is None or not dims:
            continue
        arr = raw.load() if isinstance(raw, _Lazy) else np.asarray(raw)
        if arr.size != int(np.prod(dims)):
            continue
        arr = np.asarray(arr).reshape(dims[::-1])                      # x fastest -> (z, y, x)
        cal = _as_list(d.get("Calibrations", {}).get("Dimension", {}))
        scales, origins, units = [], [], []
        for c in cal[::-1]:
            scales.append(float(c.get("Scale", 1.0)))
            origins.append(float(c.get("Origin", 0.0)))
            u = c.get("Units", "")
            units.append(u if isinstance(u, str) else "")
        images.append(dict(data=arr, scales=scales, origins=origins, units=units, name=img.get("Name", ""),
                           tags=_plain(img.get("ImageTags", {}))))
    return images, _plain(tags)
