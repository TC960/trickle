"""Residency manager: the AirLLM half of the system.

Holds a byte budget and decides which layer shards are in memory. Layers are
loaded just before they run and evicted once the budget is exceeded, so peak
footprint is bounded by the budget rather than by model size.

The budget is a dial, not a switch:

  budget >= model size   every layer stays resident after first touch; the
                         system converges to an ordinary quantized model
  budget ~= a few layers true streaming, ~1 layer of I/O per layer of compute
  budget == 1 layer      minimum footprint, maximum I/O

A background thread prefetches the next shard while the current one computes,
which hides most of the load latency whenever compute per layer exceeds the
~127 MB read a ternary layer needs.
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Queue

import torch
from safetensors import safe_open


class ShardStore:
    """Lazy, mmap-backed reader over the shard directory."""

    def __init__(self, shard_dir, manifest: dict, device="cpu"):
        self.shard_dir = Path(shard_dir)
        self.manifest = manifest
        self.device = device
        self._handles = {}
        self._lock = threading.Lock()

    def _handle(self, shard_key: str):
        # safe_open handles are not thread-safe to create, but are cheap to keep.
        with self._lock:
            if shard_key not in self._handles:
                filename = self.manifest["shards"][shard_key]["file"]
                self._handles[shard_key] = safe_open(
                    str(self.shard_dir / filename), framework="pt", device="cpu"
                )
            return self._handles[shard_key]

    def read(self, shard_key: str) -> dict:
        """Read every tensor in a shard into memory on the target device."""
        handle = self._handle(shard_key)
        return {name: handle.get_tensor(name).to(self.device) for name in handle.keys()}

    def nbytes(self, shard_key: str) -> int:
        return self.manifest["shards"][shard_key]["nbytes"]

    def close(self):
        with self._lock:
            self._handles.clear()


class ResidencyManager:
    """LRU cache over layer shards, bounded by a byte budget."""

    def __init__(
        self,
        store: ShardStore,
        *,
        budget_bytes: int,
        prefetch: bool = True,
        pinned: tuple = ("globals",),
    ):
        self.store = store
        self.budget_bytes = budget_bytes
        self.pinned = set(pinned)

        # shard_key -> loaded tensor dict, in LRU order (oldest first).
        self._resident = OrderedDict()
        self._resident_bytes = 0
        self._lock = threading.RLock()

        # Consumers register a callback that binds tensors into live modules.
        self._binders = {}

        self.stats = {
            "hits": 0, "misses": 0, "evictions": 0,
            "bytes_read": 0, "load_seconds": 0.0, "prefetch_hits": 0,
        }

        self._prefetch_queue = Queue(maxsize=4) if prefetch else None
        self._staged = {}
        self._stop = threading.Event()
        self._thread = None
        if prefetch:
            self._thread = threading.Thread(
                target=self._prefetch_worker, daemon=True, name="shard-prefetch"
            )
            self._thread.start()

    def register_binder(self, shard_key: str, binder):
        """Attach the function that installs a shard's tensors into modules."""
        self._binders[shard_key] = binder

    def _prefetch_worker(self):
        """Read upcoming shards off the critical path."""
        while not self._stop.is_set():
            try:
                shard_key = self._prefetch_queue.get(timeout=0.25)
            except Empty:
                continue
            if shard_key is None:
                break
            with self._lock:
                already = shard_key in self._resident or shard_key in self._staged
            if already:
                continue
            try:
                tensors = self.store.read(shard_key)
                with self._lock:
                    self._staged[shard_key] = tensors
            except Exception:
                # A failed prefetch is not fatal; ensure() will read it inline.
                pass

    def hint(self, shard_key: str):
        """Signal that `shard_key` will be needed soon."""
        if self._prefetch_queue is None:
            return
        with self._lock:
            if shard_key in self._resident or shard_key in self._staged:
                return
        try:
            self._prefetch_queue.put_nowait(shard_key)
        except Exception:
            pass  # queue full: prefetch is best-effort by design

    def ensure(self, shard_key: str):
        """Guarantee a shard is resident and bound. Evicts others as needed."""
        with self._lock:
            if shard_key in self._resident:
                self._resident.move_to_end(shard_key)
                self.stats["hits"] += 1
                return

            self.stats["misses"] += 1
            staged = self._staged.pop(shard_key, None)

        started = time.perf_counter()
        if staged is not None:
            tensors = staged
            self.stats["prefetch_hits"] += 1
        else:
            tensors = self.store.read(shard_key)

        nbytes = self.store.nbytes(shard_key)

        with self._lock:
            # Evict before inserting so peak stays under budget. Pinned shards
            # (embeddings) are never candidates.
            self._evict_until(self.budget_bytes - nbytes)

            binder = self._binders.get(shard_key)
            if binder is not None:
                binder(tensors)
            self._resident[shard_key] = tensors
            self._resident_bytes += nbytes

            self.stats["bytes_read"] += nbytes
            self.stats["load_seconds"] += time.perf_counter() - started

    def _evict_until(self, target_bytes: int):
        """Drop least-recently-used shards until resident bytes fit `target_bytes`."""
        while self._resident_bytes > target_bytes and self._resident:
            for key in list(self._resident):
                if key in self.pinned:
                    continue
                tensors = self._resident.pop(key)
                self._resident_bytes -= self.store.nbytes(key)
                binder = self._binders.get(key)
                if binder is not None:
                    binder(None)  # None means "evict"
                del tensors
                self.stats["evictions"] += 1
                break
            else:
                return  # only pinned shards left; nothing more to give back

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def report(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "hit_rate": self.stats["hits"] / total if total else 0.0,
            "resident_mb": round(self._resident_bytes / 1e6, 1),
            "budget_mb": round(self.budget_bytes / 1e6, 1),
            "gb_read": round(self.stats["bytes_read"] / 1e9, 2),
        }

    def close(self):
        self._stop.set()
        if self._prefetch_queue is not None:
            try:
                self._prefetch_queue.put_nowait(None)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            self._resident.clear()
            self._staged.clear()
            self._resident_bytes = 0
        self.store.close()
