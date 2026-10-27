"""Bounded Q5_K row reads and exact prepared decoding outside CUDA graphs."""
from __future__ import annotations

import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared
from .._shared.disk_table import DiskRowCache
from ._q5 import Q5KEmbeddingWeight
from ._preparation import plan, query_from_call


class Q5KDiskTable:
    """Own a bounded raw row cache; callers supply global IDs and BF16 output.

    Each source row is one 256-element GGUF Q5_K block. Disk I/O and decode
    run eagerly; downstream CUDA graphs may consume the stable output.
    """

    def __init__(self, *, path, offset, table_rows, max_lookups, device,
                 shard_start=0, shard_end=None, queue_depth=128):
        self.cache = DiskRowCache(
            device=device, max_lookups=max_lookups, table_rows=table_rows,
            shard_start=shard_start, shard_end=table_rows if shard_end is None else shard_end,
            shard_rows=table_rows, weight_row_bytes=176, queue_depth=queue_depth,
        )
        try:
            self.cache.add_shard(0, path, offset)
            self.cache.freeze()
            self.weight = Q5KEmbeddingWeight(self.cache.weight)
            self.ids = torch.arange(max_lookups, dtype=torch.int64, device=self.cache.device)
            # Use zero live rows for preparation: disk staging is not populated yet.
            out = torch.empty(max_lookups, 256, dtype=torch.bfloat16, device=self.cache.device)
            self.plan = plan(query_from_call(self.weight, self.ids, out=out), device=self.cache.device)
            with PreparationSession(device=self.cache.device, autotune=False, compile_workers=1) as session:
                session.prepare((self.plan.request(name='Q5_K disk rows', prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(self.weight, self.ids[:0], out=out[:0]))),))
                session.freeze()
                self.state = require_prepared(self.plan, 'sequence.embedding', self.cache.device)
        except Exception:
            self.cache.close()
            raise

    def lookup(self, ids, out):
        count = ids.numel()
        if count > self.cache.max_lookups or out.shape != (*ids.shape, 256):
            raise ValueError('Q5_K disk lookup exceeds capacity or has wrong output shape')
        if out.dtype != torch.bfloat16 or out.device != self.cache.device or not out.is_contiguous():
            raise ValueError('Q5_K disk output must be contiguous BF16 on the cache device')
        with self.cache.transaction():
            self.cache.read_rows(ids, count)
            self.state.run(self.weight, self.ids[:count], out=out.view(count, 256))

    def close(self):
        self.cache.close()

    def stats(self):
        return self.cache.stats()
