# Canonical 31.727B dolmino corpus — persisted off tmpfs, 2026-08-17

Verbatim output of the persist run (guards: refuse-if-dest-exists → size → sha256 → shape assert;
wrote `.partial` and `mv`d only after the hash matched):

```
src bytes    = 126907244672
avail bytes  = 10695114489856
headroom OK
=== copying 2026-08-17T13:37:58+08:00 ===
copied bytes = 126907244672  in 365s
=== sha256 both sides (this is the real check, not size) 2026-08-17T13:44:03+08:00 ===
src sha=b16a91f0c3d83953893d2ee3df1f250159db4a523976c4395b79eee5b699a574
dst sha=b16a91f0c3d83953893d2ee3df1f250159db4a523976c4395b79eee5b699a574
=== shape re-read from the PERSISTED file ===
persisted shape: (15491607, 2048) uint32 tokens=31.727B
SHAPE ASSERT OK
=== DONE 2026-08-17T13:48:14+08:00 rc=0 ===
```

**Why this was urgent**: `data/dolmino_now15b.npy` is only the 7,570,911-row **prefix**
(15.505B tokens). The 15,491,607-row canonical array existed **only** in `/dev/shm` on LOCAL,
and tmpfs is wiped when the pod is rebuilt — which happened to LOCAL on 2026-08-14.
Every B03 ladder cell must load this corpus; losing it would force a rebuild from
`data/dolmino_olmo2_shards/` (86 files).

The tmpfs original is left in place — it is still the faster read path for anything running now.
