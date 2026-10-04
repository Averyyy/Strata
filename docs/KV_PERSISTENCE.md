# Disk KV cache

The text engine can preserve completed conversations across graceful restarts. Entries contain the actual INT8 K/V, QSA index state, GDN recurrence and convolution state, PLE history, prompt checkpoints for every GPU stage, and the MTP draft K/V. Transfers stream through 64 KiB buffers into the existing state arenas.

Configure the engine flags in the server's existing `args` list:

```json
{
  "args": [
    "--kv-persist",
    "--kv-persist-dir", "data/kv-cache",
    "--kv-persist-max-mib", "30720",
    "--kv-persist-identity", "model-v1:int8"
  ],
  "engine_close_s": 300
}
```

Add these flags to the model's other engine arguments. `--kv-persist-dir` defaults to `data/kv-cache`; relative paths resolve against the engine working directory (`cwd` in the server config). Absolute paths are also supported. `--kv-persist-max-mib` defaults to 30720 MiB, or 30 GiB. These options also work when invoking the engine directly.

`--kv-persist-identity` identifies the model revision, quantization and state representation; change it when those facts change. Entries with different identities share the directory budget and LRU order. Engine geometry, GPU layer carves and KV layout must match before restoration. The cache supports text-only INT8 serving with MTP and prompt checkpoints enabled; RAM conversation parking must be disabled.

A completed outgoing conversation is saved before a switch or branch rewind. The current completed state is saved on QUIT or stdin EOF. Live continuation and repeated prompts use in-memory state without snapshot writes. Cancelled or incomplete work is not committed. After a process kill, the last committed entries remain available; work since the last save needs another prompt pass.

The cache chooses the longest exact token/checkpoint prefix that exceeds the resident prefix. It restores the entry's state and resumes from its live state or deepest matching checkpoint. Candidate metadata is read one entry at a time. Startup reads the index, without scanning K/V payloads.

One engine owns a cache directory at a time. The durable index records a monotonic usage sequence. The directory budget covers snapshots, temporary writes and 2 MiB reserved for the atomic index. Saves evict the least recently used entries when space is needed. `index.kv`, `entry-N.kv` and their lock/temporary files belong to the cache; files outside that namespace are left intact.

Writes use a temporary file, durable flush and atomic replacement. I/O failures and incompatible selected state end the operation with an error. `engine_close_s` controls the server's orderly shutdown deadline: 300 seconds by default with persistence enabled, 20 otherwise. Unloading succeeds after a successful engine exit.

Engine logs report `KV_PERSIST saved`, `restored` and `evicted`. Request metrics expose `kv_persist_restored_tokens`, `kv_persist_read_bytes`, `kv_persist_restore_ms`, `kv_persist_write_bytes` and `kv_persist_commit_ms` for snapshot transfers. Prompt/decode timing excludes disk switching; request wall time includes it.

Build `kv_persistence_test` with `STRATA_BUILD_CONVERSATION_TESTS=ON` for streamed state roundtrip, atomic failed-write, relative paths and durable global LRU eviction. Server lifecycle and metrics tests are in `serve/test_kv_persistence.py`.
