"""Precompute frozen TraceAnything window tokens for a recorded dataset (see `TokenCacheBuilder`).

init    check the window layout against the sampler, create meta.json + an empty float16 memmap
encode  encode this rank's episodes (episode % world == rank); resumable
verify  re-encode random windows online and compare; writes COMPLETE once everything is encoded
bench   random-row read throughput from finished episodes
"""

import tyro

from markovian_policy.stages import cache_tokens

if __name__ == "__main__":
    cache_tokens.run(tyro.cli(cache_tokens.Config))
