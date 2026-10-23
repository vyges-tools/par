# vyges-par

A deterministic **hypergraph partitioner** in Rust: multilevel coarsening, initial partitioning
(random, vile and exact ILP starts), k-way FM, pairwise and greedy refinement, cut-overlay
clustering and V-cycles.

> **Part of Vyges Loom.** A library, not a command: [`vyges-mpl`](https://github.com/vyges-tools/mpl)
> links it to split the large flat clusters of a macro-placement run.

## Same input, same answer

A partitioner's answer decides where a placer puts things, so it has to be reproducible. For a
given call (hyperedges, weights, number of parts, balance, seed) `vyges-par` returns the same
blocks on every run and every machine:

- its own Mersenne Twister and shuffle, not the platform's;
- single-precision weights where the algorithm keeps them in single precision, so sums round the
  same way;
- no result depends on hash-map iteration order: hash maps are looked up, never iterated;
- the exact stages solved by a pinned solver build
  ([`vyges-ortools-lib`](https://github.com/vyges-tools/ortools-lib)), whose optimum is the same
  one, tie included, on every run.

## Use it

```toml
[dependencies]
vyges-par = { git = "https://github.com/vyges-tools/par", branch = "main", features = ["ilp"] }
```

```rust
use vyges_par::multilevel::multilevel_partition;
use vyges_par::{simple_hypergraph, Params};

// four vertices, two hyperedges, unit weights: split in two, 1 % imbalance, seed 0
let g = simple_hypergraph(&[vec![0, 1], vec![2, 3]], &[1.0; 4], &[1.0; 2]);
let (blocks, trace) = multilevel_partition(&Params::simple(2, 1.0, 0), &g);
// `blocks`: Ok(block of each vertex), or Err(Refused) for a stage this build cannot run
// `trace`: one line per stage, in call order
```

`Params::simple` is a k-way partition with no timing, placement, fixed vertices or groups.

## Features

| Feature | What it adds |
|---|---|
| `ilp` | the exact (ILP) stages, through `vyges-ortools-lib` (Linux x86_64). Without it a call that reaches one returns `Refused::NeedsIlp` instead of an approximation. |
| `cli` | `par-stage`, a developer tool that replays recorded calls (JSON lines) and prints each trace. Not installed or released. |

With `ilp`, set `VYGES_ORTOOLS_PREBUILT_DIR` to the prebuilt solver bundle
(`vyges-tools-ortools-lib/scripts/fetch-bundle.sh <dir>`) to skip the solver's source build.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`. Integrating and need help? → <https://vyges.com/contact>.
