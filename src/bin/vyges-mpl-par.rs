// SPDX-License-Identifier: Apache-2.0
//! `vyges-mpl-par run …` — the macro placer's `run`, unchanged, with this crate partitioning the
//! large flat clusters it would otherwise refuse.
//!
//! Each call is the partitioner's simple k-way mode: a fresh partitioner, the call's own balance
//! and seed, every other parameter at its default.

use std::process::ExitCode;

use vyges_mpl::flatpart::{FlatPartitioner, PartitionCall};
use vyges_par::multilevel::multilevel_partition;
use vyges_par::{simple_hypergraph, Params};

struct Par;

impl FlatPartitioner for Par {
    fn partition_k_way_simple(&mut self, c: &PartitionCall) -> Result<Vec<usize>, String> {
        let g = simple_hypergraph(&c.hyperedges, &c.vertex_weights, &c.hyperedge_weights);
        let p = Params::simple(c.num_parts, c.balance, c.seed);
        multilevel_partition(&p, &g).0.map_err(|r| format!("{r:?}"))
    }
}

fn main() -> ExitCode {
    vyges_mpl::run::init_logging();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => vyges_mpl::run::run_main(&args[1..], Some(&mut Par)),
        _ => {
            eprintln!("usage: vyges-mpl-par run <design.odb> [vyges-mpl run options]");
            ExitCode::from(2)
        }
    }
}
