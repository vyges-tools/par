// SPDX-License-Identifier: Apache-2.0
//! Runs the implemented stages on recorded partitioning calls and prints their trace lines.
//!
//! Input: one JSON object per line — `num_parts`, `balance`, `seed`, `hyperedges`,
//! `vertex_weights`, `hyperedge_weights` (a `solution` field, if present, is ignored).
//! Output, per call: `call N` then, per candidate coarsening, its levels as
//! `Level L :: num_vertices = V, num_hyperedges = E`, then its initial-partitioning trace lines.
//!
//! Usage: par-stage <calls.jsonl>

use std::io::{BufRead, BufReader};

use vyges_par::{candidates, simple_hypergraph, Params};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: par-stage <calls.jsonl>");
        std::process::exit(2);
    });
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        eprintln!("par-stage: {path}: {e}");
        std::process::exit(2);
    });
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.expect("readable line");
        let call: serde_json::Value = serde_json::from_str(&line).unwrap_or_else(|e| {
            eprintln!("par-stage: line {}: {e}", n + 1);
            std::process::exit(2);
        });
        let ints = |v: &serde_json::Value| -> Vec<usize> {
            v.as_array().expect("array").iter().map(|x| x.as_u64().expect("index") as usize).collect()
        };
        let floats = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number") as f32).collect()
        };
        let hyperedges: Vec<Vec<usize>> = call["hyperedges"].as_array().expect("hyperedges").iter().map(ints).collect();
        let g = simple_hypergraph(&hyperedges, &floats(&call["vertex_weights"]), &floats(&call["hyperedge_weights"]));
        let p = Params::simple(
            call["num_parts"].as_u64().expect("num_parts") as usize,
            call["balance"].as_f64().expect("balance") as f32,
            call["seed"].as_i64().expect("seed") as i32,
        );
        println!("call {}", n + 1);
        for c in candidates(&p, &g) {
            println!("coarsening");
            for (l, (nv, ne)) in c.levels.iter().enumerate() {
                println!("Level {l} :: num_vertices = {nv}, num_hyperedges = {ne}");
            }
            match c.initial {
                Ok(ip) => ip.trace().iter().for_each(|l| println!("initial {l}")),
                Err(r) => println!("initial refused {r:?}"),
            }
        }
    }
}
