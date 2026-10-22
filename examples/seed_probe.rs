// PROBE: level-1 size of one coarsening per seed, on the first call of a calls file.
use std::io::{BufRead, BufReader};
use vyges_par::{simple_hypergraph, Params};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let line = BufReader::new(std::fs::File::open(path).unwrap()).lines().next().unwrap().unwrap();
    let call: serde_json::Value = serde_json::from_str(&line).unwrap();
    let ints = |v: &serde_json::Value| -> Vec<usize> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect() };
    let floats = |v: &serde_json::Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
    let he: Vec<Vec<usize>> = call["hyperedges"].as_array().unwrap().iter().map(ints).collect();
    let g = simple_hypergraph(&he, &floats(&call["vertex_weights"]), &floats(&call["hyperedge_weights"]));
    let p = Params::simple(2, 1.0, 0);
    let mut co = p.coarsener(&g);
    co.thr_coarsen_hyperedge_size_skip = p.global_net_threshold;
    let grouped = co.group_vertices(&g);
    co.thr_coarsen_hyperedge_size_skip = p.thr_coarsen_hyperedge_size_skip;
    co.max_coarsen_iters = 1;
    for seed in 0..12 {
        co.random_seed = seed;
        let l = co.lazy_first_choice(&grouped);
        println!("seed {seed}: level 1 = {} / {}", l[0].num_vertices(), l[0].num_hyperedges());
    }
}
