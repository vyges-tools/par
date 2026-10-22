// SPDX-License-Identifier: Apache-2.0
//! Initial partitions of the coarsest graph: random (balanced or "vile" — deliberately
//! unbalanced) and vile (round robin by weight). The refiner then makes them legal.

use crate::evaluator::Evaluator;
use crate::hypergraph::{add, lex_ge, Hypergraph, Weights};
use crate::ilp::{ilp_partition_inst, IlpInstance, IlpOutcome};
use crate::initial::Refused;
use crate::mt19937::Mt19937;
use crate::shuffle::deterministic_shuffle;

/// `RandomPart`. Fixed vertices keep their block; the others, shuffled with `seed`, fill block 0
/// until it reaches its LOWER bound, then block 1, and so on round robin. In vile mode a block
/// fills to its UPPER bound and the rest all go to the last block.
pub fn random_part(
    g: &Hypergraph,
    num_parts: usize,
    upper: &[Weights],
    lower: &[Weights],
    seed: i32,
    vile_mode: bool,
) -> Vec<usize> {
    let mut solution = vec![0usize; g.num_vertices()];
    let mut balance = vec![vec![0.0f32; g.vertex_dimensions()]; num_parts];
    let mut visited = vec![false; g.num_vertices()];
    if g.has_fixed_vertices() {
        for v in 0..g.num_vertices() {
            if g.fixed_attr(v) > -1 {
                solution[v] = g.fixed_attr(v) as usize;
                balance[solution[v]] = add(&balance[solution[v]], g.vertex_weights(v));
                visited[v] = true;
            }
        }
    }
    let mut vertices: Vec<usize> = (0..g.num_vertices()).filter(|&v| !visited[v]).collect();
    let mut gen = Mt19937::new(seed as u32);
    deterministic_shuffle(&mut vertices, &mut gen);

    let mut block = 0;
    if !vile_mode {
        for &v in &vertices {
            solution[v] = block;
            balance[block] = add(&balance[block], g.vertex_weights(v));
            if lex_ge(&balance[block], &lower[block]) {
                block = (block + 1) % num_parts;
            }
        }
    } else {
        let mut stop = false;
        for &v in &vertices {
            solution[v] = block;
            balance[block] = add(&balance[block], g.vertex_weights(v));
            // ⚠️ The weight stays counted in the block it left: the balance is only this loop's.
            if lex_ge(&balance[block], &upper[block]) && !stop {
                block += 1;
                solution[v] = block;
                if block == num_parts - 1 {
                    stop = true;
                }
            }
        }
    }
    solution
}

/// `ILPPart` (every hyperedge used — the default): the exact partition of `g`, replacing
/// `solution`. When the solver gives no optimum, a balanced random partition with the
/// partitioner's CURRENT seed (the last one a random start set) replaces it instead. Refused when
/// built without the solver (before its trace lines, which the reference prints first), or when
/// a vertex comes back in no block (the reference would index with -1).
#[allow(clippy::too_many_arguments)]
pub fn ilp_part(
    g: &Hypergraph,
    num_parts: usize,
    ev: &Evaluator,
    upper: &[Weights],
    lower: &[Weights],
    seed: i32,
    solution: &mut Vec<usize>,
    trace: &mut Vec<String>,
) -> Result<(), Refused> {
    if !cfg!(feature = "ilp") {
        return Err(Refused::NeedsIlp { num_vertices: g.num_vertices() });
    }
    trace.push("[partitioning] Starting Optimal ILP-based Partitioning".into());
    trace.push("[partitioning] All the hyperedges will be used in the ILP-based partitioning!".into());
    let hyperedges: Vec<Vec<usize>> = (0..g.num_hyperedges()).map(|e| g.vertices(e).to_vec()).collect();
    let hyperedge_weights: Vec<f32> = (0..g.num_hyperedges()).map(|e| ev.hyperedge_cost(e, g)).collect();
    let inst = IlpInstance {
        num_parts,
        vertex_weight_dimension: g.vertex_dimensions(),
        hyperedges: &hyperedges,
        hyperedge_weights: &hyperedge_weights,
        vertex_weights: g.all_vertex_weights(),
        upper,
        lower,
    };
    match ilp_partition_inst(&inst) {
        Ok(parts) => {
            if parts.iter().any(|&b| b < 0) {
                return Err(Refused::IlpVertexUnassigned);
            }
            *solution = parts.into_iter().map(|b| b as usize).collect();
        }
        Err(IlpOutcome::NotSolved) => {
            trace.push("[partitioning] Optimal ILP-based Partitioning failed. Calling random partitioning.".into());
            *solution = random_part(g, num_parts, upper, lower, seed, false);
        }
        Err(IlpOutcome::Unavailable) => return Err(Refused::NeedsIlp { num_vertices: g.num_vertices() }),
    }
    Ok(())
}

/// `VilePart`: free vertices sorted by (weight, id), dealt out round robin.
pub fn vile_part(g: &Hypergraph, num_parts: usize, ev: &Evaluator) -> Vec<usize> {
    let mut solution = vec![0usize; g.num_vertices()];
    let mut sizes = vec![0.0f32; g.num_vertices()];
    let mut unvisited = Vec::with_capacity(g.num_vertices());
    for v in 0..g.num_vertices() {
        if g.has_fixed_vertices() && g.fixed_attr(v) > -1 {
            solution[v] = g.fixed_attr(v) as usize;
        } else {
            unvisited.push(v);
            sizes[v] = ev.vertex_weight_norm(v, g);
        }
    }
    unvisited.sort_by(|&x, &y| sizes[x].partial_cmp(&sizes[y]).expect("finite weight").then(x.cmp(&y)));
    for (i, &v) in unvisited.iter().enumerate() {
        solution[v] = i % num_parts;
    }
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(weights: &[f32]) -> Hypergraph {
        let n = weights.len();
        let edges: Vec<Vec<usize>> = (1..n).map(|v| vec![v - 1, v]).collect();
        Hypergraph::new(1, 1, &edges, weights.iter().map(|&w| vec![w]).collect(), vec![vec![1.0]; n - 1], vec![], vec![])
    }

    /// Balanced mode moves on once a block REACHES its lower bound (>=), wrapping round.
    #[test]
    fn balanced_random_moves_on_at_the_lower_bound() {
        let g = line(&[1.0; 6]);
        let lower = vec![vec![2.0]; 2];
        let upper = vec![vec![4.0]; 2];
        let sol = random_part(&g, 2, &upper, &lower, 7, false);
        let mut order: Vec<usize> = (0..6).collect();
        deterministic_shuffle(&mut order, &mut Mt19937::new(7));
        // Block 0 is already at its bound when it comes round again, so it takes one vertex.
        let expect = [0, 0, 1, 1, 0, 1];
        for (i, &v) in order.iter().enumerate() {
            assert_eq!(sol[v], expect[i], "position {i}");
        }
    }

    /// Vile mode moves the vertex that reaches the upper bound into the NEXT block, then stops.
    #[test]
    fn vile_random_fills_to_the_upper_bound_then_stops() {
        let g = line(&[1.0; 6]);
        let lower = vec![vec![2.0]; 2];
        let upper = vec![vec![2.0]; 2];
        let sol = random_part(&g, 2, &upper, &lower, 3, true);
        let mut order: Vec<usize> = (0..6).collect();
        deterministic_shuffle(&mut order, &mut Mt19937::new(3));
        let expect = [0, 1, 1, 1, 1, 1];
        for (i, &v) in order.iter().enumerate() {
            assert_eq!(sol[v], expect[i], "position {i}");
        }
    }

    /// With three blocks: the vertex that fills a block is MOVED on, but its weight stays counted
    /// in the block it filled, so the next block starts one vertex lighter. (Two blocks cannot show
    /// it: the last block is never checked.)
    #[test]
    fn vile_random_leaves_the_mover_weight_behind() {
        let g = line(&[1.0; 6]);
        let bound = vec![vec![2.0]; 3];
        let sol = random_part(&g, 3, &bound, &bound, 5, true);
        let mut order: Vec<usize> = (0..6).collect();
        deterministic_shuffle(&mut order, &mut Mt19937::new(5));
        let expect = [0, 1, 1, 2, 2, 2];
        for (i, &v) in order.iter().enumerate() {
            assert_eq!(sol[v], expect[i], "position {i}");
        }
    }

    /// Ties on weight go to the smaller id; dealing is round robin in that order.
    #[test]
    fn vile_deals_by_weight_then_id() {
        let g = line(&[3.0, 1.0, 1.0, 2.0]);
        let ev = Evaluator { num_parts: 2, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] };
        // order 1, 2, 3, 0
        assert_eq!(vile_part(&g, 2, &ev), vec![1, 0, 1, 0]);
    }
}
