// SPDX-License-Identifier: Apache-2.0
//! Uncoarsening with refinement (`RefinePartition`): the kept solutions are carried from each
//! level to the next finer one and refined there, the best tracked per level.

use crate::evaluator::Evaluator;
use crate::hypergraph::{Hypergraph, Weights};
use crate::refiner::{Greedy, KWayFm, KWayPm};

/// The three refiners `CallRefiner` runs, in its order.
pub struct Refiners {
    pub pm: KWayPm,
    pub fm: KWayFm,
    pub greedy: Greedy,
}

impl Refiners {
    /// `CallRefiner`: pairwise FM (only with more than one block), k-way FM, greedy.
    pub fn call(&self, num_parts: usize, g: &Hypergraph, upper: &[Weights], lower: &[Weights], solution: &mut [usize]) {
        if num_parts > 1 {
            self.pm.refine(g, upper, lower, solution);
        }
        self.fm.refine(g, upper, lower, solution);
        self.greedy.refine(g, upper, lower, solution);
    }
}

/// `RefinePartition` over `hierarchy` (finest first). From the coarsest up, every solution is
/// projected onto the next finer graph (each cluster's block given to the vertices it was made
/// of) and refined; the reference refines them in parallel, each on its own solution with
/// read-only refiners, so in turn is the same. The best is the FIRST of the lowest cost. Returns
/// the per-level trace lines.
#[allow(clippy::too_many_arguments)]
pub fn refine_partition(
    hierarchy: &[&Hypergraph],
    num_parts: usize,
    ev: &Evaluator,
    refiners: &Refiners,
    upper: &[Weights],
    lower: &[Weights],
    top: &mut [Vec<usize>],
    best_solution_id: &mut usize,
) -> Vec<String> {
    let mut lines = Vec::new();
    for (num_level, pair) in hierarchy.windows(2).rev().enumerate() {
        let (g, coarse) = (pair[0], pair[1]);
        for solution in top.iter_mut() {
            let mut refined = vec![0usize; g.num_vertices()];
            for (cluster, &part) in solution.iter().enumerate().take(coarse.num_vertices()) {
                for &v in coarse.vertex_c_attr(cluster) {
                    refined[v] = part;
                }
            }
            *solution = refined;
            refiners.call(num_parts, g, upper, lower, solution);
        }
        let mut best_cost = f32::MAX;
        for (i, solution) in top.iter().enumerate() {
            let cost = ev.cut_evaluator(g, solution).0;
            if best_cost > cost {
                best_cost = cost;
                *best_solution_id = i;
            }
        }
        lines.push(format!(
            "Level {} :: num_vertices = {}, num_hyperedges = {}, cutcost = {}, best_solution_id = {}",
            num_level + 1,
            g.num_vertices(),
            g.num_hyperedges(),
            best_cost,
            best_solution_id
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use crate::hypergraph::Hypergraph;
    use crate::Params;

    /// `CallRefiner` order is pairwise FM, then k-way FM, then greedy. With three blocks the
    /// pairwise pass touches only the best-connected pair, so the order changes the result
    /// (with two blocks and a legal start the two FM passes coincide and no order shows).
    /// The expected solution is this crate's under that order — the ORDER is the reference rule.
    #[test]
    fn the_pairwise_refiner_runs_first() {
        let edges = vec![vec![4, 3], vec![3, 6], vec![0, 5], vec![0, 3], vec![4, 6], vec![5, 6], vec![2, 5], vec![6, 3], vec![4, 1]];
        let ew: Vec<Vec<f32>> = [1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0].iter().map(|&w| vec![w]).collect();
        let g = Hypergraph::new(1, 1, &edges, vec![vec![1.0]; 7], ew, vec![], vec![]);
        let p = Params::simple(3, 20.0, 0);
        let (upper, lower) = p.block_bounds(&g);
        let mut sol = vec![1, 2, 2, 0, 0, 1, 0];
        p.refiners(p.k_way_fm()).call(3, &g, &upper, &lower, &mut sol);
        assert_eq!(sol, vec![0, 2, 0, 1, 2, 0, 1]);
    }
}
