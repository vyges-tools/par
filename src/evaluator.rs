// SPDX-License-Identifier: Apache-2.0
//! The cost model, without timing or placement terms: weight factors per hyperedge and vertex
//! dimension.

use std::collections::BTreeMap;

use crate::hypergraph::{add, Hypergraph, Weights};

#[derive(Clone, Debug)]
pub struct Evaluator {
    pub num_parts: usize,
    pub e_wt_factors: Vec<f32>,
    pub v_wt_factors: Vec<f32>,
}

/// `std::inner_product(a, b, 0.0)` with a DOUBLE initial value: each product is a float product,
/// the running sum is a double, and the caller narrows the result back to float.
fn inner_product_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |acc, (x, y)| acc + (x * y) as f64)
}

/// The same with a FLOAT initial value: the sum stays a float throughout.
fn inner_product_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |acc, (x, y)| acc + x * y)
}

impl Evaluator {
    /// `CalculateHyperedgeCost` (no timing): the weighted hyperedge weight.
    pub fn hyperedge_cost(&self, e: usize, g: &Hypergraph) -> f32 {
        inner_product_f64(g.hyperedge_weights(e), &self.e_wt_factors) as f32
    }

    /// `GetNormEdgeScore`: the cost shared out over the hyperedge's other vertices.
    pub fn norm_edge_score(&self, e: usize, g: &Hypergraph) -> f32 {
        let size = g.vertices(e).len();
        if size <= 1 {
            return 0.0;
        }
        self.hyperedge_cost(e, g) / (size - 1) as f32
    }

    /// `GetVertexWeightNorm`: the weighted vertex weight (float accumulation).
    pub fn vertex_weight_norm(&self, v: usize, g: &Hypergraph) -> f32 {
        inner_product_f32(g.vertex_weights(v), &self.v_wt_factors)
    }

    /// `GetNetDegrees`: per hyperedge, how many of its vertices each block holds.
    pub fn net_degrees(&self, g: &Hypergraph, solution: &[usize]) -> Vec<Vec<i32>> {
        (0..g.num_hyperedges())
            .map(|e| {
                let mut degs = vec![0; self.num_parts];
                for &v in g.vertices(e) {
                    degs[solution[v]] += 1;
                }
                degs
            })
            .collect()
    }

    /// `GetBlockBalance`: per block, the weight of its vertices summed in vertex order.
    pub fn block_balance(&self, g: &Hypergraph, solution: &[usize]) -> Vec<Weights> {
        let mut balance = vec![vec![0.0f32; g.vertex_dimensions()]; self.num_parts];
        for v in 0..g.num_vertices() {
            balance[solution[v]] = add(&balance[solution[v]], g.vertex_weights(v));
        }
        balance
    }

    /// `GetCutHyperedges`: the hyperedges with a vertex outside the first vertex's block, in order.
    pub fn cut_hyperedges(&self, g: &Hypergraph, solution: &[usize]) -> Vec<usize> {
        (0..g.num_hyperedges())
            .filter(|&e| match g.vertices(e).split_first() {
                Some((&first, rest)) => rest.iter().any(|&v| solution[v] != solution[first]),
                None => false,
            })
            .collect()
    }

    /// `CalculateHyperedgeVertexWtSum`: the weighted weights of a hyperedge's vertices, summed in
    /// float in its vertex order.
    pub fn hyperedge_vertex_wt_sum(&self, e: usize, g: &Hypergraph) -> f32 {
        let mut weight = 0.0f32;
        for &v in g.vertices(e) {
            weight += self.vertex_weight_norm(v, g);
        }
        weight
    }

    /// `GetMatchingConnectivity`: for EVERY block pair a < b (zero included), the cost of the
    /// hyperedges touching both, summed in float in hyperedge order. A `std::map` ⟹ pair order.
    pub fn matching_connectivity(&self, g: &Hypergraph, solution: &[usize]) -> BTreeMap<(usize, usize), f32> {
        let mut out = BTreeMap::new();
        for a in 0..self.num_parts {
            for b in a + 1..self.num_parts {
                let mut score = 0.0f32;
                for e in 0..g.num_hyperedges() {
                    let touches = |p: usize| g.vertices(e).iter().any(|&v| solution[v] == p);
                    if touches(a) && touches(b) {
                        score += self.hyperedge_cost(e, g);
                    }
                }
                out.insert((a, b), score);
            }
        }
        out
    }

    /// `CutEvaluator` (no timing): the cut cost, summed in float in hyperedge order, and the block
    /// balance.
    pub fn cut_evaluator(&self, g: &Hypergraph, solution: &[usize]) -> (f32, Vec<Weights>) {
        let balance = self.block_balance(g, solution);
        let mut edge_cost = 0.0f32;
        for e in self.cut_hyperedges(g, solution) {
            edge_cost += self.hyperedge_cost(e, g);
        }
        (edge_cost, balance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hyperedge's score is its cost divided by (size - 1); a single-vertex one scores 0.
    #[test]
    fn the_edge_score_shares_the_cost_over_the_other_vertices() {
        let g = Hypergraph::new(1, 1, &[vec![0, 1, 2], vec![2]], vec![vec![1.0]; 3], vec![vec![3.0], vec![5.0]], vec![], vec![]);
        let ev = Evaluator { num_parts: 2, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] };
        assert_eq!(ev.norm_edge_score(0, &g), 1.5);
        assert_eq!(ev.norm_edge_score(1, &g), 0.0);
    }

    /// A hyperedge is cut when any vertex leaves its FIRST vertex's block; its cost counts once.
    #[test]
    fn the_cut_counts_each_split_hyperedge_once() {
        let g = Hypergraph::new(1, 1, &[vec![0, 1, 2], vec![1, 2], vec![2]], vec![vec![1.0], vec![2.0], vec![4.0]], vec![vec![3.0], vec![5.0], vec![7.0]], vec![], vec![]);
        let ev = Evaluator { num_parts: 2, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] };
        let (cost, balance) = ev.cut_evaluator(&g, &[0, 1, 1]);
        assert_eq!(cost, 3.0);
        assert_eq!(balance, vec![vec![1.0], vec![6.0]]);
        assert_eq!(ev.net_degrees(&g, &[0, 1, 1]), vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    }
}
