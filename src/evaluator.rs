// SPDX-License-Identifier: Apache-2.0
//! The cost model, without timing or placement terms: weight factors per hyperedge and vertex
//! dimension.

use crate::hypergraph::Hypergraph;

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
}
