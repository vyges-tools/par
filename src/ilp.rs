// SPDX-License-Identifier: Apache-2.0
//! The exact partition (`ILPPartitionInst`): a 0/1 program over "vertex v in block i" and
//! "hyperedge e wholly in block i", maximising the weight of uncut hyperedges, with a tiny
//! per-block bonus that prefers higher block ids among otherwise equal optima. Built call for call
//! in the reference's order, so the solver receives the same model; ties among optima are the
//! solver's, which is why the solver is linked rather than rewritten.

use crate::hypergraph::Weights;

/// Why no exact solution came back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IlpOutcome {
    /// The model was trivial or the solver did not prove optimality: the caller falls back.
    NotSolved,
    /// Built without the solver.
    Unavailable,
}

/// The model's inputs, as the caller passes them.
pub struct IlpInstance<'a> {
    pub num_parts: usize,
    pub vertex_weight_dimension: usize,
    pub hyperedges: &'a [Vec<usize>],
    pub hyperedge_weights: &'a [f32],
    pub vertex_weights: &'a [Weights],
    pub upper: &'a [Weights],
    pub lower: &'a [Weights],
}

/// `ILPPartitionInst` (no fixed vertices). Each vertex gets the first block whose variable is
/// EXACTLY 1.0 in the solution; -1 when none is.
#[cfg(feature = "ilp")]
pub fn ilp_partition_inst(inst: &IlpInstance) -> Result<Vec<i64>, IlpOutcome> {
    use vyges_ortools_lib::{ResultStatus, Solver};
    let num_vertices = inst.vertex_weights.len();
    let num_hyperedges = inst.hyperedge_weights.len();
    if num_vertices == 0 || num_hyperedges == 0 || inst.num_parts <= 1 || inst.vertex_weight_dimension == 0 {
        return Err(IlpOutcome::NotSolved);
    }
    let fail = |_| IlpOutcome::NotSolved;
    let mut solver = Solver::create("SCIP").ok_or(IlpOutcome::Unavailable)?;
    let mut x = Vec::with_capacity(inst.num_parts);
    for _ in 0..inst.num_parts {
        x.push((0..num_vertices).map(|_| solver.make_int_var(0.0, 1.0)).collect::<Result<Vec<_>, _>>().map_err(fail)?);
    }
    let mut y = Vec::with_capacity(inst.num_parts);
    for _ in 0..inst.num_parts {
        y.push((0..num_hyperedges).map(|_| solver.make_int_var(0.0, 1.0)).collect::<Result<Vec<_>, _>>().map_err(fail)?);
    }
    let infinity = Solver::infinity();
    for i in 0..inst.num_parts {
        for j in 0..inst.vertex_weight_dimension {
            let upper = solver.make_row_constraint(0.0, inst.upper[i][j] as f64).map_err(fail)?;
            let lower = solver.make_row_constraint(inst.lower[i][j] as f64, infinity).map_err(fail)?;
            for v in 0..num_vertices {
                let vwt = inst.vertex_weights[v][j] as f64;
                solver.set_coefficient(upper, x[i][v], vwt).map_err(fail)?;
                solver.set_coefficient(lower, x[i][v], vwt).map_err(fail)?;
            }
        }
    }
    // (Fixed vertices would add their rows here.)
    for v in 0..num_vertices {
        let row = solver.make_row_constraint(1.0, 1.0).map_err(fail)?;
        for xi in &x {
            solver.set_coefficient(row, xi[v], 1.0).map_err(fail)?;
        }
    }
    for (e, hyperedge) in inst.hyperedges.iter().enumerate() {
        for &v in hyperedge {
            for i in 0..inst.num_parts {
                let row = solver.make_row_constraint(0.0, infinity).map_err(fail)?;
                solver.set_coefficient(row, x[i][v], 1.0).map_err(fail)?;
                solver.set_coefficient(row, y[i][e], -1.0).map_err(fail)?;
            }
        }
    }
    let step = cost_step(inst.hyperedge_weights, inst.num_parts, num_vertices);
    for (e, &w) in inst.hyperedge_weights.iter().enumerate() {
        for yi in &y {
            solver.objective_set_coefficient(yi[e], w as f64).map_err(fail)?;
        }
    }
    for v in 0..num_vertices {
        for (block, xb) in x.iter().enumerate().skip(1) {
            solver.objective_set_coefficient(xb[v], (step * block as f32) as f64).map_err(fail)?;
        }
    }
    solver.objective_set_maximization();
    if solver.solve() != ResultStatus::Optimal {
        return Err(IlpOutcome::NotSolved);
    }
    Ok((0..num_vertices)
        .map(|v| (0..inst.num_parts).find(|&i| solver.solution_value(x[i][v]) == 1.0).map_or(-1, |i| i as i64))
        .collect())
}

/// Without the solver every exact solve is unavailable.
#[cfg(not(feature = "ilp"))]
pub fn ilp_partition_inst(_inst: &IlpInstance) -> Result<Vec<i64>, IlpOutcome> {
    Err(IlpOutcome::Unavailable)
}

/// The tie-break step: the smallest hyperedge weight (at most 1), over the blocks, over
/// n(n+1)/2 — float divided by an int, then by a double, stored back in a float.
pub fn cost_step(hyperedge_weights: &[f32], num_parts: usize, num_vertices: usize) -> f32 {
    let mut step = 1.0f32;
    for &w in hyperedge_weights {
        step = step.min(w);
    }
    let step = step / num_parts as f32;
    (step as f64 / (num_vertices as f64 * (num_vertices as f64 + 1.0) / 2.0)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// min(1, weights) / parts / (n(n+1)/2): 0.5 / 2 / 6 for weights {3, 0.5}, 2 blocks, 3 vertices.
    #[test]
    fn the_tie_break_step_is_tiny() {
        assert_eq!(cost_step(&[3.0, 0.5], 2, 3), (0.25f64 / 6.0) as f32);
        assert_eq!(cost_step(&[3.0], 2, 3), (0.5f64 / 6.0) as f32);
    }
}
