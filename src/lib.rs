// SPDX-License-Identifier: Apache-2.0
//! Hypergraph partitioning: multilevel coarsening, initial partitioning and refinement.
//!
//! Built in stages, each checked against a reference before the next is written; this crate holds
//! the stages done so far. `Params::simple` is the parameter set of a k-way partition with no
//! timing, placement, fixed vertices or groups.

pub mod coarsener;
pub mod evaluator;
pub mod hypergraph;
pub mod initial;
pub mod mt19937;
pub mod multilevel;
pub mod partitioner;
pub mod priority_queue;
pub mod refinement;
pub mod refiner;
pub mod shuffle;

use coarsener::{CoarsenOrder, Coarsener};
use evaluator::Evaluator;
use hypergraph::{divide_factor, Hypergraph, Weights};
use initial::InitialParams;
use refinement::Refiners;
use refiner::{Greedy, KWayFm, KWayPm};

/// The partitioner's parameters.
#[derive(Clone, Debug)]
pub struct Params {
    pub num_parts: usize,
    /// Allowed imbalance, in percent of the total weight per block.
    pub ub_factor: f32,
    pub seed: i32,
    /// Hyperedges larger than this are dropped when the graph is first grouped.
    pub global_net_threshold: usize,
    pub thr_coarsen_hyperedge_size_skip: usize,
    pub thr_coarsen_vertices: usize,
    pub thr_coarsen_hyperedges: usize,
    pub coarsening_ratio: f32,
    pub max_coarsen_iters: usize,
    pub adj_diff_ratio: f32,
    pub min_num_vertices_each_part: usize,
    pub coarsen_order: CoarsenOrder,
    pub num_coarsen_solutions: usize,
    pub num_initial_solutions: usize,
    pub num_best_initial_solutions: usize,
    pub num_vertices_threshold_ilp: usize,
    pub refiner_iters: usize,
    pub max_moves: usize,
    pub total_corking_passes: usize,
    pub v_cycle: bool,
    pub max_num_vcycle: usize,
}

impl Params {
    /// The defaults, for `num_parts` blocks with `ub_factor` % imbalance and `seed`.
    pub fn simple(num_parts: usize, ub_factor: f32, seed: i32) -> Self {
        Self {
            num_parts,
            ub_factor,
            seed,
            global_net_threshold: 1000,
            thr_coarsen_hyperedge_size_skip: 50,
            thr_coarsen_vertices: 200,
            thr_coarsen_hyperedges: 50,
            coarsening_ratio: 1.5,
            max_coarsen_iters: 20,
            adj_diff_ratio: 0.0001,
            min_num_vertices_each_part: 4,
            coarsen_order: CoarsenOrder::Random,
            num_coarsen_solutions: 3,
            num_initial_solutions: 50,
            num_best_initial_solutions: 10,
            num_vertices_threshold_ilp: 50,
            refiner_iters: 2,
            max_moves: 50,
            total_corking_passes: 25,
            v_cycle: true,
            max_num_vcycle: 5,
        }
    }

    /// One-dimensional weights with unit factors.
    pub fn evaluator(&self) -> Evaluator {
        Evaluator { num_parts: self.num_parts, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] }
    }

    /// The block bounds, from the ORIGINAL graph: equal shares (`1.0 / parts`, a double stored
    /// in a float, divided by a unit scale factor) plus and minus the imbalance.
    pub fn block_bounds(&self, original: &Hypergraph) -> (Vec<Weights>, Vec<Weights>) {
        let base: Vec<f32> = (0..self.num_parts).map(|_| (1.0f64 / self.num_parts as f64) as f32 / 1.0f32).collect();
        (
            original.upper_vertex_balance(self.num_parts, self.ub_factor, &base),
            original.lower_vertex_balance(self.num_parts, self.ub_factor, &base),
        )
    }

    /// The k-way FM refiner at its defaults.
    pub fn k_way_fm(&self) -> KWayFm {
        KWayFm::new(self.num_parts, self.refiner_iters, self.max_moves, self.total_corking_passes, self.evaluator())
    }

    /// The refiners `RefinePartition` calls, at their defaults (the FM one is `fm`, shared with
    /// initial partitioning).
    pub fn refiners(&self, fm: KWayFm) -> Refiners {
        Refiners {
            pm: KWayPm { fm: self.k_way_fm() },
            fm,
            greedy: Greedy {
                num_parts: self.num_parts,
                refiner_iters: self.refiner_iters,
                max_move: self.max_moves,
                evaluator: self.evaluator(),
            },
        }
    }

    pub fn initial_params(&self) -> InitialParams {
        InitialParams {
            num_parts: self.num_parts,
            seed: self.seed,
            num_initial_random_solutions: self.num_initial_solutions,
            num_best_initial_solutions: self.num_best_initial_solutions,
            num_vertices_threshold_ilp: self.num_vertices_threshold_ilp,
        }
    }

    /// The coarsener for `g`: a cluster may weigh at most total / (min vertices per part × parts).
    pub fn coarsener(&self, g: &Hypergraph) -> Coarsener {
        Coarsener {
            num_parts: self.num_parts,
            thr_coarsen_hyperedge_size_skip: self.thr_coarsen_hyperedge_size_skip,
            thr_coarsen_vertices: self.thr_coarsen_vertices,
            thr_coarsen_hyperedges: self.thr_coarsen_hyperedges,
            coarsening_ratio: self.coarsening_ratio,
            max_coarsen_iters: self.max_coarsen_iters,
            adj_diff_ratio: self.adj_diff_ratio,
            thr_cluster_weight: divide_factor(
                &g.total_vertex_weights(),
                (self.min_num_vertices_each_part * self.num_parts) as f32,
            ),
            random_seed: self.seed,
            vertex_order_choice: self.coarsen_order,
            evaluator: self.evaluator(),
        }
    }
}

/// The input graph of a one-dimensional partition: hyperedges as vertex lists, one weight each.
pub fn simple_hypergraph(hyperedges: &[Vec<usize>], vertex_weights: &[f32], hyperedge_weights: &[f32]) -> Hypergraph {
    Hypergraph::new(
        1,
        1,
        hyperedges,
        vertex_weights.iter().map(|&w| vec![w]).collect(),
        hyperedge_weights.iter().map(|&w| vec![w]).collect(),
        Vec::new(),
        Vec::new(),
    )
}
