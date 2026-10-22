// SPDX-License-Identifier: Apache-2.0
//! The multilevel partitioner's call sequence, one thin function per reference entry point, each
//! writing the reference's trace lines (`[group] text`, timing lines left out) in its order.
//!
//! Stages that need an exact (ILP) solve are refused: the run stops there with `Refused`, and the
//! trace ends with the lines printed up to that point.

use crate::coarsener::Coarsener;
use crate::evaluator::Evaluator;
use crate::hypergraph::{Hypergraph, Weights};
use crate::initial::{initial_partition, Refused};
use crate::partitioner::ilp_part;
use crate::refinement::{refine_partition, Refiners};
use crate::Params;

pub struct Multilevel {
    pub p: Params,
    pub ev: Evaluator,
    pub co: Coarsener,
    pub refiners: Refiners,
    pub upper: Vec<Weights>,
    pub lower: Vec<Weights>,
    /// The partitioner's seed: the call's, until a random start sets it.
    pub partitioner_seed: i32,
    pub trace: Vec<String>,
}

/// `TritonPart::MultiLevelPartition` for a simple k-way call: group the original graph, partition
/// it, carry the solution back to the original vertices, then a last V-cycle on the original
/// graph — with the coarsener's hyperedge-size limit left at the GLOBAL threshold.
pub fn multilevel_partition(p: &Params, original: &Hypergraph) -> (Result<Vec<usize>, Refused>, Vec<String>) {
    let mut ml = Multilevel::new(p, original);
    let result = ml.run(original);
    (result, ml.trace)
}

/// `Partition`'s choice among the candidates: the LAST of the cheapest (`cost <= best`).
fn last_of_cheapest(costs: &[f32]) -> usize {
    let mut best_cost = f32::MAX;
    let mut best_id = 0;
    for (id, &cost) in costs.iter().enumerate() {
        if cost <= best_cost {
            best_cost = cost;
            best_id = id;
        }
    }
    best_id
}

impl Multilevel {
    /// The partitioner for `original` (bounds and cluster-weight limit from it), its trace
    /// opened with the reference's start-up lines.
    pub fn new(p: &Params, original: &Hypergraph) -> Self {
        let (upper, lower) = p.block_bounds(original);
        Multilevel {
            p: p.clone(),
            ev: p.evaluator(),
            co: p.coarsener(original),
            refiners: p.refiners(p.k_way_fm()),
            upper,
            lower,
            partitioner_seed: p.seed,
            trace: vec![
                "[multilevel_partitioning] Starting multilevel partitioning.".into(),
                "[multilevel_partitioning] no base balance is specified. Use default value.".into(),
                "[multilevel_partitioning] no scale factor is specified. Use default value.".into(),
                "[multilevel_partitioning] No hyperedge weighting is specified. Use default value of 1.".into(),
                "[multilevel_partitioning] No vertex weighting is specified. Use default value of 1.".into(),
            ],
        }
    }

    fn run(&mut self, original: &Hypergraph) -> Result<Vec<usize>, Refused> {
        self.co.thr_coarsen_hyperedge_size_skip = self.p.global_net_threshold;
        let mut grouped = self.co.group_vertices(original);
        self.co.thr_coarsen_hyperedge_size_skip = self.p.thr_coarsen_hyperedge_size_skip;
        let solution = self.partition(&mut grouped)?;
        let mut original_solution = vec![0usize; original.num_vertices()];
        for (cluster, &part) in solution.iter().enumerate() {
            for &v in grouped.vertex_c_attr(cluster) {
                original_solution[v] = part;
            }
        }
        self.last_minute_refinement(original, &mut original_solution)?;
        Ok(original_solution)
    }

    /// The "last-minute refinement": a V-cycle on the ORIGINAL graph, with the coarsener's
    /// hyperedge-size limit set to the GLOBAL threshold (and left there).
    pub fn last_minute_refinement(&mut self, original: &Hypergraph, solution: &mut Vec<usize>) -> Result<(), Refused> {
        self.co.thr_coarsen_hyperedge_size_skip = self.p.global_net_threshold;
        let mut g = original.clone();
        self.vcycle_refinement(&mut g, solution)
    }

    /// `Partition`: one candidate per coarsening seed (the LAST of the cheapest is best), the
    /// cut-overlay partition of all candidates, then the V-cycles.
    fn partition(&mut self, g: &mut Hypergraph) -> Result<Vec<usize>, Refused> {
        let mut top = Vec::new();
        let mut costs = Vec::new();
        for _ in 0..self.p.num_coarsen_solutions {
            self.co.increase_random_seed();
            top.push(self.single_level_partition(g)?);
            costs.push(self.ev.cut_evaluator(g, top.last().expect("just pushed")).0);
        }
        let best_id = last_of_cheapest(&costs);
        self.trace.push("[multilevel_partitioning] Finish Candidate Solutions Generation".into());
        let mut best = self.cut_overlay_ilp_part(g, &top, best_id)?;
        self.trace.push("[multilevel_partitioning] Finish Cut-Overlay Clustering and Optimal Partitioning".into());
        if self.p.v_cycle {
            self.vcycle_refinement(g, &mut best)?;
        }
        self.trace.push("[multilevel_partitioning] Finish Vcycle Refinement".into());
        Ok(best)
    }

    /// `SingleLevelPartition`: coarsen, partition the coarsest graph, refine back up, then the
    /// cut-overlay partition of the kept solutions.
    fn single_level_partition(&mut self, g: &Hypergraph) -> Result<Vec<usize>, Refused> {
        let levels = self.lazy_first_choice(g);
        let hierarchy: Vec<&Hypergraph> = std::iter::once(g).chain(&levels).collect();
        let coarsest = *hierarchy.last().expect("level 0");
        let ip = initial_partition(
            &self.p.initial_params(),
            &self.ev,
            &mut self.refiners.fm,
            coarsest,
            &self.upper,
            &self.lower,
            &mut self.partitioner_seed,
            &mut self.trace,
        )?;
        let mut top = ip.top;
        let mut best_id = 0;
        self.refine_partition(&hierarchy, &mut top, &mut best_id);
        self.cut_overlay_ilp_part(g, &top, best_id)
    }

    /// `VcycleRefinement`: each cycle re-coarsens `g` within the current solution's blocks (set
    /// as `g`'s communities — a change `g` keeps) and refines; then the cut-overlay partition of
    /// the starting solution and every cycle's, the last one as the hint.
    fn vcycle_refinement(&mut self, g: &mut Hypergraph, best: &mut Vec<usize>) -> Result<(), Refused> {
        let mut candidates = vec![best.clone()];
        for num_cycles in 0..self.p.max_num_vcycle {
            g.set_community(best.iter().map(|&b| b as i32).collect());
            *best = self.single_cycle_refinement(g)?;
            candidates.push(best.clone());
            let cost = self.ev.cut_evaluator(g, best).0;
            self.trace.push(format!("[v_cycle_refinement] num_cycles = {num_cycles}, cutcost = {cost}"));
        }
        *best = self.cut_overlay_ilp_part(g, &candidates, candidates.len() - 1)?;
        Ok(())
    }

    /// `SingleCycleRefinement`: coarsen (within communities), take the coarsest graph's
    /// communities as its solution — replaced by the exact partition when that graph is small —
    /// and refine back up.
    fn single_cycle_refinement(&mut self, g: &Hypergraph) -> Result<Vec<usize>, Refused> {
        let levels = self.lazy_first_choice(g);
        let hierarchy: Vec<&Hypergraph> = std::iter::once(g).chain(&levels).collect();
        let coarsest = *hierarchy.last().expect("level 0");
        let mut top = vec![coarsest.copy_community().iter().map(|&c| c as usize).collect::<Vec<usize>>()];
        if coarsest.num_vertices() <= self.p.num_vertices_threshold_ilp {
            self.ilp(coarsest, &mut top[0])?;
        }
        let mut best_id = 0;
        self.refine_partition(&hierarchy, &mut top, &mut best_id);
        Ok(top.swap_remove(best_id))
    }

    /// `CutOverlayILPPart`: hyperedges cut by ANY of `top` are removed; the connected components
    /// of what is left (breadth-first, components numbered in order of their lowest vertex)
    /// become the vertices of a grouped graph, starting in the hint solution's block of that
    /// vertex. That graph is partitioned exactly when small, else by one refinement cycle; the
    /// result is carried back onto the hint.
    fn cut_overlay_ilp_part(&mut self, g: &Hypergraph, top: &[Vec<usize>], best_id: usize) -> Result<Vec<usize>, Refused> {
        let mut optimal = top[best_id].clone();
        let mut mask = vec![false; g.num_hyperedges()];
        for solution in top {
            for (e, m) in mask.iter_mut().enumerate() {
                if *m {
                    continue;
                }
                if let Some((&first, rest)) = g.vertices(e).split_first() {
                    *m = rest.iter().any(|&v| solution[v] != solution[first]);
                }
            }
        }
        let mut cluster_of = vec![-1i64; g.num_vertices()];
        let mut init_solution = Vec::new();
        let mut cluster_id = -1i64;
        for v in 0..g.num_vertices() {
            if cluster_of[v] != -1 {
                continue;
            }
            cluster_id += 1;
            cluster_of[v] = cluster_id;
            init_solution.push(top[best_id][v]);
            let mut wavefront = std::collections::VecDeque::from([v]);
            while let Some(u) = wavefront.pop_front() {
                for &e in g.edges(u) {
                    if mask[e] {
                        continue;
                    }
                    for &nbr in g.vertices(e) {
                        if cluster_of[nbr] == -1 {
                            cluster_of[nbr] = cluster_id;
                            wavefront.push_back(nbr);
                        }
                    }
                }
            }
        }
        let num_clusters = (cluster_id + 1) as usize;
        let mut groups = vec![Vec::new(); num_clusters];
        for v in 0..g.num_vertices() {
            groups[cluster_of[v] as usize].push(v);
        }
        let mut clustered = self.co.group_vertices_with(g, &groups);
        self.trace.push(format!(
            "[cut_overlay_clustering] num_vertices = {}, num_hyperedges = {}",
            clustered.num_vertices(),
            clustered.num_hyperedges()
        ));
        let solution = if num_clusters <= self.p.num_vertices_threshold_ilp {
            let mut s = init_solution;
            self.ilp(&clustered, &mut s)?;
            s
        } else {
            clustered.set_community(init_solution.iter().map(|&b| b as i32).collect());
            self.single_cycle_refinement(&clustered)?
        };
        for (c, &part) in solution.iter().enumerate().take(clustered.num_vertices()) {
            for &v in clustered.vertex_c_attr(c) {
                optimal[v] = part;
            }
        }
        self.trace.push("[cut_overlay_clustering] Statistics of cut-overlay solution:".into());
        Ok(optimal)
    }

    /// `LazyFirstChoice` with its trace: the opening line, then every level including `g` itself.
    fn lazy_first_choice(&mut self, g: &Hypergraph) -> Vec<Hypergraph> {
        self.trace.push("[coarsening] Running FC Multilevel Coarsening...".into());
        let levels = self.co.lazy_first_choice(g);
        for (l, h) in std::iter::once(g).chain(&levels).enumerate() {
            self.trace.push(format!("[coarsening] Level {l} :: num_vertices = {}, num_hyperedges = {}", h.num_vertices(), h.num_hyperedges()));
        }
        levels
    }

    fn ilp(&mut self, g: &Hypergraph, solution: &mut Vec<usize>) -> Result<(), Refused> {
        ilp_part(g, self.p.num_parts, &self.ev, &self.upper, &self.lower, self.partitioner_seed, solution, &mut self.trace)
    }

    fn refine_partition(&mut self, hierarchy: &[&Hypergraph], top: &mut [Vec<usize>], best_id: &mut usize) {
        refine_partition(hierarchy, self.p.num_parts, &self.ev, &self.refiners, &self.upper, &self.lower, top, best_id, &mut self.trace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple_hypergraph;

    /// Equal costs: the LATER candidate wins.
    #[test]
    fn the_last_of_the_cheapest_candidates_is_best() {
        assert_eq!(last_of_cheapest(&[5.0, 3.0, 3.0]), 2);
        assert_eq!(last_of_cheapest(&[3.0, 4.0]), 0);
    }

    /// The last-minute V-cycle coarsens with the GLOBAL hyperedge-size limit: a 61-vertex
    /// hyperedge (over the coarsening limit of 50) still steers the first matching pass.
    #[test]
    fn the_last_minute_vcycle_keeps_large_hyperedges() {
        let mut edges: Vec<Vec<usize>> = (0..119).map(|v| vec![v, v + 1]).collect();
        edges.push((0..61).collect());
        let g = simple_hypergraph(&edges, &[1.0; 120], &[1.0; 120]);
        let p = Params::simple(2, 1.0, 0);
        let mut ml = Multilevel::new(&p, &g);
        let mut solution: Vec<usize> = (0..120).map(|v| usize::from(v >= 60)).collect();
        let _ = ml.last_minute_refinement(&g, &mut solution);
        let first_pass = ml.trace.iter().find(|l| l.starts_with("[coarsening] Level 1 ")).expect("one pass");
        // (This crate's value; with the 50 limit the first pass differs — the rule is the limit.)
        assert_eq!(first_pass, "[coarsening] Level 1 :: num_vertices = 80, num_hyperedges = 80");
    }
}
