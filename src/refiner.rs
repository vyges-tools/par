// SPDX-License-Identifier: Apache-2.0
//! Move-based refinement without timing: the shared vertex-move machinery and the k-way FM
//! refiner (one gain bucket per destination block).
//!
//! The reference fills and updates the buckets one thread per bucket; each thread touches only
//! its own bucket and reads state no thread writes, so doing them in block order is the same.

use std::collections::BTreeSet;

use crate::evaluator::Evaluator;
use crate::hypergraph::{add, all_less, lex_le, sub, Hypergraph, Weights};
use crate::priority_queue::{PriorityQueue, VertexGain};

/// The state a pass works on and leaves behind.
pub struct State<'a> {
    pub solution: &'a mut [usize],
    pub balance: Vec<Weights>,
    pub net_degs: Vec<Vec<i32>>,
    pub visited: Vec<bool>,
}

/// `CalculateVertexGain`: what moving `v` from `from` to `to` saves in cut cost. A hyperedge
/// wholly in `from` (with another vertex there) would become cut; one with `v` the only vertex
/// outside `to`, spanning exactly two blocks, would become uncut.
pub fn vertex_gain(ev: &Evaluator, g: &Hypergraph, v: usize, from: usize, to: usize, net_degs: &[Vec<i32>]) -> VertexGain {
    let cell = |gain: f32| VertexGain { vertex: v as i64, source_part: from as i64, destination_part: to as i64, gain };
    if from == to {
        return cell(0.0);
    }
    let mut cut_score = 0.0f32;
    for &e in g.edges(v) {
        let connectivity = net_degs[e].iter().filter(|&&n| n > 0).count();
        let e_score = ev.hyperedge_cost(e, g);
        if connectivity == 0 {
            continue;
        }
        if connectivity == 1 && net_degs[e][from] > 1 {
            cut_score -= e_score;
        } else if connectivity == 2 && net_degs[e][from] == 1 && net_degs[e][to] > 0 {
            cut_score += e_score;
        }
    }
    // + the timing-path score, 0 without timing.
    cell(cut_score + 0.0)
}

/// `AcceptVertexGain`.
fn accept(g: &Hypergraph, cell: &VertexGain, total_delta_gain: &mut f32, st: &mut State) {
    let v = cell.vertex as usize;
    st.visited[v] = true;
    *total_delta_gain += cell.gain;
    let (from, to) = (cell.source_part as usize, cell.destination_part as usize);
    st.solution[v] = to;
    st.balance[from] = sub(&st.balance[from], g.vertex_weights(v));
    st.balance[to] = add(&st.balance[to], g.vertex_weights(v));
    for &e in g.edges(v) {
        st.net_degs[e][from] -= 1;
        st.net_degs[e][to] += 1;
    }
}

/// `RollBackVertexGain`.
fn roll_back(g: &Hypergraph, cell: &VertexGain, st: &mut State) {
    let v = cell.vertex as usize;
    st.visited[v] = false;
    let (from, to) = (cell.source_part as usize, cell.destination_part as usize);
    st.solution[v] = from;
    st.balance[from] = add(&st.balance[from], g.vertex_weights(v));
    st.balance[to] = sub(&st.balance[to], g.vertex_weights(v));
    for &e in g.edges(v) {
        st.net_degs[e][from] += 1;
        st.net_degs[e][to] -= 1;
    }
}

/// `CheckVertexMoveLegality`: both blocks stay within their bounds (lexicographic `<=`).
fn move_is_legal(g: &Hypergraph, v: usize, to: usize, from: usize, balance: &[Weights], upper: &[Weights], lower: &[Weights]) -> bool {
    lex_le(&add(&balance[to], g.vertex_weights(v)), &upper[to]) && lex_le(&lower[from], &sub(&balance[from], g.vertex_weights(v)))
}

/// `FindBoundaryVertices`: unvisited vertices on a hyperedge spanning two or more blocks.
fn boundary_vertices(g: &Hypergraph, net_degs: &[Vec<i32>], visited: &[bool]) -> Vec<usize> {
    let boundary_net: Vec<bool> = net_degs.iter().map(|d| d.iter().filter(|&&n| n > 0).count() >= 2).collect();
    (0..g.num_vertices()).filter(|&v| !visited[v] && g.edges(v).iter().any(|&e| boundary_net[e])).collect()
}

/// `FindNeighbors`: the unvisited vertices sharing a hyperedge with `v` (itself included when
/// unvisited), ascending — the reference collects them in a `std::set`.
fn neighbors(g: &Hypergraph, v: usize, visited: &[bool]) -> Vec<usize> {
    let mut set = BTreeSet::new();
    for &e in g.edges(v) {
        for &u in g.vertices(e) {
            if !visited[u] {
                set.insert(u);
            }
        }
    }
    set.into_iter().collect()
}

/// The k-way FM refiner. `max_move` and `refiner_iters` can be overridden and restored.
pub struct KWayFm {
    pub num_parts: usize,
    pub refiner_iters: usize,
    pub max_move: usize,
    pub total_corking_passes: usize,
    refiner_iters_default: usize,
    max_move_default: usize,
    pub evaluator: Evaluator,
}

impl KWayFm {
    pub fn new(num_parts: usize, refiner_iters: usize, max_move: usize, total_corking_passes: usize, evaluator: Evaluator) -> Self {
        Self {
            num_parts,
            refiner_iters,
            max_move,
            total_corking_passes,
            refiner_iters_default: refiner_iters,
            max_move_default: max_move,
            evaluator,
        }
    }

    pub fn restore_default_parameters(&mut self) {
        self.max_move = self.max_move_default;
        self.refiner_iters = self.refiner_iters_default;
    }

    /// `Refine`: up to `refiner_iters` passes, stopping at the first that gains nothing.
    pub fn refine(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], solution: &mut [usize]) {
        if self.max_move == 0 {
            return;
        }
        let balance = self.evaluator.block_balance(g, solution);
        let net_degs = self.evaluator.net_degrees(g, solution);
        let mut st = State { solution, balance, net_degs, visited: Vec::new() };
        for _ in 0..self.refiner_iters {
            st.visited = (0..g.num_vertices()).map(|v| g.has_fixed_vertices() && g.fixed_attr(v) > -1).collect();
            if self.pass(g, upper, lower, &mut st) <= 0.0 {
                return;
            }
        }
    }

    /// `Pass`: up to `max_move` best legal moves, each locking its vertex, then roll back to the
    /// LAST move at which the running gain was at its best (`>=`). An illegal starting balance
    /// makes any prefix acceptable.
    fn pass(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], st: &mut State) -> f32 {
        let mut buckets: Vec<PriorityQueue> =
            (0..self.num_parts).map(|_| PriorityQueue::new(g.num_vertices(), self.total_corking_passes)).collect();
        let boundary = boundary_vertices(g, &st.net_degs, &st.visited);
        if boundary.is_empty() {
            return 0.0;
        }
        self.initialize_gain_buckets(&mut buckets, g, &boundary, st);
        let mut moves_trace: Vec<VertexGain> = Vec::new();
        let mut total_delta_gain = 0.0f32;
        let mut best_gain = 0.0f32;
        for b in 0..self.num_parts {
            if all_less(&upper[b], &st.balance[b]) || all_less(&st.balance[b], &lower[b]) {
                best_gain = -f32::MAX;
                break;
            }
        }
        let mut best_vertex: i64 = -1;
        for _ in 0..self.max_move {
            let candidate = self.pick_move(&buckets, g, &st.balance, upper, lower);
            if candidate.vertex < 0 {
                break;
            }
            let vertex = candidate.vertex as usize;
            moves_trace.push(candidate);
            accept(g, &candidate, &mut total_delta_gain, st);
            for bucket in &mut buckets {
                bucket.remove(g, vertex);
            }
            let around = neighbors(g, vertex, &st.visited);
            for (to, bucket) in buckets.iter_mut().enumerate() {
                self.update_single_gain_bucket(to, bucket, g, &around, st);
            }
            if total_delta_gain >= best_gain {
                best_gain = total_delta_gain;
                best_vertex = candidate.vertex;
            }
        }
        for cell in moves_trace.iter().rev() {
            if cell.vertex == best_vertex {
                break;
            }
            roll_back(g, cell, st);
        }
        best_gain
    }

    /// `InitializeGainBucketsKWay`: bucket `to` holds every boundary vertex not already in `to`;
    /// an empty one is left inactive.
    fn initialize_gain_buckets(&self, buckets: &mut [PriorityQueue], g: &Hypergraph, boundary: &[usize], st: &State) {
        for (to, bucket) in buckets.iter_mut().enumerate() {
            bucket.set_active();
            for &v in boundary {
                let from = st.solution[v];
                if from == to {
                    continue;
                }
                bucket.insert(g, vertex_gain(&self.evaluator, g, v, from, to, &st.net_degs));
            }
            if bucket.is_empty() {
                bucket.set_deactive();
            }
        }
    }

    /// `PickMoveKWay`: of the active buckets' tops, the first with the strictly highest gain that
    /// is a legal move; failing that, the bucket with the highest top is searched for a move that
    /// fits strictly inside the bounds.
    fn pick_move(&self, buckets: &[PriorityQueue], g: &Hypergraph, balance: &[Weights], upper: &[Weights], lower: &[Weights]) -> VertexGain {
        let mut to_pid: i64 = -1;
        let mut candidate = VertexGain::default();
        let mut best_to: i64 = -1;
        let mut best_gain = -f32::MAX;
        for (i, bucket) in buckets.iter().enumerate() {
            if !bucket.status() {
                continue;
            }
            let ele = bucket.get_max();
            if ele.gain > candidate.gain
                && move_is_legal(g, ele.vertex as usize, i, ele.source_part as usize, balance, upper, lower)
            {
                to_pid = i as i64;
                candidate = ele;
            }
            if ele.gain > best_gain {
                best_gain = ele.gain;
                best_to = i as i64;
            }
        }
        if to_pid > -1 || best_to == -1 {
            return candidate;
        }
        buckets[best_to as usize].best_candidate(g, balance, upper, lower)
    }

    /// `UpdateSingleGainBucket`: the neighbours' gains toward `part`, re-keyed or inserted.
    fn update_single_gain_bucket(&self, part: usize, bucket: &mut PriorityQueue, g: &Hypergraph, around: &[usize], st: &State) {
        for &v in around {
            let from = st.solution[v];
            if from == part {
                continue;
            }
            let cell = vertex_gain(&self.evaluator, g, v, from, part, &st.net_degs);
            if bucket.contains(v) {
                bucket.change_priority(g, v, cell);
            } else {
                bucket.insert(g, cell);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev() -> Evaluator {
        Evaluator { num_parts: 2, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] }
    }

    /// Moving the only outside vertex of a two-block hyperedge uncuts it; moving a vertex out of a
    /// hyperedge wholly in its block cuts it.
    #[test]
    fn gain_counts_uncut_minus_newly_cut() {
        // e0 = {0,1} (1 in block 1), e1 = {0,2} (both block 0), e2 = {0} alone.
        let g = Hypergraph::new(1, 1, &[vec![0, 1], vec![0, 2], vec![0]], vec![vec![1.0]; 3], vec![vec![4.0], vec![1.0], vec![9.0]], vec![], vec![]);
        let sol = [0, 1, 0];
        let nd = ev().net_degrees(&g, &sol);
        assert_eq!(vertex_gain(&ev(), &g, 0, 0, 1, &nd).gain, 3.0);
        assert_eq!(vertex_gain(&ev(), &g, 0, 0, 0, &nd).gain, 0.0);
    }

    /// Neighbours are ascending and include the moved vertex only if still unvisited.
    #[test]
    fn neighbours_are_a_sorted_set() {
        let g = Hypergraph::new(1, 1, &[vec![3, 1], vec![1, 0, 3]], vec![vec![1.0]; 4], vec![vec![1.0]; 2], vec![], vec![]);
        assert_eq!(neighbors(&g, 1, &[false, true, false, false]), vec![0, 3]);
    }

    fn graph(weights: &[f32], edges: &[Vec<usize>], edge_weights: &[f32], fixed: Vec<i32>) -> Hypergraph {
        Hypergraph::new(1, 1, edges, weights.iter().map(|&w| vec![w]).collect(), edge_weights.iter().map(|&w| vec![w]).collect(), fixed, vec![])
    }

    /// Equal top gains in two buckets: the FIRST bucket's move is taken (strictly greater wins).
    #[test]
    fn a_tie_between_buckets_goes_to_the_first() {
        let g = graph(&[1.0; 4], &[vec![0, 1]], &[1.0], vec![]);
        let mut sol = vec![0, 1, 0, 1];
        KWayFm::new(2, 2, 50, 25, ev()).refine(&g, &vec![vec![3.0]; 2], &vec![vec![1.0]; 2], &mut sol);
        assert_eq!(sol, vec![0, 0, 0, 1]);
    }

    /// An illegal start accepts the best prefix even when every prefix LOSES cut cost.
    #[test]
    fn an_illegal_start_keeps_a_losing_move() {
        let g = graph(&[2.0, 1.0, 1.0], &[vec![0, 1], vec![1, 2]], &[5.0, 1.0], vec![]);
        let mut sol = vec![0, 0, 1];
        KWayFm::new(2, 2, 50, 25, ev()).refine(&g, &vec![vec![2.2]; 2], &vec![vec![1.8]; 2], &mut sol);
        assert_eq!(sol, vec![0, 1, 1]);
    }

    /// A bucket emptied by a move stays INACTIVE when neighbours are later inserted into it, so
    /// its moves are not offered for the rest of the pass.
    #[test]
    fn an_emptied_bucket_stays_inactive() {
        // v0 fixed in block 0. v2 -> 1 (gain 5 - 1 = 4) beats v1 -> 0 (1 + 5 - 3 = 3) and empties
        // bucket 1; v3 -> 1 (gain 1) then lands in it unseen, so only v1 -> 0 (now -7) is offered,
        // which loses and is rolled back.
        let g = graph(&[1.0; 5], &[vec![0, 1], vec![1, 2], vec![2, 3], vec![1, 4]], &[1.0, 5.0, 1.0, 3.0], vec![0, -1, -1, -1, -1]);
        let mut sol = vec![0, 1, 0, 0, 1];
        KWayFm::new(2, 1, 50, 25, ev()).refine(&g, &vec![vec![10.0]; 2], &vec![vec![0.0]; 2], &mut sol);
        assert_eq!(sol, vec![0, 1, 1, 0, 1]);
    }

    /// A pass that gains exactly 0 ends the refinement (`gain <= 0`), even with passes left. Here
    /// the zero-gain move v1 -> 1 is kept.
    #[test]
    fn a_zero_gain_pass_stops_refinement() {
        let g = graph(&[1.0; 3], &[vec![0, 1], vec![1, 2]], &[1.0, 1.0], vec![]);
        let mut sol = vec![0, 0, 1];
        // Two passes (the default); a second would move v1 back. (An odd count would hide it: the
        // passes alternate between the two solutions.)
        KWayFm::new(2, 2, 50, 25, ev()).refine(&g, &vec![vec![2.0]; 2], &vec![vec![1.0]; 2], &mut sol);
        assert_eq!(sol, vec![0, 1, 1]);
    }

    /// A pass on a cut two-cluster graph moves the stray vertex home and keeps the move.
    #[test]
    fn refine_fixes_a_stray_vertex() {
        // {0,1,2} and {3,4,5} triangles joined by one edge; vertex 2 starts in the wrong block.
        let edges = vec![vec![0, 1], vec![1, 2], vec![0, 2], vec![3, 4], vec![4, 5], vec![3, 5], vec![2, 3]];
        let g = Hypergraph::new(1, 1, &edges, vec![vec![1.0]; 6], vec![vec![1.0]; 7], vec![], vec![]);
        let upper = vec![vec![4.0]; 2];
        let lower = vec![vec![2.0]; 2];
        let mut sol = vec![0, 0, 1, 1, 1, 1];
        let fm = KWayFm::new(2, 2, 50, 25, ev());
        fm.refine(&g, &upper, &lower, &mut sol);
        assert_eq!(sol, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(ev().cut_evaluator(&g, &sol).0, 1.0);
    }
}
