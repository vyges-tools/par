// SPDX-License-Identifier: Apache-2.0
//! Move-based refinement without timing: the shared vertex-move machinery and the k-way FM
//! refiner (one gain bucket per destination block).
//!
//! The reference fills and updates the buckets one thread per bucket; each thread touches only
//! its own bucket and reads state no thread writes, so doing them in block order is the same.

use std::collections::{BTreeMap, BTreeSet};

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

/// `FindBoundaryVertices`: unvisited vertices on a boundary hyperedge — one spanning two or more
/// blocks, or with a `pair`, one touching BOTH blocks of the pair (its vertices may be in a third).
fn boundary_vertices(g: &Hypergraph, net_degs: &[Vec<i32>], visited: &[bool], pair: Option<(usize, usize)>) -> Vec<usize> {
    let boundary_net: Vec<bool> = net_degs
        .iter()
        .map(|d| match pair {
            None => d.iter().filter(|&&n| n > 0).count() >= 2,
            Some((a, b)) => d[a] > 0 && d[b] > 0,
        })
        .collect();
    (0..g.num_vertices()).filter(|&v| !visited[v] && g.edges(v).iter().any(|&e| boundary_net[e])).collect()
}

/// `FindNeighbors`: the unvisited vertices sharing a hyperedge with `v` (itself included when
/// unvisited) — with a `pair`, only those in its blocks — ascending: the reference collects them
/// in a `std::set`.
fn neighbors(g: &Hypergraph, v: usize, visited: &[bool], solution: &[usize], pair: Option<(usize, usize)>) -> Vec<usize> {
    let mut set = BTreeSet::new();
    for &e in g.edges(v) {
        for &u in g.vertices(e) {
            if !visited[u] && pair.is_none_or(|(a, b)| solution[u] == a || solution[u] == b) {
                set.insert(u);
            }
        }
    }
    set.into_iter().collect()
}

/// `Refine`, shared by every refiner: up to `iters` passes over fresh visit flags (fixed vertices
/// pre-visited), stopping at the first pass that gains nothing (`<= 0`).
fn refine_loop(
    ev: &Evaluator,
    max_move: usize,
    iters: usize,
    g: &Hypergraph,
    solution: &mut [usize],
    mut pass: impl FnMut(&mut State) -> f32,
) {
    if max_move == 0 {
        return;
    }
    let balance = ev.block_balance(g, solution);
    let net_degs = ev.net_degrees(g, solution);
    let mut st = State { solution, balance, net_degs, visited: Vec::new() };
    for _ in 0..iters {
        st.visited = (0..g.num_vertices()).map(|v| g.has_fixed_vertices() && g.fixed_attr(v) > -1).collect();
        if pass(&mut st) <= 0.0 {
            return;
        }
    }
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
        refine_loop(&self.evaluator, self.max_move, self.refiner_iters, g, solution, |st| self.pass(g, upper, lower, st));
    }

    /// `Pass`: up to `max_move` best legal moves, each locking its vertex, then roll back to the
    /// LAST move at which the running gain was at its best (`>=`). An illegal starting balance
    /// makes any prefix acceptable.
    fn pass(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], st: &mut State) -> f32 {
        let mut buckets: Vec<PriorityQueue> =
            (0..self.num_parts).map(|_| PriorityQueue::new(g.num_vertices(), self.total_corking_passes)).collect();
        let boundary = boundary_vertices(g, &st.net_degs, &st.visited, None);
        if boundary.is_empty() {
            return 0.0;
        }
        for (to, bucket) in buckets.iter_mut().enumerate() {
            self.initialize_single_gain_bucket(to, bucket, g, &boundary, st);
        }
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
            let around = neighbors(g, vertex, &st.visited, st.solution, None);
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

    /// `InitializeSingleGainBucket`: bucket `to` holds every boundary vertex not already in `to`;
    /// an empty one is left inactive.
    fn initialize_single_gain_bucket(&self, to: usize, bucket: &mut PriorityQueue, g: &Hypergraph, boundary: &[usize], st: &State) {
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

    /// `PerformPairFM`: the FM pass confined to one block pair — boundary and neighbours from the
    /// pair, only its two buckets filled and updated — with NO illegal-start rule: best gain
    /// starts at 0. The buckets are the pass's, cleared before and after.
    #[allow(clippy::too_many_arguments)]
    fn perform_pair_fm(
        &self,
        g: &Hypergraph,
        upper: &[Weights],
        lower: &[Weights],
        st: &mut State,
        buckets: &mut [PriorityQueue],
        pair: (usize, usize),
    ) -> f32 {
        let blocks = [pair.0, pair.1];
        for &b in &blocks {
            buckets[b].clear();
        }
        let boundary = boundary_vertices(g, &st.net_degs, &st.visited, Some(pair));
        for &to in &blocks {
            self.initialize_single_gain_bucket(to, &mut buckets[to], g, &boundary, st);
        }
        let mut moves_trace: Vec<VertexGain> = Vec::new();
        let mut total_delta_gain = 0.0f32;
        let mut best_gain = 0.0f32;
        let mut best_vertex: i64 = -1;
        for _ in 0..self.max_move {
            let candidate = self.pick_move(buckets, g, &st.balance, upper, lower);
            if candidate.vertex < 0 {
                break;
            }
            let vertex = candidate.vertex as usize;
            moves_trace.push(candidate);
            accept(g, &candidate, &mut total_delta_gain, st);
            for bucket in buckets.iter_mut() {
                bucket.remove(g, vertex);
            }
            let around = neighbors(g, vertex, &st.visited, st.solution, Some(pair));
            for &to in &blocks {
                self.update_single_gain_bucket(to, &mut buckets[to], g, &around, st);
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
        for &b in &blocks {
            buckets[b].clear();
        }
        best_gain
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

/// The k-way pairwise FM refiner: per pass, blocks are matched greedily by connectivity and
/// each matched pair is refined by a confined FM pass. Shares the FM refiner's machinery.
pub struct KWayPm {
    pub fm: KWayFm,
}

impl KWayPm {
    pub fn refine(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], solution: &mut [usize]) {
        let fm = &self.fm;
        refine_loop(&fm.evaluator, fm.max_move, fm.refiner_iters, g, solution, |st| self.pass(g, upper, lower, st));
    }

    /// `Pass`: the pairs of `maximum_match` in order, gains summed.
    fn pass(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], st: &mut State) -> f32 {
        let fm = &self.fm;
        let matches = maximum_match(fm.num_parts, &fm.evaluator.matching_connectivity(g, st.solution));
        let mut delta_gain = 0.0f32;
        let mut buckets: Vec<PriorityQueue> =
            (0..fm.num_parts).map(|_| PriorityQueue::new(g.num_vertices(), fm.total_corking_passes)).collect();
        for pair in matches {
            delta_gain += fm.perform_pair_fm(g, upper, lower, st, &mut buckets, pair);
        }
        delta_gain
    }
}

/// `CalculateMaximumMatch`: pairs by score descending, then pair ascending; a pair is taken when
/// neither block is matched yet, stopping once at least `num_parts - 1` blocks are.
fn maximum_match(num_parts: usize, scores: &BTreeMap<(usize, usize), f32>) -> Vec<(usize, usize)> {
    let mut sorted: Vec<((usize, usize), f32)> = scores.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_by(|a, b| {
        if a.1 != b.1 {
            b.1.partial_cmp(&a.1).expect("finite score")
        } else {
            a.0.cmp(&b.0)
        }
    });
    let mut matched = vec![false; num_parts];
    let mut num_matched = 0;
    let mut out = Vec::new();
    for ((a, b), _) in sorted {
        if !matched[a] && !matched[b] {
            out.push((a, b));
            matched[a] = true;
            matched[b] = true;
            num_matched += 2;
            if num_matched + 1 >= num_parts {
                break;
            }
        }
    }
    out
}

/// The greedy refiner: whole cut hyperedges moved into one block when that gains (or breaks even).
pub struct Greedy {
    pub num_parts: usize,
    pub refiner_iters: usize,
    pub max_move: usize,
    pub evaluator: Evaluator,
}

/// A whole-hyperedge move.
struct HyperedgeGain {
    hyperedge: usize,
    destination_part: usize,
    gain: f32,
}

impl Greedy {
    pub fn refine(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], solution: &mut [usize]) {
        refine_loop(&self.evaluator, self.max_move, self.refiner_iters, g, solution, |st| self.pass(g, upper, lower, st));
    }

    /// `Pass`: hyperedges in order; each one spanning two or more blocks counts as a move — the
    /// pass ENDS when the count reaches `max_move`, before that one is tried. Its best legal
    /// destination is taken if the gain is >= 0: highest gain, then the lighter hyperedge — but
    /// both candidates ARE this hyperedge, so that tie-break never fires and a tie keeps the
    /// first block.
    fn pass(&self, g: &Hypergraph, upper: &[Weights], lower: &[Weights], st: &mut State) -> f32 {
        let mut total_gain = 0.0f32;
        let mut num_move = 0;
        for e in 0..g.num_hyperedges() {
            if st.net_degs[e].iter().filter(|&&n| n > 0).count() <= 1 {
                continue;
            }
            num_move += 1;
            if num_move >= self.max_move {
                return total_gain;
            }
            let mut best: Option<HyperedgeGain> = None;
            for to in 0..self.num_parts {
                if !self.hyperedge_move_is_legal(g, e, to, st, upper, lower) {
                    continue;
                }
                let cand = self.hyperedge_gain(g, e, to, st);
                let better = match &best {
                    None => true,
                    Some(b) => {
                        cand.gain > b.gain
                            || (cand.gain == b.gain
                                && self.evaluator.hyperedge_vertex_wt_sum(cand.hyperedge, g)
                                    < self.evaluator.hyperedge_vertex_wt_sum(b.hyperedge, g))
                    }
                };
                if better {
                    best = Some(cand);
                }
            }
            if let Some(b) = best {
                if b.gain >= 0.0 {
                    self.accept_hyperedge(g, &b, &mut total_gain, st);
                }
            }
        }
        total_gain
    }

    /// `CalculateHyperedgeGain`: the vertices outside `to` move one by one (in the hyperedge's
    /// order), each scored as a vertex move against net degrees updated by the moves before it.
    fn hyperedge_gain(&self, g: &Hypergraph, e: usize, to: usize, st: &State) -> HyperedgeGain {
        let movers: Vec<(usize, usize)> =
            g.vertices(e).iter().filter(|&&v| st.solution[v] != to).map(|&v| (v, st.solution[v])).collect();
        let mut cut_score = 0.0f32;
        let mut degs: BTreeMap<usize, Vec<i32>> = BTreeMap::new();
        for &(v, _) in &movers {
            for &f in g.edges(v) {
                degs.entry(f).or_insert_with(|| st.net_degs[f].clone());
            }
        }
        for &(v, from) in &movers {
            for &f in g.edges(v) {
                let d = degs.get_mut(&f).expect("collected above");
                let connectivity = d.iter().filter(|&&n| n > 0).count();
                let f_score = self.evaluator.hyperedge_cost(f, g);
                if connectivity == 0 {
                    continue;
                }
                if connectivity == 1 && d[from] > 1 {
                    cut_score -= f_score;
                } else if connectivity == 2 && d[from] == 1 && d[to] > 0 {
                    cut_score += f_score;
                }
                d[from] -= 1;
                d[to] += 1;
            }
        }
        HyperedgeGain { hyperedge: e, destination_part: to, gain: cut_score + 0.0 }
    }

    /// `CheckHyperedgeMoveLegality`: after the move `to` must not exceed its upper bound and no
    /// other block may fall below its lower one (the partitioner's all-dimensions `<`). ⚠️ With
    /// fixed vertices, ANY vertex not fixed to `to` — a free one too — makes it illegal.
    fn hyperedge_move_is_legal(&self, g: &Hypergraph, e: usize, to: usize, st: &State, upper: &[Weights], lower: &[Weights]) -> bool {
        let mut balance = st.balance.clone();
        for &v in g.vertices(e) {
            if g.has_fixed_vertices() && g.fixed_attr(v) != to as i32 {
                return false;
            }
            let from = st.solution[v];
            if from != to {
                balance[to] = add(&balance[to], g.vertex_weights(v));
                balance[from] = sub(&balance[from], g.vertex_weights(v));
            }
        }
        if all_less(&upper[to], &balance[to]) {
            return false;
        }
        (0..self.num_parts).all(|p| p == to || !all_less(&balance[p], &lower[p]))
    }

    /// `AcceptHyperedgeGain`.
    fn accept_hyperedge(&self, g: &Hypergraph, hg: &HyperedgeGain, total_gain: &mut f32, st: &mut State) {
        *total_gain += hg.gain;
        let to = hg.destination_part;
        for &v in g.vertices(hg.hyperedge) {
            let from = st.solution[v];
            if from == to {
                continue;
            }
            st.solution[v] = to;
            st.balance[from] = sub(&st.balance[from], g.vertex_weights(v));
            st.balance[to] = add(&st.balance[to], g.vertex_weights(v));
            for &f in g.edges(v) {
                st.net_degs[f][from] -= 1;
                st.net_degs[f][to] += 1;
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
        assert_eq!(neighbors(&g, 1, &[false, true, false, false], &[0, 0, 0, 0], None), vec![0, 3]);
        // With a pair, only vertices in its two blocks.
        assert_eq!(neighbors(&g, 1, &[false, true, false, false], &[2, 0, 0, 1], Some((0, 1))), vec![3]);
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

    fn greedy(max_move: usize) -> Greedy {
        Greedy { num_parts: 2, refiner_iters: 2, max_move, evaluator: ev() }
    }

    /// Greedy counts every cut hyperedge as a move and ENDS the pass when the count reaches
    /// `max_move` — before trying that hyperedge. Here the 2nd cut hyperedge (gain 1) is never
    /// tried with `max_move` 2; the 1st loses either way.
    #[test]
    fn greedy_stops_at_the_max_move_th_cut_hyperedge() {
        let edges = vec![vec![0, 3], vec![2, 5], vec![0, 1], vec![3, 4]];
        let g = graph(&[1.0; 6], &edges, &[1.0, 1.0, 10.0, 10.0], vec![]);
        let (upper, lower) = (vec![vec![10.0]; 2], vec![vec![0.0]; 2]);
        let mut sol = vec![0, 0, 0, 1, 1, 1];
        greedy(2).refine(&g, &upper, &lower, &mut sol);
        assert_eq!(sol, vec![0, 0, 0, 1, 1, 1]);
        greedy(3).refine(&g, &upper, &lower, &mut sol);
        assert_eq!(sol, vec![0, 0, 0, 1, 1, 0]);
    }

    /// The upper-bound check of a hyperedge move is the partitioner's ALL-dimensions `<`:
    /// [3, 9] < [4, 1] is false (9 >= 1), so a block reaching [4, 1] under bound [3, 9] is legal
    /// — a lexicographic compare would refuse it.
    #[test]
    fn greedy_upper_bound_is_all_dimensions() {
        let g = Hypergraph::new(2, 1, &[vec![0, 1]], vec![vec![2.0, 0.5]; 2], vec![vec![1.0]], vec![], vec![]);
        let (upper, lower) = (vec![vec![3.0, 9.0]; 2], vec![vec![0.0, 0.0]; 2]);
        let mut sol = vec![0, 1];
        greedy(50).refine(&g, &upper, &lower, &mut sol);
        assert_eq!(sol, vec![0, 0]);
    }

    /// The pairwise pass has NO illegal-start rule: best gain starts at 0, so from an illegal
    /// start a losing move is rolled back (the k-way FM pass keeps it — see
    /// `an_illegal_start_keeps_a_losing_move`).
    #[test]
    fn pairwise_rolls_back_a_losing_move_even_from_an_illegal_start() {
        let g = graph(&[2.0, 1.0, 1.0], &[vec![0, 1], vec![1, 2]], &[5.0, 1.0], vec![]);
        let mut sol = vec![0, 0, 1];
        KWayPm { fm: KWayFm::new(2, 2, 50, 25, ev()) }.refine(&g, &vec![vec![2.2]; 2], &vec![vec![1.8]; 2], &mut sol);
        assert_eq!(sol, vec![0, 0, 1]);
    }

    /// Pairs go by connectivity descending, then pair ascending, until all but at most one
    /// block are matched.
    #[test]
    fn matching_takes_the_strongest_free_pairs() {
        let scores = BTreeMap::from([((0, 1), 1.0), ((0, 2), 5.0), ((1, 2), 5.0), ((0, 3), 0.0), ((1, 3), 2.0), ((2, 3), 0.0)]);
        assert_eq!(maximum_match(4, &scores), vec![(0, 2), (1, 3)]);
        assert_eq!(maximum_match(3, &BTreeMap::from([((0, 1), 0.0), ((0, 2), 0.0), ((1, 2), 0.0)])), vec![(0, 1)]);
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
