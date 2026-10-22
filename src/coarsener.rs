// SPDX-License-Identifier: Apache-2.0
//! First-choice coarsening: each pass matches every vertex with its best-scoring neighbour (or the
//! cluster that neighbour already joined), then contracts the matched clusters into a coarser
//! hypergraph, merging hyperedges that became parallel. `lazy_first_choice` repeats the pass until
//! the graph is small enough or stops shrinking.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::evaluator::Evaluator;
use crate::hypergraph::{add, lex_greater, Hypergraph, Weights};
use crate::mt19937::Mt19937;
use crate::shuffle::deterministic_shuffle;

/// The order vertices are visited in by a matching pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoarsenOrder {
    /// Shuffled with the coarsener's seed (the default).
    Random,
    /// Vertex id order.
    Default,
}

#[derive(Clone, Debug)]
pub struct Coarsener {
    pub num_parts: usize,
    /// Hyperedges with more vertices than this are ignored (matching and contraction).
    pub thr_coarsen_hyperedge_size_skip: usize,
    /// Stop once a level has at most this many vertices ...
    pub thr_coarsen_vertices: usize,
    /// ... or at most this many hyperedges.
    pub thr_coarsen_hyperedges: usize,
    pub coarsening_ratio: f32,
    pub max_coarsen_iters: usize,
    /// Stop when a pass removes at most this fraction of the vertices.
    pub adj_diff_ratio: f32,
    /// No cluster may weigh more than this.
    pub thr_cluster_weight: Weights,
    pub random_seed: i32,
    pub vertex_order_choice: CoarsenOrder,
    pub evaluator: Evaluator,
}

/// A cluster's attributes while a matching pass builds them.
#[derive(Default)]
struct Clusters {
    /// The cluster of each vertex of the finer graph (-1 = not yet matched).
    of: Vec<i64>,
    weights: Vec<Weights>,
    community: Vec<i32>,
    fixed: Vec<i32>,
}

impl Clusters {
    /// A new cluster holding `v` alone.
    fn singleton(&mut self, g: &Hypergraph, v: usize) {
        self.of[v] = self.weights.len() as i64;
        self.weights.push(g.vertex_weights(v).clone());
        if g.has_community() {
            self.community.push(g.community(v));
        }
        if g.has_fixed_vertices() {
            self.fixed.push(g.fixed_attr(v));
        }
    }
}

impl Coarsener {
    /// `IncreaseRandomSeed`: called before each independent coarsening.
    pub fn increase_random_seed(&mut self) {
        self.random_seed += 1;
    }

    /// `LazyFirstChoice`: the sequence of ever coarser graphs, NOT including `g` itself (level 0).
    pub fn lazy_first_choice(&self, g: &Hypergraph) -> Vec<Hypergraph> {
        let mut levels: Vec<Hypergraph> = Vec::new();
        for _ in 0..self.max_coarsen_iters {
            let last = levels.last().unwrap_or(g);
            let pre = last.num_vertices();
            let hg = self.aggregate(last);
            let cur = hg.num_vertices();
            if pre == cur {
                break;
            }
            let (nv, ne) = (hg.num_vertices(), hg.num_hyperedges());
            levels.push(hg);
            // `(pre - cur) <= pre * adj_diff_ratio`: an int against a float product.
            if nv <= self.thr_coarsen_vertices
                || ne <= self.thr_coarsen_hyperedges
                || ((pre - cur) as f32) <= pre as f32 * self.adj_diff_ratio
            {
                break;
            }
        }
        levels
    }

    /// `GroupVertices` with no groups: see `group_vertices_with`.
    pub fn group_vertices(&self, g: &Hypergraph) -> Hypergraph {
        self.group_vertices_with(g, &[])
    }

    /// `GroupVertices` (`ClusterBasedGroupInfo` then one contraction — which drops single-vertex
    /// and oversized hyperedges and merges parallel ones). With no groups and no fixed vertices
    /// every vertex is its own cluster. Otherwise: one group per block of its fixed vertices,
    /// then the given groups, in that order; a group joins the HIGHEST cluster any of its
    /// vertices is already in (else a new one), the rest become new clusters in vertex order.
    /// ⚠️ A cluster a later group takes all vertices from stays, EMPTY (weight 0). A cluster's
    /// community and fixed block are the max of its vertices' (from -1); weights summed in
    /// vertex order.
    pub fn group_vertices_with(&self, g: &Hypergraph, groups: &[Vec<usize>]) -> Hypergraph {
        if groups.is_empty() && g.fixed_attr_size() == 0 {
            let of: Vec<i64> = (0..g.num_vertices() as i64).collect();
            return self.contraction(g, &of, g.all_vertex_weights().to_vec(), g.copy_community(), g.copy_fixed_attr());
        }
        let mut all: Vec<Vec<usize>> = vec![Vec::new(); self.num_parts];
        if g.fixed_attr_size() > 0 {
            for v in 0..g.num_vertices() {
                if g.fixed_attr(v) > -1 {
                    all[g.fixed_attr(v) as usize].push(v);
                }
            }
        }
        all.retain(|grp| !grp.is_empty());
        all.extend(groups.iter().cloned());
        let mut of = vec![-1i64; g.num_vertices()];
        let mut next = 0i64;
        for grp in &all {
            let mut cur = grp.iter().map(|&v| of[v]).max().unwrap_or(-1);
            if cur == -1 {
                cur = next;
                next += 1;
            }
            for &v in grp {
                of[v] = cur;
            }
        }
        for c in of.iter_mut() {
            if *c == -1 {
                *c = next;
                next += 1;
            }
        }
        let n = next as usize;
        let mut weights = vec![vec![0.0f32; g.vertex_dimensions()]; n];
        let mut community = if g.has_community() { vec![-1; n] } else { Vec::new() };
        let mut fixed = if g.has_fixed_vertices() { vec![-1; n] } else { Vec::new() };
        for (v, &cluster) in of.iter().enumerate() {
            let c = cluster as usize;
            if g.has_community() {
                community[c] = community[c].max(g.community(v));
            }
            if g.has_fixed_vertices() {
                fixed[c] = fixed[c].max(g.fixed_attr(v));
            }
            weights[c] = add(&weights[c], g.vertex_weights(v));
        }
        self.contraction(g, &of, weights, community, fixed)
    }

    /// `Aggregate`: one matching pass, then its contraction.
    fn aggregate(&self, g: &Hypergraph) -> Hypergraph {
        let c = self.vertex_matching(g);
        self.contraction(g, &c.of, c.weights, c.community, c.fixed)
    }

    /// `VertexMatching`.
    fn vertex_matching(&self, g: &Hypergraph) -> Clusters {
        let n = g.num_vertices();
        let mut c = Clusters { of: vec![-1; n], ..Default::default() };
        let mut unvisited: Vec<usize> = Vec::with_capacity(n);
        if !g.has_fixed_vertices() {
            unvisited.extend(0..n);
        } else {
            for v in 0..n {
                if g.fixed_attr(v) > -1 {
                    c.singleton(g, v);
                } else {
                    unvisited.push(v);
                }
            }
        }
        self.order_vertices(&mut unvisited);
        // `unvisited.size() / coarsening_ratio_`: a float quotient, truncated.
        let num_early_stop = (unvisited.len() as f32 / self.coarsening_ratio) as i64;
        let mut num_visited: i64 = 0;
        let mut it = 0;
        while it < unvisited.len() {
            let v = unvisited[it];
            if c.of[v] > -1 {
                it += 1;
                continue;
            }
            // Ordered by neighbour id: the best-neighbour scan below walks it in that order.
            let mut score: BTreeMap<usize, f32> = BTreeMap::new();
            for &he in g.edges(v) {
                let range = g.vertices(he);
                if range.len() <= 1 || range.len() > self.thr_coarsen_hyperedge_size_skip {
                    continue;
                }
                let he_score = self.evaluator.norm_edge_score(he, g);
                for &nbr in range {
                    if nbr == v {
                        continue;
                    }
                    if let Some(s) = score.get_mut(&nbr) {
                        *s += he_score;
                        continue;
                    }
                    if (g.has_fixed_vertices() && g.fixed_attr(nbr) > -1)
                        || (g.has_community() && g.community(v) != g.community(nbr))
                    {
                        continue;
                    }
                    let nbr_weight = if c.of[nbr] > -1 { &c.weights[c.of[nbr] as usize] } else { g.vertex_weights(nbr) };
                    if lex_greater(&add(g.vertex_weights(v), nbr_weight), &self.thr_cluster_weight) {
                        continue;
                    }
                    score.insert(nbr, he_score);
                }
            }
            if score.is_empty() {
                num_visited += 1;
                c.singleton(g, v);
                it += 1;
                continue;
            }
            // The best neighbour: highest score; on a tie, a later neighbour wins only if it is
            // still unmatched.
            let mut best_score = -f32::MAX;
            let mut best: Option<usize> = None;
            for (&u, &s) in &score {
                if s > best_score {
                    best = Some(u);
                    best_score = s;
                } else if s == best_score && c.of[u] == -1 {
                    best = Some(u);
                }
            }
            let Some(best) = best else {
                num_visited += 1;
                c.singleton(g, v);
                it += 1;
                continue;
            };
            if c.of[best] > -1 {
                // Join the cluster `best` is already in.
                num_visited += 1;
                let id = c.of[best] as usize;
                c.of[v] = id as i64;
                c.weights[id] = add(&c.weights[id], g.vertex_weights(v));
            } else {
                num_visited += 2;
                let id = c.weights.len() as i64;
                c.of[best] = id;
                c.of[v] = id;
                c.weights.push(add(g.vertex_weights(best), g.vertex_weights(v)));
                if g.has_community() {
                    c.community.push(g.community(v));
                }
                if g.has_fixed_vertices() {
                    c.fixed.push(g.fixed_attr(v));
                }
            }
            let remaining = n as i64 + c.weights.len() as i64 - num_visited;
            if remaining <= num_early_stop {
                // Every vertex not yet matched becomes its own cluster, in visiting order.
                for &rest in &unvisited[it + 1..] {
                    if c.of[rest] > -1 {
                        continue;
                    }
                    c.singleton(g, rest);
                }
                return c;
            }
            it += 1;
        }
        c
    }

    /// `OrderVertices`.
    fn order_vertices(&self, vertices: &mut [usize]) {
        match self.vertex_order_choice {
            CoarsenOrder::Random => {
                // `std::mt19937 generator(random_seed_)`: the int seed, taken modulo 2^32.
                let mut gen = Mt19937::new(self.random_seed as u32);
                deterministic_shuffle(vertices, &mut gen);
            }
            CoarsenOrder::Default => {}
        }
    }

    /// `Contraction`: hyperedges over clusters. A hyperedge that is single-vertex, oversized, or
    /// falls inside one cluster is dropped; one whose cluster set was already seen is merged into
    /// the first (weights added). Kept hyperedges are in first-seen order, each as its SORTED set
    /// of cluster ids.
    fn contraction(&self, g: &Hypergraph, of: &[i64], weights: Vec<Weights>, community: Vec<i32>, fixed: Vec<i32>) -> Hypergraph {
        let mut hyperedges_c: Vec<Vec<usize>> = Vec::new();
        let mut hyperedge_weights_c: Vec<Weights> = Vec::new();
        // The first hyperedge per hash, then any others with the same hash (different sets).
        let mut first: HashMap<u64, usize> = HashMap::new();
        let mut parallel: HashMap<u64, Vec<usize>> = HashMap::new();
        for e in 0..g.num_hyperedges() {
            let range = g.vertices(e);
            if range.len() <= 1 || range.len() > self.thr_coarsen_hyperedge_size_skip {
                continue;
            }
            let set: BTreeSet<usize> = range.iter().map(|&v| of[v] as usize).collect();
            if set.len() <= 1 {
                continue;
            }
            // `std::inner_product(set, set, size_t{0})`: the sum of squared cluster ids.
            let hash = set.iter().fold(0u64, |acc, &c| acc.wrapping_add((c as u64).wrapping_mul(c as u64)));
            let he: Vec<usize> = set.into_iter().collect();
            let Some(&rep) = first.get(&hash) else {
                first.insert(hash, hyperedges_c.len());
                hyperedges_c.push(he);
                hyperedge_weights_c.push(g.hyperedge_weights(e).clone());
                continue;
            };
            let found = if he == hyperedges_c[rep] {
                Some(rep)
            } else {
                parallel.get(&hash).and_then(|cands| cands.iter().copied().find(|&id| he == hyperedges_c[id]))
            };
            match found {
                None => {
                    parallel.entry(hash).or_default().push(hyperedges_c.len());
                    hyperedges_c.push(he);
                    hyperedge_weights_c.push(g.hyperedge_weights(e).clone());
                }
                Some(id) => hyperedge_weights_c[id] = add(&hyperedge_weights_c[id], g.hyperedge_weights(e)),
            }
        }
        let mut out = Hypergraph::new(
            g.vertex_dimensions(),
            g.hyperedge_dimensions(),
            &hyperedges_c,
            weights,
            hyperedge_weights_c,
            fixed,
            community,
        );
        out.reset_vertex_c_attr();
        for (v, &cluster) in of.iter().enumerate() {
            out.add_vertex_c_attr(cluster as usize, v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarsener(skip: usize) -> Coarsener {
        Coarsener {
            num_parts: 2,
            thr_coarsen_hyperedge_size_skip: skip,
            thr_coarsen_vertices: 200,
            thr_coarsen_hyperedges: 50,
            coarsening_ratio: 1.5,
            max_coarsen_iters: 20,
            adj_diff_ratio: 0.0001,
            thr_cluster_weight: vec![100.0],
            random_seed: 0,
            vertex_order_choice: CoarsenOrder::Default,
            evaluator: Evaluator { num_parts: 2, e_wt_factors: vec![1.0], v_wt_factors: vec![1.0] },
        }
    }

    /// Grouping alone: single-vertex and oversized hyperedges go, and two hyperedges over the same
    /// vertex set merge into the first, weights added — the second's position is not kept.
    #[test]
    fn grouping_drops_trivial_hyperedges_and_merges_parallel_ones() {
        let g = Hypergraph::new(
            1,
            1,
            &[vec![0], vec![1, 0], vec![2, 3, 0, 1], vec![0, 1], vec![2, 3]],
            vec![vec![1.0]; 4],
            vec![vec![1.0], vec![2.0], vec![4.0], vec![8.0], vec![16.0]],
            vec![],
            vec![],
        );
        let c = coarsener(3).group_vertices(&g);
        assert_eq!(c.num_hyperedges(), 2);
        // {0,1} first seen as e1 (weight 2), merged with e3 (8): sorted ids, summed weight.
        assert_eq!((c.vertices(0), c.hyperedge_weights(0)), (&[0usize, 1][..], &vec![10.0]));
        assert_eq!((c.vertices(1), c.hyperedge_weights(1)), (&[2usize, 3][..], &vec![16.0]));
    }

    /// Two different sets with the same sum of squares (0²+5² = 3²+4²) are NOT merged: the hash
    /// only finds candidates, the sets decide.
    #[test]
    fn a_hash_collision_does_not_merge_different_sets() {
        let g = Hypergraph::new(1, 1, &[vec![0, 5], vec![3, 4]], vec![vec![1.0]; 6], vec![vec![1.0]; 2], vec![], vec![]);
        assert_eq!(coarsener(50).group_vertices(&g).num_hyperedges(), 2);
    }

    /// Groups: a group joins the HIGHEST cluster one of its vertices is in, which can leave an
    /// earlier cluster EMPTY (weight 0); communities are the max over a cluster's vertices;
    /// ungrouped vertices follow in vertex order.
    #[test]
    fn grouping_merges_into_the_highest_cluster() {
        let mut g = Hypergraph::new(1, 1, &[vec![0, 4], vec![1, 3]], vec![vec![1.0], vec![2.0], vec![4.0], vec![8.0], vec![16.0]], vec![vec![1.0]; 2], vec![], vec![]);
        g.set_community(vec![3, 1, 0, 2, 5]);
        // [0,1] -> 0, [2] -> 1, [1,2,3] -> max(0,1) = 1 (cluster 0 keeps only vertex 0), 4 -> 2.
        let c = coarsener(50).group_vertices_with(&g, &[vec![0, 1], vec![2], vec![1, 2, 3]]);
        assert_eq!(c.num_vertices(), 3);
        assert_eq!(c.all_vertex_weights(), &[vec![1.0], vec![14.0], vec![16.0]]);
        assert_eq!((c.community(0), c.community(1), c.community(2)), (3, 2, 5));
        assert_eq!(c.vertex_c_attr(1), &[1, 2, 3]);
        // A cluster emptied entirely stays, weightless: [0] -> 0, [1] -> 1, [0,1] -> 1.
        let c = coarsener(50).group_vertices_with(&g, &[vec![0], vec![1], vec![0, 1]]);
        assert_eq!(c.num_vertices(), 5);
        assert_eq!(c.all_vertex_weights()[0], vec![0.0]);
        assert_eq!(c.vertex_c_attr(1), &[0, 1]);
    }
}
