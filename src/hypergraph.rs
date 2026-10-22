// SPDX-License-Identifier: Apache-2.0
//! The hypergraph: hyperedges as vertex lists (CSR `eptr`/`eind`), and each vertex's incident
//! hyperedges (CSR `vptr`/`vind`) in hyperedge order.
//!
//! Only what partitioning WITHOUT timing or placement needs: vertex and hyperedge weights, fixed
//! blocks, communities, and the map from each vertex to the finer vertices it was contracted from.

/// A weight per dimension.
pub type Weights = Vec<f32>;

#[derive(Clone, Debug)]
pub struct Hypergraph {
    vertex_dimensions: usize,
    hyperedge_dimensions: usize,
    vertex_weights: Vec<Weights>,
    hyperedge_weights: Vec<Weights>,
    eptr: Vec<usize>,
    eind: Vec<usize>,
    vptr: Vec<usize>,
    vind: Vec<usize>,
    /// The block of each fixed vertex (-1 = free); present only when given for every vertex.
    fixed: Option<Vec<i32>>,
    /// Each vertex's community; present only when given for every vertex.
    community: Option<Vec<i32>>,
    /// For a contracted graph: the vertices of the graph it was contracted from, per vertex.
    vertex_c_attr: Vec<Vec<usize>>,
}

impl Hypergraph {
    /// Builds both CSR views. ⚠️ An attribute vector counts only when its length is the vertex
    /// count; any other length (empty included) means "none".
    pub fn new(
        vertex_dimensions: usize,
        hyperedge_dimensions: usize,
        hyperedges: &[Vec<usize>],
        vertex_weights: Vec<Weights>,
        hyperedge_weights: Vec<Weights>,
        fixed: Vec<i32>,
        community: Vec<i32>,
    ) -> Self {
        let num_vertices = vertex_weights.len();
        let num_hyperedges = hyperedge_weights.len();
        let mut eptr = vec![0];
        let mut eind = Vec::new();
        for he in hyperedges {
            eind.extend_from_slice(he);
            eptr.push(eind.len());
        }
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
        for (e, he) in hyperedges.iter().enumerate().take(num_hyperedges) {
            for &v in he {
                incident[v].push(e);
            }
        }
        let mut vptr = vec![0];
        let mut vind = Vec::new();
        for edges in &incident {
            vind.extend_from_slice(edges);
            vptr.push(vind.len());
        }
        Self {
            vertex_dimensions,
            hyperedge_dimensions,
            vertex_weights,
            hyperedge_weights,
            eptr,
            eind,
            vptr,
            vind,
            fixed: (fixed.len() == num_vertices).then_some(fixed),
            community: (community.len() == num_vertices).then_some(community),
            vertex_c_attr: Vec::new(),
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.vertex_weights.len()
    }
    pub fn num_hyperedges(&self) -> usize {
        self.hyperedge_weights.len()
    }
    pub fn vertex_dimensions(&self) -> usize {
        self.vertex_dimensions
    }
    pub fn hyperedge_dimensions(&self) -> usize {
        self.hyperedge_dimensions
    }
    /// The vertices of hyperedge `e`, in the order given.
    pub fn vertices(&self, e: usize) -> &[usize] {
        &self.eind[self.eptr[e]..self.eptr[e + 1]]
    }
    /// The hyperedges incident to `v`, in hyperedge order.
    pub fn edges(&self, v: usize) -> &[usize] {
        &self.vind[self.vptr[v]..self.vptr[v + 1]]
    }
    pub fn vertex_weights(&self, v: usize) -> &Weights {
        &self.vertex_weights[v]
    }
    pub fn all_vertex_weights(&self) -> &[Weights] {
        &self.vertex_weights
    }
    pub fn hyperedge_weights(&self, e: usize) -> &Weights {
        &self.hyperedge_weights[e]
    }
    pub fn has_fixed_vertices(&self) -> bool {
        self.fixed.is_some()
    }
    pub fn fixed_attr(&self, v: usize) -> i32 {
        self.fixed.as_ref().map_or(-1, |f| f[v])
    }
    pub fn fixed_attr_size(&self) -> usize {
        self.fixed.as_ref().map_or(0, Vec::len)
    }
    pub fn has_community(&self) -> bool {
        self.community.is_some()
    }
    pub fn community(&self, v: usize) -> i32 {
        self.community.as_ref().map_or(-1, |c| c[v])
    }
    /// `SetCommunity`: from now on the graph HAS communities, whatever it had before.
    pub fn set_community(&mut self, attr: Vec<i32>) {
        self.community = Some(attr);
    }
    pub fn copy_community(&self) -> Vec<i32> {
        self.community.clone().unwrap_or_default()
    }
    pub fn copy_fixed_attr(&self) -> Vec<i32> {
        self.fixed.clone().unwrap_or_default()
    }

    /// `GetTotalVertexWeights`: the element-wise sum, in vertex order.
    pub fn total_vertex_weights(&self) -> Weights {
        let mut total = vec![0.0f32; self.vertex_dimensions];
        for w in &self.vertex_weights {
            total = add(&total, w);
        }
        total
    }

    /// `GetUpperVertexBalance`: per block, the total weight times (base share + ub_factor %).
    pub fn upper_vertex_balance(&self, num_parts: usize, ub_factor: f32, base_balance: &[f32]) -> Vec<Weights> {
        let total = self.total_vertex_weights();
        // `value += ub_factor * 0.01`: a float times a double literal, added to a float.
        let base: Vec<f32> = base_balance.iter().map(|&b| (b as f64 + ub_factor as f64 * 0.01) as f32).collect();
        (0..num_parts).map(|i| multiply_factor(&total, base[i])).collect()
    }

    /// `GetLowerVertexBalance`: as the upper one with the margin subtracted, floored at 0.
    pub fn lower_vertex_balance(&self, num_parts: usize, ub_factor: f32, base_balance: &[f32]) -> Vec<Weights> {
        let total = self.total_vertex_weights();
        let base: Vec<f32> = base_balance
            .iter()
            .map(|&b| {
                let v = (b as f64 - ub_factor as f64 * 0.01) as f32;
                // `std::max<double>(value, 0.0)`, stored back into the float.
                (v as f64).max(0.0) as f32
            })
            .collect();
        (0..num_parts).map(|i| multiply_factor(&total, base[i])).collect()
    }

    pub fn reset_vertex_c_attr(&mut self) {
        self.vertex_c_attr = vec![Vec::new(); self.num_vertices()];
    }
    pub fn add_vertex_c_attr(&mut self, cluster: usize, v: usize) {
        self.vertex_c_attr[cluster].push(v);
    }
    pub fn vertex_c_attr(&self, cluster: usize) -> &[usize] {
        &self.vertex_c_attr[cluster]
    }
}

/// Element-wise `a + b`.
pub fn add(a: &[f32], b: &[f32]) -> Weights {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `MultiplyFactor`.
pub fn multiply_factor(a: &[f32], factor: f32) -> Weights {
    a.iter().map(|x| x * factor).collect()
}

/// `DivideFactor`.
pub fn divide_factor(a: &[f32], factor: f32) -> Weights {
    a.iter().map(|x| x / factor).collect()
}

/// Element-wise `a - b`.
pub fn sub(a: &[f32], b: &[f32]) -> Weights {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// The partitioner's own `a < b` on weight vectors: EVERY dimension strictly less. It is the one
/// found for `<` inside the partitioner; `<=`, `>` and `>=` have no own version and are the
/// standard library's lexicographic ones.
/// Written as the reference's "stop at the first `a >= b`", so a NaN does not stop it.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn all_less(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b).all(|(x, y)| !(x >= y))
}

/// The standard library's lexicographic `a < b`.
pub fn lex_less(a: &[f32], b: &[f32]) -> bool {
    lex_greater(b, a)
}

/// The standard library's lexicographic `a <= b`.
pub fn lex_le(a: &[f32], b: &[f32]) -> bool {
    !lex_greater(a, b)
}

/// The standard library's lexicographic `a >= b`.
pub fn lex_ge(a: &[f32], b: &[f32]) -> bool {
    !lex_less(a, b)
}

/// The partitioner's `a <= b` on per-block weights: every block (up to the shorter list) is
/// all-less (`all_less`) or equal.
pub fn blocks_le(a: &[Weights], b: &[Weights]) -> bool {
    a.iter().zip(b).all(|(x, y)| all_less(x, y) || x == y)
}

/// `a > b` on weight vectors is the standard library's LEXICOGRAPHIC comparison (the partitioner
/// defines no `>` of its own): the first dimension where they differ decides.
pub fn lex_greater(a: &[f32], b: &[f32]) -> bool {
    for (x, y) in a.iter().zip(b) {
        if y < x {
            return true;
        }
        if x < y {
            return false;
        }
    }
    b.len() < a.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Hypergraph {
        // e0 = {0,1}, e1 = {1,2,3}, e2 = {3}
        Hypergraph::new(1, 1, &[vec![0, 1], vec![1, 2, 3], vec![3]], vec![vec![1.0]; 4], vec![vec![1.0]; 3], vec![], vec![])
    }

    /// Each vertex lists its hyperedges in hyperedge order, as both CSR views are built.
    #[test]
    fn incident_hyperedges_are_in_hyperedge_order() {
        let g = tiny();
        assert_eq!(g.edges(1), &[0, 1]);
        assert_eq!(g.edges(3), &[1, 2]);
        assert_eq!(g.vertices(1), &[1, 2, 3]);
    }

    /// An attribute counts only when it covers every vertex; an empty one is "none".
    #[test]
    fn an_attribute_counts_only_at_full_length() {
        assert!(!tiny().has_fixed_vertices());
        assert!(!tiny().has_community());
        let g = Hypergraph::new(1, 1, &[vec![0, 1]], vec![vec![1.0]; 2], vec![vec![1.0]], vec![-1], vec![0, 1]);
        assert!(!g.has_fixed_vertices());
        assert!(g.has_community());
    }

    /// Two equal blocks with a 1% margin: each may hold 51% and must hold 49% of the weight.
    #[test]
    fn the_balance_bounds_are_the_share_plus_and_minus_the_margin() {
        let g = tiny();
        assert_eq!(g.upper_vertex_balance(2, 1.0, &[0.5, 0.5]), vec![vec![4.0 * 0.51f32]; 2]);
        assert_eq!(g.lower_vertex_balance(2, 1.0, &[0.5, 0.5]), vec![vec![4.0 * 0.49f32]; 2]);
    }

    /// The partitioner's `<` needs EVERY dimension less; the standard one only the first that
    /// differs.
    #[test]
    fn all_less_is_not_lexicographic() {
        assert!(!all_less(&[1.0, 9.0], &[2.0, 0.0]));
        assert!(lex_less(&[1.0, 9.0], &[2.0, 0.0]));
        assert!(all_less(&[1.0, 0.0], &[2.0, 1.0]));
        assert!(blocks_le(&[vec![1.0, 0.0], vec![3.0, 3.0]], &[vec![2.0, 1.0], vec![3.0, 3.0]]));
        assert!(!blocks_le(&[vec![1.0, 2.0]], &[vec![2.0, 1.0]]));
    }

    /// Lexicographic: the first differing dimension decides, later ones do not matter.
    #[test]
    fn weight_comparison_is_lexicographic() {
        assert!(lex_greater(&[2.0, 0.0], &[1.0, 9.0]));
        assert!(!lex_greater(&[1.0, 9.0], &[2.0, 0.0]));
        assert!(!lex_greater(&[1.0], &[1.0]));
    }
}
