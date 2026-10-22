// SPDX-License-Identifier: Apache-2.0
//! The gain bucket of one destination block: a binary max-heap of vertex moves with a position
//! map, ordered by gain, then LIGHTER vertex, then smaller vertex id.

use crate::hypergraph::{add, all_less, lex_greater, sub, Hypergraph, Weights};

/// A candidate move. The default is the "no move" cell: vertex -1, gain -FLT_MAX.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexGain {
    pub vertex: i64,
    pub source_part: i64,
    pub destination_part: i64,
    pub gain: f32,
}

impl Default for VertexGain {
    fn default() -> Self {
        Self { vertex: -1, source_part: -1, destination_part: -1, gain: -f32::MAX }
    }
}

pub struct PriorityQueue {
    active: bool,
    vertices: Vec<VertexGain>,
    /// Heap position of each vertex, -1 when absent.
    map: Vec<i64>,
    maximum_traverse_level: usize,
}

impl PriorityQueue {
    /// Starts INACTIVE; only `set_active` changes that — inserting does not.
    pub fn new(total_vertices: usize, maximum_traverse_level: usize) -> Self {
        Self { active: false, vertices: Vec::new(), map: vec![-1; total_vertices], maximum_traverse_level }
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
    pub fn set_active(&mut self) {
        self.active = true;
    }
    pub fn set_deactive(&mut self) {
        self.active = false;
    }
    pub fn status(&self) -> bool {
        self.active
    }
    pub fn contains(&self, v: usize) -> bool {
        self.map[v] > -1
    }
    pub fn get_max(&self) -> VertexGain {
        self.vertices[0]
    }

    pub fn insert(&mut self, g: &Hypergraph, element: VertexGain) {
        self.vertices.push(element);
        let last = self.vertices.len() - 1;
        self.map[element.vertex as usize] = last as i64;
        self.heapify_up(g, last);
    }

    /// The root goes; the last element takes its place and sinks.
    pub fn extract_max(&mut self, g: &Hypergraph) -> VertexGain {
        let max = self.vertices[0];
        let last = self.vertices.len() - 1;
        self.vertices[0] = self.vertices[last];
        self.map[self.vertices[last].vertex as usize] = 0;
        self.vertices.pop();
        self.heapify_down(g, 0);
        self.map[max.vertex as usize] = -1;
        max
    }

    /// `Remove`: the element is given the root's gain + 1, floated up, and extracted.
    /// ⚠️ When + 1 is lost to rounding the element ties the root and the tie-break decides which
    /// one goes — the reference behaves the same.
    pub fn remove(&mut self, g: &Hypergraph, v: usize) {
        let index = self.map[v];
        if index == -1 {
            return;
        }
        let index = index as usize;
        self.vertices[index].gain = (self.get_max().gain as f64 + 1.0) as f32;
        self.heapify_up(g, index);
        self.extract_max(g);
        if self.vertices.is_empty() {
            self.active = false;
        }
    }

    /// `ChangePriority`: replaced in place, then up when the gain rose, else down.
    pub fn change_priority(&mut self, g: &Hypergraph, v: usize, new_element: VertexGain) {
        let index = self.map[v];
        if index == -1 {
            return;
        }
        let index = index as usize;
        let old = self.vertices[index].gain;
        self.vertices[index] = new_element;
        if new_element.gain > old {
            self.heapify_up(g, index);
        } else {
            self.heapify_down(g, index);
        }
    }

    /// `GetBestCandidate`: the root if moving it keeps both blocks STRICTLY inside their bounds,
    /// else down the larger child's path for up to `maximum_traverse_level` levels, taking the
    /// first level where a child qualifies (the right one only when larger than a qualifying left).
    pub fn best_candidate(&self, g: &Hypergraph, curr: &[Weights], upper: &[Weights], lower: &[Weights]) -> VertexGain {
        if self.vertices.is_empty() {
            return VertexGain::default();
        }
        let check_balance = |index: usize| {
            let e = &self.vertices[index];
            let w = g.vertex_weights(e.vertex as usize);
            let (to, from) = (e.destination_part as usize, e.source_part as usize);
            all_less(&add(&curr[to], w), &upper[to]) && lex_greater(&sub(&curr[from], w), &lower[from])
        };
        let mut index = 0usize;
        if check_balance(index) {
            return self.vertices[index];
        }
        let total = self.vertices.len();
        let mut candidate: Option<usize> = None;
        for _ in 0..self.maximum_traverse_level {
            let left = 2 * index + 1;
            if left < total && check_balance(left) {
                candidate = Some(left);
            }
            let right = 2 * index + 2;
            if right < total
                && check_balance(right)
                && candidate.is_none_or(|c| self.larger(g, right, c))
            {
                candidate = Some(right);
            }
            if let Some(c) = candidate {
                return self.vertices[c];
            }
            if left >= total || right >= total {
                return VertexGain::default();
            }
            index = if self.larger(g, right, left) { right } else { left };
        }
        VertexGain::default()
    }

    /// `CompareElementLargeThan`.
    fn larger(&self, g: &Hypergraph, a: usize, b: usize) -> bool {
        let (ga, gb) = (&self.vertices[a], &self.vertices[b]);
        if ga.gain > gb.gain {
            return true;
        }
        if ga.gain < gb.gain {
            return false;
        }
        let wa = g.vertex_weights(ga.vertex as usize);
        let wb = g.vertex_weights(gb.vertex as usize);
        if wa != wb {
            return all_less(wa, wb);
        }
        ga.vertex < gb.vertex
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.map[self.vertices[a].vertex as usize] = b as i64;
        self.map[self.vertices[b].vertex as usize] = a as i64;
        self.vertices.swap(a, b);
    }

    fn heapify_up(&mut self, g: &Hypergraph, mut index: usize) {
        while index > 0 && self.larger(g, index, (index - 1) / 2) {
            let parent = (index - 1) / 2;
            self.swap(index, parent);
            index = parent;
        }
    }

    fn heapify_down(&mut self, g: &Hypergraph, index: usize) {
        let total = self.vertices.len();
        let mut max = index;
        let left = 2 * index + 1;
        if left < total && self.larger(g, left, max) {
            max = left;
        }
        let right = 2 * index + 2;
        if right < total && self.larger(g, right, max) {
            max = right;
        }
        if max == index {
            return;
        }
        self.swap(index, max);
        self.heapify_down(g, max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(weights: &[f32]) -> Hypergraph {
        Hypergraph::new(1, 1, &[], weights.iter().map(|&w| vec![w]).collect(), vec![], vec![], vec![])
    }

    fn cell(v: i64, gain: f32) -> VertexGain {
        VertexGain { vertex: v, source_part: 0, destination_part: 1, gain }
    }

    /// Equal gains: the LIGHTER vertex is larger; equal weights: the smaller id.
    #[test]
    fn ties_go_to_the_lighter_then_smaller_vertex() {
        let g = graph(&[2.0, 1.0, 1.0]);
        let mut q = PriorityQueue::new(3, 25);
        q.insert(&g, cell(0, 5.0));
        q.insert(&g, cell(2, 5.0));
        q.insert(&g, cell(1, 5.0));
        assert_eq!(q.extract_max(&g).vertex, 1);
        assert_eq!(q.extract_max(&g).vertex, 2);
        assert_eq!(q.extract_max(&g).vertex, 0);
    }

    /// Removing the last element deactivates the bucket; inserting never re-activates it.
    #[test]
    fn remove_deactivates_and_insert_does_not_reactivate() {
        let g = graph(&[1.0, 1.0]);
        let mut q = PriorityQueue::new(2, 25);
        q.set_active();
        q.insert(&g, cell(0, 1.0));
        q.remove(&g, 0);
        assert!(!q.status() && q.is_empty() && !q.contains(0));
        q.insert(&g, cell(1, 1.0));
        assert!(!q.status());
    }

    /// A lowered priority sinks; the position map follows every swap.
    #[test]
    fn change_priority_keeps_the_map() {
        let g = graph(&[1.0; 4]);
        let mut q = PriorityQueue::new(4, 25);
        for v in 0..4 {
            q.insert(&g, cell(v, v as f32));
        }
        q.change_priority(&g, 3, cell(3, -1.0));
        assert_eq!(q.get_max().vertex, 2);
        q.remove(&g, 1);
        let order: Vec<i64> = std::iter::from_fn(|| (!q.is_empty()).then(|| q.extract_max(&g).vertex)).collect();
        assert_eq!(order, vec![2, 0, 3]);
    }

    /// The root that would overfill its destination is passed over for a child that fits.
    #[test]
    fn best_candidate_walks_to_a_legal_child() {
        let g = graph(&[5.0, 1.0, 1.0]);
        let mut q = PriorityQueue::new(3, 25);
        q.insert(&g, cell(0, 9.0));
        q.insert(&g, cell(1, 3.0));
        q.insert(&g, cell(2, 4.0));
        let curr = vec![vec![10.0], vec![2.0]];
        let upper = vec![vec![8.0], vec![6.0]];
        let lower = vec![vec![1.0], vec![1.0]];
        assert_eq!(q.best_candidate(&g, &curr, &upper, &lower).vertex, 2);
    }
}
