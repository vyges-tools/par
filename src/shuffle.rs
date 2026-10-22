// SPDX-License-Identifier: Apache-2.0
//! The partitioner's own shuffle, NOT a standard library's: Fisher-Yates from the back, each index
//! drawn by rejection sampling on the raw generator output. A library `shuffle` is free to draw
//! differently, so it is never used for these orders.

use crate::mt19937::Mt19937;

/// Shuffles `v` in place with `gen`.
pub fn deterministic_shuffle<T>(v: &mut [T], gen: &mut Mt19937) {
    let count = v.len();
    if count <= 1 {
        return;
    }
    let range: u64 = (Mt19937::MAX as u64) - (Mt19937::MIN as u64) + 1;
    for i in (1..count).rev() {
        let bound = i as u64 + 1;
        let limit = range - (range % bound);
        let value = loop {
            let value = gen.next_u32() as u64 - Mt19937::MIN as u64;
            if value < limit {
                break value;
            }
        };
        v.swap(i, (value % bound) as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each step swaps position i with `draw % (i + 1)`, from the back; the draws are the
    /// generator's raw values (none are rejected for small bounds unless they fall in the top
    /// `2^32 mod bound` values).
    #[test]
    fn each_step_swaps_with_the_draw_modulo_the_remaining_count() {
        let mut want: Vec<u32> = (0..5).collect();
        let mut g = Mt19937::new(1);
        for i in (1..5usize).rev() {
            let j = (g.next_u32() as u64 % (i as u64 + 1)) as usize;
            want.swap(i, j);
        }
        let mut got: Vec<u32> = (0..5).collect();
        deterministic_shuffle(&mut got, &mut Mt19937::new(1));
        assert_eq!(got, want);
    }

    /// Nothing to do for zero or one element: the generator is not drawn.
    #[test]
    fn a_single_element_draws_nothing() {
        let mut g = Mt19937::new(7);
        deterministic_shuffle(&mut [42], &mut g);
        assert_eq!(g.next_u32(), Mt19937::new(7).next_u32());
    }
}
