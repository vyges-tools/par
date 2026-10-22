// SPDX-License-Identifier: Apache-2.0
//! The 32-bit Mersenne Twister, as the C++ standard specifies `std::mt19937`: seeded from one
//! integer, so every library produces the same stream. Random orders here must be that stream.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// `std::mt19937 gen(seed)`: the seed is taken modulo 2^32.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        Self { state, index: N }
    }

    /// `URBG::min()` and `URBG::max()`.
    pub const MIN: u32 = 0;
    pub const MAX: u32 = u32::MAX;

    /// `gen()`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER) | (self.state[(i + 1) % N] & LOWER);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C++ standard fixes it: a default-constructed `std::mt19937` (seed 5489) yields
    /// 4123659995 as its 10000th value.
    #[test]
    fn the_ten_thousandth_value_of_the_default_seed_is_the_standards() {
        let mut g = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10_000 {
            v = g.next_u32();
        }
        assert_eq!(v, 4_123_659_995);
    }

    /// The first outputs for seed 1 (published for every conforming implementation).
    #[test]
    fn seed_one_starts_as_every_library_does() {
        let mut g = Mt19937::new(1);
        assert_eq!([g.next_u32(), g.next_u32(), g.next_u32()], [1_791_095_845, 4_282_876_139, 3_093_770_124]);
    }
}
