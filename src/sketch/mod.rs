use ahash::RandomState;
use std::alloc::{Allocator, Global};
use std::hash::{BuildHasher, Hash};

#[cfg(target_arch = "x86_64")]
mod intrinsics;

macro_rules! cfn_assert {
    ($x:expr $(,)*) => {{
        let b: bool = $x;
        let _ = ASSERT[!b as usize];
    }};
}

const BINOMIAL_8_2: usize = 28;
const BINOMIAL_16_4: usize = 1_820;
const ROT16_LEN: usize = 112;
const PACKED_LEN: usize = 119;
const ASSERT: [(); 1] = [()];
const UNPACKED: [u16; BINOMIAL_16_4] = unpacked_4_bits_set();
const PACKED: [u16; PACKED_LEN] = packed_4_bits_set();
const BLOCKS: [(u8, u8); BINOMIAL_8_2] = block_indices();

const fn block_indices() -> [(u8, u8); BINOMIAL_8_2] {
    let (mut i, mut x) = (0, 0u8);
    let mut arr = [(0u8, 0u8); BINOMIAL_8_2];
    loop {
        if x.count_ones() == 2 {
            arr[i] = (x.trailing_zeros() as u8, 7 - x.leading_zeros() as u8);
            i += 1;
        }
        if x == u8::MAX {
            cfn_assert!(i == BINOMIAL_8_2);
            break arr;
        }
        x += 1;
    }
}

const fn unpacked_4_bits_set() -> [u16; BINOMIAL_16_4] {
    let (mut i, mut x) = (0, 0u16);
    let mut arr = [0; BINOMIAL_16_4];
    loop {
        if x.count_ones() == 4 {
            arr[i] = x;
            i += 1;
        }
        if x == u16::MAX {
            cfn_assert!(i == BINOMIAL_16_4);
            break arr;
        }
        x += 1;
    }
}

const fn packed_4_bits_set() -> [u16; PACKED_LEN] {
    let (mut i, mut rot16, mut rot4) = (0, 0, ROT16_LEN);
    let mut set = [false; 1 << 16];
    let mut packed = [0u16; PACKED_LEN];
    loop {
        if i == BINOMIAL_16_4 {
            cfn_assert!(rot16 == ROT16_LEN && rot4 == PACKED_LEN);
            break packed;
        }
        let x = UNPACKED[i];
        i += 1;
        if set[x as usize] {
            continue;
        }
        set[x as usize] = true;
        let mut rotation = 1;
        let self_rotation = loop {
            let rotated = x.rotate_left(rotation);
            if rotated == x {
                break rotation;
            }
            set[rotated as usize] = true;
            rotation += 1;
        };
        match self_rotation {
            4 => {
                packed[rot4] = x;
                rot4 += 1;
            }
            8 => {
                packed[rot4] = x;
                packed[rot4 + 1] = x.rotate_left(4);
                rot4 += 2;
            }
            _ => {
                cfn_assert!(self_rotation == 16);
                packed[rot16] = x;
                rot16 += 1;
            }
        }
    }
}

fn fast_range(hash: u32, range: u32) -> u32 {
    ((hash as u64 * range as u64) >> 32) as u32
}

fn index_and_rotation(idx: u32) -> (u32, u32) {
    const ROT16_END: i32 = ROT16_LEN as i32 * 16;
    let rot4 = idx as i32 - ROT16_END;
    if rot4 < 0 {
        (idx / 16, idx % 16)
    } else {
        (ROT16_LEN as u32 + rot4 as u32 / 4, rot4 as u32 % 4)
    }
}

fn rotate_hash(hash: &mut u64, rotation: u32) -> u32 {
    let h = *hash as u32;
    *hash = hash.rotate_left(rotation);
    h
}

fn four_bits_set_h(hash: u32) -> u16 {
    let (idx, rot) = index_and_rotation(fast_range(hash, BINOMIAL_16_4 as u32));
    unsafe { PACKED.get_unchecked(idx as usize).rotate_left(rot) }
}

fn block_indices_h(hash: u32) -> (usize, usize) {
    let (idx_1, idx_2) = BLOCKS[fast_range(hash, BINOMIAL_8_2 as u32) as usize];
    (idx_1 as usize, idx_2 as usize)
}

fn block_masks(hash: &mut u64) -> (u16, u16) {
    let (hash_1, hash_2) = (rotate_hash(hash, 12), rotate_hash(hash, 12));
    (four_bits_set_h(hash_1), four_bits_set_h(hash_2))
}

#[repr(C, align(64))]
#[derive(Default, Clone, Copy)]
struct CacheLine([u64; 8]);

impl CacheLine {
    fn index_h(&self, hash: u32) -> (u64, u64) {
        let (idx_1, idx_2) = block_indices_h(hash);
        unsafe { (*self.0.get_unchecked(idx_1), *self.0.get_unchecked(idx_2)) }
    }

    fn index_mut_h(&mut self, hash: u32) -> (&mut u64, &mut u64) {
        let (idx_1, idx_2) = block_indices_h(hash);
        unsafe {
            let block_1 = &mut *(self.0.get_unchecked_mut(idx_1 as usize) as *mut _);
            let block_2 = &mut *(self.0.get_unchecked_mut(idx_2 as usize) as *mut _);
            (block_1, block_2)
        }
    }

    pub fn frequency(&self, hash: &mut u64) -> u8 {
        intrinsics::frequency(self.index_h(rotate_hash(hash, 8)), block_masks(hash))
    }

    pub fn increment(&mut self, hash: &mut u64) -> (u8, bool) {
        intrinsics::increment(self.index_mut_h(rotate_hash(hash, 8)), block_masks(hash))
    }
}

pub struct FrequencySketch<S: BuildHasher = RandomState, A: Allocator = Global> {
    sketch: Box<[CacheLine], A>,
    size: usize,
    sample_size: usize,
    hash_builder: S,
}

fn make_hash<Q: Hash + ?Sized, S: BuildHasher>(hash_builder: &S, val: &Q) -> u64 {
    use core::hash::Hasher;
    let mut state = hash_builder.build_hasher();
    val.hash(&mut state);
    state.finish()
}

fn cache_line_index(hash: u32, len: usize) -> usize {
    fast_range(hash, len as u32) as usize
}

impl FrequencySketch<RandomState, Global> {
    pub fn with_capacity(sketch_size: usize) -> Self {
        Self::with_capacity_and_hasher_in(sketch_size, RandomState::new(), Global)
    }
}

impl<S: BuildHasher, A: Allocator> FrequencySketch<S, A> {
    pub fn with_capacity_and_hasher_in(sketch_size: usize, hasher: S, alloc: A) -> Self {
        assert!(
            sketch_size > 0 && sketch_size <= u32::MAX as _,
            "0 < sketch <= u32::MAX"
        );
        let sketch = unsafe { Box::new_zeroed_slice_in(sketch_size, alloc).assume_init() };
        Self {
            sketch,
            size: 0,
            sample_size: sketch_size * 80,
            hash_builder: hasher,
        }
    }

    pub fn frequency<Q: Hash + ?Sized>(&self, key: &Q) -> u8 {
        let hash = &mut make_hash(&self.hash_builder, key);
        let index = cache_line_index(rotate_hash(hash, 32), self.sketch.len());
        self.sketch[index].frequency(hash)
    }

    pub fn increment<Q: Hash + ?Sized>(&mut self, key: &Q) -> u8 {
        let hash = &mut make_hash(&self.hash_builder, key);
        let index = cache_line_index(rotate_hash(hash, 32), self.sketch.len());
        let (frequency, saturated) = self.sketch[index].increment(hash);
        self.size += !saturated as usize;
        if self.size >= self.sample_size {
            self.reset();
        }
        frequency
    }

    pub fn reset(&mut self) {
        let mut count = 0;
        for cache_line in self.sketch.iter_mut() {
            count += intrinsics::reset(cache_line) as usize;
        }
        self.size = (self.size >> 1) - (count >> 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_packed_unpacked_eq() {
        let mut unpacked = [0; BINOMIAL_16_4];
        for (idx, elem) in unpacked.iter_mut().enumerate() {
            let (idx, rot) = index_and_rotation(idx as u32);
            *elem = PACKED[idx as usize].rotate_left(rot);
        }
        unpacked.sort_unstable();
        assert_eq!(unpacked, UNPACKED);
    }
}
