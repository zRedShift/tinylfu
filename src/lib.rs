#![cfg_attr(feature = "stdavx512", feature(stdsimd))]
#![feature(allocator_api)]
#![feature(new_uninit)]

#[macro_use]
extern crate cfg_if;

pub mod sketch;

#[derive(Debug, Copy, Clone)]
pub struct LkkRemainder(u64, u32);

impl LkkRemainder {
    pub const fn new(divisor: u32) -> LkkRemainder {
        LkkRemainder(u64::MAX / divisor as u64 + 1, divisor)
    }
}

impl LkkRemainder {
    pub const fn rem(self, dividend: u32) -> u32 {
        ((self.0.wrapping_mul(dividend as u64) as u128 * self.1 as u128) >> 64) as u32
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MagicModulo {
    pub size: u32,
    mul: u32,
    shift: u8,
}

impl MagicModulo {
    pub const fn new(size: u32) -> MagicModulo {
        let mut size = if size > 0 { size } else { 0 };
        loop {
            if size == u32::MAX {
                return MagicModulo {
                    size: u32::MAX,
                    mul: 0x8000_0001,
                    shift: 63,
                };
            }
            let ilog2 = 31u8 - size.leading_zeros() as u8;
            if size & (size - 1) == 0 {
                return MagicModulo {
                    size,
                    mul: 1,
                    shift: ilog2,
                };
            }

            let u = 1u64 << (ilog2 + 32);
            let div = (u / size as u64) as u32;
            let rem = (u % size as u64) as u32;
            if size - rem < (1u32 << ilog2) {
                return MagicModulo {
                    size,
                    mul: div + 1,
                    shift: ilog2 + 32,
                };
            }
            size += 1;
        }
    }

    const fn div(&self, hash: u32) -> u32 {
        (((hash as u64) * (self.mul as u64)) >> self.shift) as u32
    }

    pub const fn rem(&self, hash: u32) -> u32 {
        hash - self.div(hash) * self.size
    }
}

#[cfg(test)]
mod tests {
    use super::{LkkRemainder, MagicModulo};
    use rand::Rng;

    #[test]
    fn test_magic_modulo() {
        let mut rng = rand::thread_rng();
        for _ in 0u32..=10_000_000 {
            let x: u32 = rng.gen();
            let size: u32 = rng.gen_range(1..=u32::MAX);
            let modulo = MagicModulo::new(size);
            assert_eq!(modulo.rem(x), x % modulo.size as u32);
        }
    }

    #[test]
    fn test_lkk() {
        let mut rng = rand::thread_rng();
        for _ in 0u32..=10_000_000 {
            let x: u32 = rng.gen();
            let size: u32 = rng.gen_range(1..=u32::MAX);
            let remainder = LkkRemainder::new(size);
            assert_eq!(remainder.rem(x), x % size);
        }
    }
}
