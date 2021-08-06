use super::CacheLine;
use std::arch::x86_64::*;

union SseUnion {
    arr: [u64; 2],
    sse: __m128i,
}

union CacheLineUnion {
    arr: CacheLine,
    sse: [__m128i; 4],
    #[cfg(target_feature = "avx2")]
    avx2: [__m256i; 2],
    #[cfg(feature = "stdavx512")]
    avx512: __m512i,
}

unsafe fn mask_deinterleave(x_mask: u16, y_mask: u16) -> SseUnion {
    let mut register = _mm_setr_epi16(x_mask as _, 0, 0, 0, y_mask as _, 0, 0, 0);
    register = _mm_or_si128(register, _mm_slli_epi64::<15>(register));
    register = _mm_or_si128(register, _mm_slli_epi64::<30>(register));
    let sse = _mm_and_si128(register, _mm_set1_epi8(0x11));
    SseUnion { sse }
}

unsafe fn mask_min(num: __m128i, mask: __m128i) -> u8 {
    let mut register = _mm_mullo_epi16(mask, _mm_set1_epi16(0xF));
    register = _mm_xor_si128(register, _mm_set1_epi8(-1));
    register = _mm_or_si128(num, register);
    register = _mm_min_epu8(register, _mm_slli_epi16::<4>(register));
    register = _mm_min_epu8(register, _mm_srli_epi16::<8>(register));
    (_mm_cvtsi128_si32(_mm_minpos_epu16(register)) >> 4) as _
}

unsafe fn mask_saturating_increment(num: __m128i, mask: __m128i) -> SseUnion {
    let mut register = _mm_and_si128(num, _mm_srli_epi16::<2>(num));
    register = _mm_and_si128(register, _mm_srli_epi16::<1>(register));
    let sse = _mm_add_epi8(num, _mm_andnot_si128(register, mask));
    SseUnion { sse }
}

pub(super) fn frequency((x, y): (u64, u64), (x_mask, y_mask): (u16, u16)) -> u8 {
    unsafe {
        let num = _mm_set_epi64x(y as i64, x as i64);
        let mask = mask_deinterleave(x_mask, y_mask).sse;
        mask_min(num, mask)
    }
}

pub(super) fn increment((x, y): (&mut u64, &mut u64), (x_mask, y_mask): (u16, u16)) -> (u8, bool) {
    let min;
    let [inc_x, inc_y] = unsafe {
        let num = _mm_set_epi64x(*y as i64, *x as i64);
        let mask = mask_deinterleave(x_mask, y_mask).sse;
        min = mask_min(num, mask);
        mask_saturating_increment(num, mask).arr
    };
    *x = inc_x;
    *y = inc_y;
    let full_sat = min == 0xF;
    (min + !full_sat as u8, full_sat)
}

unsafe fn reset_sse2(cache_line: &mut CacheLine) -> u8 {
    let mut sse2 = CacheLineUnion { arr: *cache_line }.sse;
    let mut counter = _mm_setzero_si128();
    for register in sse2.iter_mut() {
        counter = _mm_add_epi8(counter, _mm_and_si128(*register, _mm_set1_epi8(0x11)));
        *register = _mm_and_si128(_mm_srli_epi16::<1>(*register), _mm_set1_epi8(0x77));
    }
    *cache_line = CacheLineUnion { sse: sse2 }.arr;
    counter = _mm_add_epi8(counter, _mm_srli_epi16::<4>(counter));
    counter = _mm_and_si128(counter, _mm_set1_epi8(0x0F));
    counter = _mm_sad_epu8(counter, _mm_setzero_si128());
    (_mm_cvtsi128_si32(counter) + _mm_extract_epi16::<4>(counter)) as _
}

#[cfg(target_feature = "avx2")]
unsafe fn reset_avx2(cache_line: &mut CacheLine) -> u8 {
    let mut avx2 = CacheLineUnion { arr: *cache_line }.avx2;
    let mut counter = _mm256_and_si256(avx2[0], _mm256_set1_epi8(0x11));
    counter = _mm256_add_epi8(counter, _mm256_and_si256(avx2[1], _mm256_set1_epi8(0x11)));
    avx2[0] = _mm256_and_si256(_mm256_srli_epi64::<1>(avx2[0]), _mm256_set1_epi8(0x77));
    avx2[1] = _mm256_and_si256(_mm256_srli_epi64::<1>(avx2[1]), _mm256_set1_epi8(0x77));
    *cache_line = CacheLineUnion { avx2 }.arr;
    counter = _mm256_add_epi8(counter, _mm256_srli_epi16::<4>(counter));
    counter = _mm256_and_si256(counter, _mm256_set1_epi8(0x0F));
    counter = _mm256_sad_epu8(counter, _mm256_setzero_si256());
    let lo = _mm256_castsi256_si128(counter);
    let hi = _mm256_extracti128_si256::<1>(counter);
    let added = _mm_add_epi64(lo, hi);
    let unpacked = _mm_unpackhi_epi64(added, added);
    _mm_cvtsi128_si64(_mm_add_epi64(added, unpacked)) as _
}

#[cfg(feature = "stdavx512")]
unsafe fn reset_avx512(cache_line: &mut CacheLine) -> u8 {
    let mut register = CacheLineUnion { arr: *cache_line }.avx512;
    let avx512 = _mm512_and_si512(_mm512_srli_epi64::<1>(register), _mm512_set1_epi8(0x77));
    *cache_line = CacheLineUnion { avx512 }.arr;
    register = _mm512_and_si512(register, _mm512_set1_epi8(0x11));
    register = _mm512_add_epi8(register, _mm512_srli_epi16::<4>(register));
    register = _mm512_and_si512(register, _mm512_set1_epi8(0x0F));
    register = _mm512_sad_epu8(register, _mm512_setzero_si512());
    _mm512_reduce_add_epi64(register) as _
}

pub(super) fn reset(cache_line: &mut CacheLine) -> u8 {
    unsafe {
        cfg_if! {
            if #[cfg(all(
                target_feature = "avx512bw",
                target_feature = "avx512f",
                feature = "stdavx512",
            ))] {
                reset_avx512(cache_line)
            } else if #[cfg(target_feature = "avx2")] {
                reset_avx2(cache_line)
            } else {
                reset_sse2(cache_line)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use rand::Rng;
    use std::collections::HashSet;

    fn deinterleave(x: u16, y: u16) -> [u64; 2] {
        unsafe { mask_deinterleave(x, y).arr }
    }

    #[test]
    fn test_mask_deinterleave() {
        const MASK: u64 = 0xEEEE_EEEE_EEEE_EEEE;
        let mut iter = UNPACKED.iter().copied();
        let mut set = HashSet::new();
        while let (Some(x), Some(y)) = (iter.next(), iter.next()) {
            let [x_1, x_2] = deinterleave(x, x);
            assert_eq!(x_1, x_2);
            let [y_1, y_2] = deinterleave(y, y);
            assert_eq!(y_1, y_2);
            let [x_1, y_1] = deinterleave(x, y);
            let [y_2, x_2] = deinterleave(y, x);
            assert_eq!(x_1, x_2);
            assert_eq!(y_1, y_2);
            assert_eq!(x_1 & MASK, 0);
            assert_eq!(y_1 & MASK, 0);
            assert_eq!(x_1.count_ones(), 4);
            assert_eq!(y_1.count_ones(), 4);
            assert!(set.insert(x_1) && set.insert(y_1));
        }
        assert_eq!(set.len(), UNPACKED.len())
    }

    #[test]
    fn test_sat_inc_and_min() {
        fn simple_min(x: u64, mask: u64) -> u8 {
            use std::array::IntoIter;
            let masked = x | !(mask * 0xF);
            let hi = IntoIter::new(masked.to_le_bytes()).min().unwrap();
            let lo = IntoIter::new((masked << 4).to_le_bytes()).min().unwrap();
            hi.min(lo) >> 4
        }
        const fn simple_inc(x: u64, mask: u64) -> u64 {
            let sat = x & (x >> 2);
            x + (!(sat & (sat >> 1)) & mask)
        }
        let mut rng = rand::thread_rng();
        for _ in 0..(1 << 22) {
            let (nums, masks): ((u64, u64), (u16, u16)) = (rng.gen(), rng.gen());
            let mut nums_mut = nums;
            let (min_simd, sat) = increment((&mut nums_mut.0, &mut nums_mut.1), masks);
            let min_simd = min_simd + sat as u8;
            assert_eq!(min_simd, frequency(nums, masks) + 1);
            let [mask_1, mask_2] = deinterleave(masks.0, masks.1);
            let min_simple = simple_min(nums.0, mask_1).min(simple_min(nums.1, mask_2));
            assert_eq!(min_simd, min_simple + 1);
            assert_eq!(
                nums_mut,
                (simple_inc(nums.0, mask_1), simple_inc(nums.1, mask_2))
            );
        }
    }

    #[test]
    fn test_reset() {
        fn reset(cache_line: &mut CacheLine) -> u8 {
            let mut count = 0;
            for x in cache_line.0.iter_mut() {
                count += (*x & 0x1111_1111_1111_1111).count_ones();
                *x = (*x >> 1) & 0x7777_7777_7777_7777;
            }
            count as _
        }
        let mut rng = rand::thread_rng();
        for _ in 0..(1 << 22) {
            let mut cache_line = CacheLine(rng.gen());
            let mut cloned = cache_line;
            let count = unsafe { reset_sse2(&mut cache_line) };
            let count_simple = reset(&mut cloned);
            assert_eq!(count, count_simple);
            assert_eq!(cache_line.0, cloned.0);
        }
    }
}
