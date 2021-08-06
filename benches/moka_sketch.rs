pub struct MokaSketch {
    sample_size: usize,
    table_mask: usize,
    table: Vec<u64>,
    size: usize,
}

// A mixture of seeds from FNV-1a, CityHash, and Murmur3. (Taken from Caffeine)
static SEED: [u64; 4] = [
    0xc3a5_c85c_97cb_3127,
    0xb492_b66f_be98_f273,
    0x9ae1_6a3b_2f90_404f,
    0xcbf2_9ce4_8422_2325,
];

static RESET_MASK: u64 = 0x7777_7777_7777_7777;

static ONE_MASK: u64 = 0x1111_1111_1111_1111;

// -------------------------------------------------------------------------------
// Some of the code and doc comments in this module were ported or copied from
// a Java class `com.github.benmanes.caffeine.cache.MokaSketch` of Caffeine.
// https://github.com/ben-manes/caffeine/blob/master/caffeine/src/main/java/com/github/benmanes/caffeine/cache/MokaSketch.java
// -------------------------------------------------------------------------------
//
// MokaSketch maintains a 4-bit CountMinSketch [1] with periodic aging to
// provide the popularity history for the TinyLfu admission policy [2].
// The time and space efficiency of the sketch allows it to cheaply estimate the
// frequency of an entry in a stream of cache access events.
//
// The counter matrix is represented as a single dimensional array holding 16
// counters per slot. A fixed depth of four balances the accuracy and cost,
// resulting in a width of four times the length of the array. To retain an
// accurate estimation the array's length equals the maximum number of entries
// in the cache, increased to the closest power-of-two to exploit more efficient
// bit masking. This configuration results in a confidence of 93.75% and error
// bound of e / width.
//
// The frequency of all entries is aged periodically using a sampling window
// based on the maximum number of entries in the cache. This is referred to as
// the reset operation by TinyLfu and keeps the sketch fresh by dividing all
// counters by two and subtracting based on the number of odd counters
// found. The O(n) cost of aging is amortized, ideal for hardware pre-fetching,
// and uses inexpensive bit manipulations per array location.
//
// [1] An Improved Data Stream Summary: The Count-Min Sketch and its Applications
//     http://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf
// [2] TinyLFU: A Highly Efficient Cache Admission Policy
//     https://dl.acm.org/citation.cfm?id=3149371
//
// -------------------------------------------------------------------------------

impl MokaSketch {
    /// Creates a frequency sketch with the capacity.
    pub fn with_capacity(cap: usize) -> Self {
        let maximum = cap.min((i32::MAX >> 1) as usize);
        let table_size = if maximum == 0 {
            1
        } else {
            maximum.next_power_of_two()
        };
        let table = vec![0; table_size];
        let table_mask = 0.max(table_size - 1);
        let sample_size = if cap == 0 {
            10
        } else {
            maximum.saturating_mul(10).min(i32::MAX as usize)
        };
        Self {
            sample_size,
            table_mask,
            table,
            size: 0,
        }
    }

    /// Takes the hash value of an element, and returns the estimated number of
    /// occurrences of the element, up to the maximum (15).
    pub fn frequency(&self, hash: u64) -> u8 {
        let start = ((hash & 3) << 2) as u8;
        let mut frequency = u8::MAX;
        for i in 0..4 {
            let index = self.index_of(hash, i);
            let count = (self.table[index] >> ((start + i) << 2) & 0xF) as u8;
            frequency = frequency.min(count);
        }
        frequency
    }

    /// Take a hash value of an element and increments the popularity of the
    /// element if it does not exceed the maximum (15). The popularity of all
    /// elements will be periodically down sampled when the observed events
    /// exceeds a threshold. This process provides a frequency aging to allow
    /// expired long term entries to fade away.
    pub fn increment(&mut self, hash: u64) {
        let start = ((hash & 3) << 2) as u8;
        let mut added = false;
        for i in 0..4 {
            let index = self.index_of(hash, i);
            added |= self.increment_at(index, start + i);
        }

        if added {
            self.size += 1;
            if self.size >= self.sample_size {
                self.reset();
            }
        }
    }

    /// Takes a table index (each entry has 16 counters) and counter index, and
    /// increments the counter by 1 if it is not already at the maximum value
    /// (15). Returns `true` if incremented.
    fn increment_at(&mut self, table_index: usize, counter_index: u8) -> bool {
        let offset = (counter_index as usize) << 2;
        let mask = 0xF_u64 << offset;
        if self.table[table_index] & mask != mask {
            self.table[table_index] += 1u64 << offset;
            true
        } else {
            false
        }
    }

    /// Reduces every counter by half of its original value.
    pub fn reset(&mut self) {
        let mut count = 0u32;
        for entry in &mut self.table {
            // Count number of odd numbers.
            count += (*entry & ONE_MASK).count_ones();
            *entry = (*entry >> 1) & RESET_MASK;
        }
        self.size = (self.size >> 1) - (count >> 2) as usize;
    }

    /// Returns the table index for the counter at the specified depth.
    fn index_of(&self, hash: u64, depth: u8) -> usize {
        let i = depth as usize;
        let mut hash = hash.wrapping_add(SEED[i]).wrapping_mul(SEED[i]);
        hash += hash >> 32;
        hash as usize & self.table_mask
    }
}
