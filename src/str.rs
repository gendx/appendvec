//! A concurrent append-only string.

use super::{AppendVec, bucket_len, bucketize};
use std::ops::{Index, Range};
use std::sync::atomic::Ordering;

/// A concurrent append-only [`String`]-like container.
#[derive(Default)]
pub struct AppendStr {
    inner: AppendVec<u8>,
}

impl AppendStr {
    /// Creates a new, empty string.
    ///
    /// ```
    /// use appendvec::AppendStr;
    ///
    /// let mut container = AppendStr::new();
    /// let index = container.push_str_mut("Hello world!");
    /// assert_eq!(&container[index], "Hello world!");
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new, empty string, pre-allocating space for at least
    /// `capacity` bytes.
    ///
    /// # Panics
    ///
    /// This function panics if the capacity exceeds the maximum allocation
    /// size.
    ///
    /// ```
    /// use appendvec::AppendStr;
    ///
    /// let mut container = AppendStr::with_capacity(42);
    /// for i in 0..42 {
    ///     let bytes = [i];
    ///     let s = std::str::from_utf8(&bytes).unwrap();
    ///     let index = container.push_str_mut(s);
    ///     assert_eq!(&container[index], s);
    /// }
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: AppendVec::with_capacity(capacity),
        }
    }

    /// Returns the length in bytes of this collection.
    ///
    /// Given that writes can happen concurrently, beware of
    /// [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use)
    /// bugs! The value returned here is only a lower-bound of the collection
    /// size.
    ///
    /// To know the index of an added item, use the return value of the
    /// [`push_str()`](Self::push_str) function.
    ///
    /// ```
    /// use appendvec::AppendStr;
    /// use std::thread;
    ///
    /// let container = AppendStr::with_capacity(42);
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push_str(std::str::from_utf8(&[i]).unwrap());
    ///             // There is only one writer thread.
    ///             assert_eq!(index, i as usize..i as usize + 1);
    ///         }
    ///     });
    ///     s.spawn(|| {
    ///         loop {
    ///             let l = container.len();
    ///             if l != 0 {
    ///                 // The unique writer thread pushes values in order.
    ///                 assert_eq!(
    ///                     &container[l - 1..l],
    ///                     std::str::from_utf8(&[l as u8 - 1]).unwrap()
    ///                 );
    ///             }
    ///             if l == 42 {
    ///                 break;
    ///             }
    ///         }
    ///     });
    /// });
    /// ```
    #[expect(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Adds the given string to this collection, and returns the byte range at
    /// which it was added.
    ///
    /// The string is guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous string. If the current
    /// bucket doesn't have enough remaining capacity to accommodate this
    /// contiguous string, it will be padded with zero bytes (which are valid
    /// UTF-8).
    ///
    /// See also [`push_str_mut()`](Self::push_str_mut), which is more efficient
    /// if you hold a mutable reference to this collection.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size.
    ///
    /// ```
    /// use appendvec::AppendStr;
    ///
    /// let container = AppendStr::new();
    /// for i in 0..42 {
    ///     let blob = vec![123; i];
    ///     let s = std::str::from_utf8(blob.as_slice()).unwrap();
    ///     let index = container.push_str(s);
    ///     assert_eq!(&container[index], s);
    /// }
    /// ```
    pub fn push_str(&self, s: &str) -> Range<usize> {
        self.inner.push_slice_copy(s.as_bytes())
    }

    /// Adds the given string to this collection, and returns the byte range at
    /// which it was added.
    ///
    /// Contrary to [`push_str()`](Self::push_str), no write lock is held
    /// internally because this function already takes an exclusive mutable
    /// reference to this collection.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size.
    ///
    /// ```
    /// use appendvec::AppendStr;
    ///
    /// let mut container = AppendStr::new();
    /// for i in 0..42 {
    ///     let blob = vec![123; i];
    ///     let s = std::str::from_utf8(blob.as_slice()).unwrap();
    ///     let index = container.push_str_mut(s);
    ///     assert_eq!(&container[index], s);
    /// }
    /// ```
    pub fn push_str_mut(&mut self, s: &str) -> Range<usize> {
        self.inner.push_slice_copy_mut(s.as_bytes())
    }

    /// Gets the byte slice at the given index range, without checking that they
    /// are valid UTF-8.
    ///
    /// Bypassing the UTF-8 check can be useful if you'll only look at bytes
    /// anyway, for example for string comparison.
    ///
    /// # Panics
    ///
    /// The passed `index` range:
    /// - must be correctly ordered, i.e. `index.start <= index.end`,
    /// - must be lower than the size of the collection, i.e. a call to
    ///   [`push_str()`](Self::push_str) (or its variants) that returned a
    ///   superset of `index` must have happened before this function call,
    /// - must be contained in a single contiguous bucket; this is always the
    ///   case when passing a subset of a range returned by a previous call to
    ///   [`push_str()`](Self::push_str).
    ///
    /// Otherwise, this function panics.
    ///
    /// ```
    /// use appendvec::AppendStr;
    ///
    /// let container = AppendStr::new();
    /// for i in 0..42 {
    ///     let blob = vec![123; i];
    ///     let s = std::str::from_utf8(blob.as_slice()).unwrap();
    ///     let index = container.push_str(s);
    ///     assert_eq!(container.get_bytes(index), blob.as_slice());
    /// }
    /// ```
    pub fn get_bytes(&self, index: Range<usize>) -> &[u8] {
        &self.inner[index]
    }
}

impl Index<Range<usize>> for AppendStr {
    type Output = str;

    /// # Panics
    ///
    /// The passed `index` range:
    /// - must be correctly ordered, i.e. `index.start <= index.end`,
    /// - must be lower than the size of the collection, i.e. a call to
    ///   [`push_str()`](Self::push_str) (or its variants) that returned a
    ///   superset of `index` must have happened before this function call,
    /// - must be contained in a single contiguous bucket; this is always the
    ///   case when passing a subset of a range returned by a previous call to
    ///   [`push_str()`](Self::push_str),
    /// - must be fall on UTF-8 boundaries; this is always the case when exactly
    ///   passing a range returned by a previous call to
    ///   [`push_str()`](Self::push_str).
    ///
    /// Otherwise, this function panics.
    fn index(&self, index: Range<usize>) -> &Self::Output {
        if index.start == index.end {
            return "";
        }

        assert!(index.start <= index.end);
        let index_len = index.end - index.start;

        let len = self.inner.len.load(Ordering::Acquire);
        assert!(index.end <= len);
        let (bucket, bucket_index) = bucketize(index.start, self.inner.bucket_offset);
        let bucket_len = bucket_len(bucket, self.inner.bucket_offset);
        assert!(index_len <= bucket_len - bucket_index);

        let bucket_ptr = self.inner.buckets[bucket].load(Ordering::Relaxed) as *const u8;
        debug_assert_ne!(bucket_ptr, std::ptr::null());

        let (last_bucket, last_bucket_len) = bucketize(len, self.inner.bucket_offset);
        let bucket_available_len = if bucket == last_bucket {
            last_bucket_len
        } else {
            bucket_len
        };

        // SAFETY:
        // - bucket_ptr is non-null, as this collection has an invariant that all
        //   buckets until the length are non null, futher confirmed by a debug
        //   assertion,
        //   - note that the index range isn't empty in this code branch,
        // - bucket_ptr is aligned as u8 has an alignment of 1,
        // - bucket_ptr is valid for reads of bucket_available_len items of type u8, as
        //   the length is the length of the bucket, clamped to the current length if it
        //   is the last bucket,
        // - bucket_ptr points to bucket_available_len initialized values of type u8, as
        //   the collection maintains the invariant that all items until `self.len` are
        //   initialized,
        // - the memory isn't mutated for the whole output lifetime, which is bound by
        //   the lifetime of this collection,
        // - index_len * size_of::<u8>() fits in an isize, as the checks within push()
        //   and its variants ensure that at most AppendVec::MAX_LEN + 1 items are
        //   stored in this collection.
        let bucket_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(bucket_ptr, bucket_available_len) };
        // SAFETY:
        // - the collection maintains the invariant that all full buckets are valid
        //   UTF-8, as they are the concatenation of valid UTF-8 strings (pushed via the
        //   public API) followed by padding of zeros (which is also valid UTF-8),
        // - likewise, the last bucket is valid UTF-8 until the collection length, as it
        //   is the concatenation of valid UTF-8 strings that have been pushed.
        let bucket_str: &str = unsafe { std::str::from_utf8_unchecked(bucket_slice) };

        &bucket_str[bucket_index..bucket_index + index_len]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::thread;

    #[test]
    fn test_2_bytes_utf8() {
        let s = AppendStr::new();
        for _ in 0..100 {
            let index = s.push_str("é");
            assert_eq!(index.end, index.start + 2);
            assert_eq!(&s[index.clone()], "é");
            assert_eq!(s.get_bytes(index), &[0xc3, 0xa9]);
        }
    }

    #[test]
    fn test_3_bytes_utf8() {
        let s = AppendStr::new();
        for _ in 0..100 {
            let index = s.push_str("⠝");
            assert_eq!(index.end, index.start + 3);
            assert_eq!(&s[index.clone()], "⠝");
            assert_eq!(s.get_bytes(index), &[0xe2, 0xa0, 0x9d]);
        }
    }

    #[test]
    fn test_4_bytes_utf8() {
        let s = AppendStr::new();
        for _ in 0..100 {
            let index = s.push_str("💖");
            assert_eq!(index.end, index.start + 4);
            assert_eq!(&s[index.clone()], "💖");
            assert_eq!(s.get_bytes(index), &[0xf0, 0x9f, 0x92, 0x96]);
        }
    }

    #[test]
    #[should_panic(expected = "byte index 2 is not a char boundary; it is inside '⠝' (bytes 0..3")]
    fn test_index_non_utf8() {
        let s = AppendStr::new();
        let index = s.push_str("⠝");
        let _ = s[index.start..index.start + 2];
    }

    const NUM_READERS: usize = 4;
    const NUM_WRITERS: usize = 4;
    #[cfg(not(miri))]
    const NUM_ITEMS: usize = 1_000_000;
    #[cfg(miri)]
    const NUM_ITEMS: usize = 100;

    #[test]
    fn test_push_str_index_concurrent_reads() {
        const STR: &str = "⠝💖";
        const STR_LEN: usize = STR.len();

        let s = AppendStr::new();
        thread::scope(|scope| {
            for _ in 0..NUM_READERS {
                scope.spawn(|| {
                    loop {
                        let len = s.len();
                        if len > 0 {
                            let last = len - STR_LEN;
                            assert_eq!(&s[last..len], STR);
                            if len >= STR_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            scope.spawn(|| {
                for j in 0..NUM_ITEMS {
                    assert!(s.push_str(STR).start >= STR_LEN * j);
                }
            });
        });
    }

    #[test]
    fn test_push_str_index_concurrent_writes() {
        const STR: &str = "⠝💖";
        const STR_LEN: usize = STR.len();

        let s = AppendStr::new();
        thread::scope(|scope| {
            scope.spawn(|| {
                loop {
                    let len = s.len();
                    if len > 0 {
                        let last = len - STR_LEN;
                        assert_eq!(&s[last..len], STR);
                        if len >= NUM_WRITERS * STR_LEN * NUM_ITEMS {
                            break;
                        }
                    }
                }
            });
            for _ in 0..NUM_WRITERS {
                scope.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(s.push_str(STR).start >= STR_LEN * j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_push_str_index_concurrent_readwrites() {
        const STR: &str = "⠝💖";
        const STR_LEN: usize = STR.len();

        let s = AppendStr::new();
        thread::scope(|scope| {
            for _ in 0..NUM_READERS {
                scope.spawn(|| {
                    loop {
                        let len = s.len();
                        if len > 0 {
                            let last = len - STR_LEN;
                            assert_eq!(&s[last..len], STR);
                            if len >= NUM_WRITERS * STR_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                scope.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(s.push_str(STR).start >= STR_LEN * j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_with_little_capacity() {
        const STR: &str = "⠝💖";
        const STR_LEN: usize = STR.len();

        for capacity in [0, STR_LEN] {
            let s = AppendStr::with_capacity(capacity);
            thread::scope(|scope| {
                for _ in 0..NUM_READERS {
                    scope.spawn(|| {
                        loop {
                            let len = s.len();
                            if len > 0 {
                                let last = len - STR_LEN;
                                assert_eq!(&s[last..len], STR);
                                if len >= NUM_WRITERS * STR_LEN * NUM_ITEMS {
                                    break;
                                }
                            }
                        }
                    });
                }
                for _ in 0..NUM_WRITERS {
                    scope.spawn(|| {
                        for j in 0..NUM_ITEMS {
                            assert!(s.push_str(STR).start >= STR_LEN * j);
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn test_with_enough_capacity() {
        const STR: &str = "⠝💖";
        const STR_LEN: usize = STR.len();

        let s = AppendStr::with_capacity(NUM_WRITERS * STR_LEN * NUM_ITEMS);
        thread::scope(|scope| {
            for _ in 0..NUM_READERS {
                scope.spawn(|| {
                    loop {
                        let len = s.len();
                        if len > 0 {
                            let last = len - STR_LEN;
                            assert_eq!(&s[last..len], STR);
                            if len == NUM_WRITERS * STR_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                scope.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(s.push_str(STR).start >= STR_LEN * j);
                    }
                });
            }
        });
        assert_eq!(s.len(), NUM_WRITERS * STR_LEN * NUM_ITEMS);
    }
}
