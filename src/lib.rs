//! A concurrent append-only container of immutable values, where reads return
//! stable references and can happen concurrently to a write.

#![forbid(
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block
)]
#![cfg_attr(not(test), forbid(clippy::undocumented_unsafe_blocks))]

use std::mem::MaybeUninit;
use std::ops::Index;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// A concurrent append-only [`Vec`]-like container.
///
/// Unlike [`Vec`], this collection provides the following features.
/// - Reads return stable references, i.e. a reference to an item stays valid
///   even when more values are appended to the collection.
/// - Lock-free reads can happen while a write operation (append) is on-going.
/// - However, multiple writes don't happen concurrently: under the hood a write
///   lock is held during each [`push()`](Self::push) operation.
///
/// This makes this collection useful for scenarios such as a caching system,
/// where a large number of readers are continuously reading items while new
/// items are occasionally added to the collection. In this cases, wrapping a
/// regular [`Vec`] inside a [`RwLock`](std::sync::RwLock) would be less
/// efficient, as readers and writers would block each other every time a value
/// is added to the collection.
///
/// The drawbacks are that this collection offers only a simple API (only
/// appends and indexing are supported), it takes a bit more space in memory
/// than [`Vec`] (a few hundred bytes of fixed overhead), and each operation
/// uses atomics (which involves some hardware synchronization).
///
/// Under the hood, this is implemented as a series of buckets with power-of-two
/// sizes. Each bucket is allocated only when needed (i.e. when all the previous
/// buckets are full). This ensures that the memory overhead of this collection
/// remains bounded by a factor two, similarly to a regular [`Vec`].
///
/// Because the values are spread over multiple buckets in memory, it's not
/// possible to obtain a slice to a sequence of items.
pub struct AppendVec<T> {
    /// Length of the collection.
    len: AtomicUsize,
    /// Pointers to allocated buckets of growing size, or null for
    /// not-yet-allocated buckets. [`bucket_len()`] gives the constant size of
    /// each bucket.
    buckets: [AtomicPtr<T>; usize::BITS as usize],
    /// Lock held during [`push()`](Self::push) operations.
    write_lock: Mutex<()>,
}

impl<T> Default for AppendVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AppendVec<T> {
    const ITEM_SIZE_LOG2: u32 = std::mem::size_of::<T>().next_power_of_two().ilog2();
    const MAX_LEN: usize = !0 >> (Self::ITEM_SIZE_LOG2 + 1);

    /// Creates a new, empty collection.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container = AppendVec::new();
    /// let index = container.push_mut(42);
    /// assert_eq!(container[index], 42);
    /// ```
    pub fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            buckets: [const { AtomicPtr::new(std::ptr::null_mut()) }; usize::BITS as usize],
            write_lock: Mutex::new(()),
        }
    }

    /// Creates a new, empty collection, pre-allocating space for at least
    /// `capacity` items.
    ///
    /// # Panics
    ///
    /// This function panics if the capacity exceeds the maximum allocation size
    /// for a collection of items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container = AppendVec::with_capacity(42);
    /// for i in 0..42 {
    ///     let index = container.push_mut(i);
    ///     assert_eq!(container[index], i);
    /// }
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity <= Self::MAX_LEN,
            "AppendVec: requested capacity is too large for the given type ({capacity}, {})",
            Self::MAX_LEN
        );

        let mut buckets = [std::ptr::null_mut(); usize::BITS as usize];
        if capacity != 0 {
            let (max_bucket, _) = bucketize(capacity - 1);
            for bucket in 0..=max_bucket {
                let bucket_len = bucket_len(bucket);
                let allocated = Box::<[T]>::new_uninit_slice(bucket_len);
                let bucket_ptr = Box::into_raw(allocated) as *mut MaybeUninit<T> as *mut T;
                buckets[bucket] = bucket_ptr;
            }
        }

        Self {
            len: AtomicUsize::new(0),
            buckets: buckets.map(AtomicPtr::new),
            write_lock: Mutex::new(()),
        }
    }

    /// Returns the length of this collection.
    ///
    /// Given that writes can happen concurrently, beware of
    /// [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use)
    /// bugs! The value returned here is only a lower-bound of the collection
    /// size.
    ///
    /// To know the index of an added item, use the return value of the
    /// [`push()`](Self::push) function.
    ///
    /// ```
    /// use appendvec::AppendVec;
    /// use std::thread;
    ///
    /// let container = AppendVec::with_capacity(42);
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push(i);
    ///             // There is only one writer thread.
    ///             assert_eq!(index, i);
    ///         }
    ///     });
    ///     s.spawn(|| {
    ///         loop {
    ///             let l = container.len();
    ///             if l != 0 {
    ///                 // The unique writer thread pushes values in order.
    ///                 assert_eq!(container[l - 1], l - 1);
    ///             }
    ///             if l == 42 {
    ///                 break;
    ///             }
    ///         }
    ///     });
    /// });
    /// ```
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns an estimate of the length of this collection, without performing
    /// any synchronization (i.e. using [`Relaxed`](Ordering::Relaxed)
    /// ordering).
    ///
    /// # Safety
    ///
    /// Passing the result of this function (minus one) to
    /// [`get_unchecked()`](Self::get_unchecked) without any other form of
    /// synchronization is unsound, as the write(s) of the appended value(s)
    /// cannot be assumed to have *happened before* the increment of the length
    /// observed here.
    ///
    /// If you don't know what ["happens
    /// before"](https://doc.rust-lang.org/stable/nomicon/atomics.html) entails,
    /// you should probably not use this function, and use the regular
    /// [`len()`](Self::len) function instead.
    ///
    /// ```
    /// use appendvec::AppendVec;
    /// use std::sync::Barrier;
    /// use std::thread;
    ///
    /// let container = AppendVec::with_capacity(42);
    /// let barrier = Barrier::new(2);
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push(i);
    ///         }
    ///         // Synchronizes all the writes with the reader thread.
    ///         barrier.wait();
    ///     });
    ///     s.spawn(|| {
    ///         // Wait for all values to be appended to the container, and synchronize.
    ///         barrier.wait();
    ///         // SAFETY: After the synchronization barrier, no more values are appended in the
    ///         // writer thread.
    ///         let len = unsafe { container.len_unsynchronized() };
    ///         for i in 0..len {
    ///             // SAFETY: The writer thread wrote until the container length before the
    ///             // synchronization barrier.
    ///             let value = unsafe { container.get_unchecked(i) };
    ///             assert_eq!(*value, i);
    ///         }
    ///     });
    /// });
    /// ```
    pub unsafe fn len_unsynchronized(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Adds the given item to this collection and returns the index at
    /// which it was added.
    ///
    /// Internally, a write lock is held during this function call, i.e. other
    /// writers must wait for this function to complete. However, readers are
    /// free to read items in the meantime.
    ///
    /// Note that while this internally appends the item at the end of the
    /// collection, and while a writer lock is internally held during the
    /// operation, there is no guarantee that the returned index will remain
    /// the last one (i.e. equal to `self.len() - 1`), as a concurrent write
    /// may happen immediately afterwards. Beware of
    /// [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use)
    /// bugs!
    ///
    /// See also [`push_mut()`](Self::push_mut), which is more efficient if
    /// you hold a mutable reference to this collection.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// // The container isn't mutable, we'll use concurrent interior mutability via the push()
    /// // function.
    /// let container = AppendVec::with_capacity(42);
    /// for i in 0..42 {
    ///     let index = container.push(i);
    ///     assert_eq!(container[index], i);
    /// }
    /// ```
    pub fn push(&self, t: T) -> usize {
        let guard = self.write_lock.lock().unwrap();

        let index = self.len.load(Ordering::Relaxed);
        if index == Self::MAX_LEN {
            // Drop the guard before panicking to avoid poisoning the Mutex.
            drop(guard);
            panic!("AppendVec is full: cannot push");
        }

        let (bucket, bucket_index) = bucketize(index);
        let bucket_ptr = {
            let ptr = self.buckets[bucket].load(Ordering::Relaxed);
            if !ptr.is_null() {
                ptr
            } else {
                let bucket_len = bucket_len(bucket);
                let allocated = Box::<[T]>::new_uninit_slice(bucket_len);
                let bucket_ptr = Box::into_raw(allocated) as *mut MaybeUninit<T> as *mut T;
                self.buckets[bucket].store(bucket_ptr, Ordering::Relaxed);
                bucket_ptr
            }
        };

        // SAFETY:
        // - bucket_index * size_of::<T>() fits in an isize, as promised by the
        //   bucketize() function, with an input index <= Self::MAX_LEN,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - ptr is properly aligned, non-null with correct provenance, because it's
        //   derived from bucket_ptr which is itself aligned as it was allocated from a
        //   boxed slice of Ts,
        // - ptr is valid for exclusive writes:
        //   - the Release store on the length just below ensures that no other thread
        //     is reading the value at the given index (via safe APIs),
        //   - the write lock ensures that no other thread is ever obtaining the same
        //     index, i.e. no other thread is ever writing to this index.
        unsafe { std::ptr::write(ptr, t) };
        self.len.store(index + 1, Ordering::Release);

        drop(guard);

        index
    }

    /// Adds the given item to this collection and returns the index at
    /// which it was added.
    ///
    /// Contrary to [`push()`](Self::push), no write lock is held internally
    /// because this function already takes an exclusive mutable reference
    /// to this collection.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container = AppendVec::with_capacity(42);
    /// for i in 0..42 {
    ///     let index = container.push_mut(i);
    ///     assert_eq!(container[index], i);
    /// }
    /// ```
    pub fn push_mut(&mut self, t: T) -> usize {
        let index = self.len.load(Ordering::Relaxed);
        assert_ne!(index, Self::MAX_LEN, "AppendVec is full: cannot push");

        let (bucket, bucket_index) = bucketize(index);
        let bucket_ptr = {
            let ptr = self.buckets[bucket].load(Ordering::Relaxed);
            if !ptr.is_null() {
                ptr
            } else {
                let bucket_len = bucket_len(bucket);
                let allocated = Box::<[T]>::new_uninit_slice(bucket_len);
                let bucket_ptr = Box::into_raw(allocated) as *mut MaybeUninit<T> as *mut T;
                self.buckets[bucket].store(bucket_ptr, Ordering::Relaxed);
                bucket_ptr
            }
        };

        // SAFETY:
        // - bucket_index * size_of::<T>() fits in an isize, as promised by the
        //   bucketize() function, with an input index <= Self::MAX_LEN,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - ptr is properly aligned, non-null with correct provenance, because it's
        //   derived from bucket_ptr which is itself aligned as it was allocated from a
        //   boxed slice of Ts,
        // - ptr is valid for exclusive writes as this function takes an exclusive `&mut
        //   self` parameter.
        unsafe { std::ptr::write(ptr, t) };
        self.len.store(index + 1, Ordering::Release);

        index
    }

    /// Obtain a reference to the item at the given index without performing
    /// bound checks.
    ///
    /// # Safety
    ///
    /// The passed `index` must be lower than the size of the collection, i.e. a
    /// call to [`push()`](Self::push) that returned `index` must have
    /// *happened before* this function call.
    ///
    /// If you don't know what ["happens
    /// before"](https://doc.rust-lang.org/stable/nomicon/atomics.html) entails,
    /// you should probably not use this function, and use the regular indexing
    /// syntax instead.
    ///
    /// ```
    /// use appendvec::AppendVec;
    /// use std::sync::Barrier;
    /// use std::thread;
    ///
    /// let container = AppendVec::with_capacity(42);
    /// let barrier = Barrier::new(2);
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push(i);
    ///         }
    ///         // Synchronizes all the writes with the reader thread.
    ///         barrier.wait();
    ///     });
    ///     s.spawn(|| {
    ///         // Wait for all values to be appended to the container, and synchronize.
    ///         barrier.wait();
    ///         // SAFETY: After the synchronization barrier, no more values are appended in the
    ///         // writer thread.
    ///         let len = unsafe { container.len_unsynchronized() };
    ///         for i in 0..len {
    ///             // SAFETY: The writer thread wrote until the container length before the
    ///             // synchronization barrier.
    ///             let value = unsafe { container.get_unchecked(i) };
    ///             assert_eq!(*value, i);
    ///         }
    ///     });
    /// });
    /// ```
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let (bucket, bucket_index) = bucketize(index);

        let bucket_ptr = self.buckets[bucket].load(Ordering::Relaxed) as *const T;
        debug_assert_ne!(bucket_ptr, std::ptr::null());

        // SAFETY:
        // - bucket_index * size_of::<T>() fits in an isize, as promised by the caller
        //   and the checks within push() and push_mut() that ensure that at most
        //   Self::MAX_LEN + 1 items are stored in this collection,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - ptr is properly aligned, non-null with correct provenance, because it's
        //   derived from bucket_ptr which is itself aligned as it was allocated from a
        //   boxed slice of Ts,
        // - ptr is valid for reads: the only write at the given index has happened
        //   before this read as promised by the caller.
        unsafe { &*ptr }
    }

    /// Obtain an iterator over this collection.
    ///
    /// Note that once this iterator has been created, it will
    /// not iterate over items added afterwards, even on the same thread. This
    /// is to minimize the number of atomic operations.
    ///
    /// ```
    /// use appendvec::AppendVec;
    /// use std::thread;
    ///
    /// let container = AppendVec::with_capacity(42);
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push(i);
    ///             assert_eq!(index, i);
    ///         }
    ///     });
    ///     s.spawn(|| {
    ///         loop {
    ///             let it = container.iter();
    ///             let len = it.len();
    ///             for (i, value) in it.enumerate() {
    ///                 assert_eq!(*value, i);
    ///             }
    ///             if len == 42 {
    ///                 break;
    ///             }
    ///         }
    ///     });
    /// });
    /// ```
    pub fn iter(&self) -> AppendVecIter<'_, T> {
        AppendVecIter {
            inner: self,
            len: self.len.load(Ordering::Acquire),
            index: 0,
            bucket_ptr: std::ptr::null(),
        }
    }
}

impl<T> Drop for AppendVec<T> {
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Acquire);
        if len == 0 {
            return;
        }

        let (max_bucket, max_index) = bucketize(len - 1);
        for bucket in 0..=max_bucket {
            let bucket_len = bucket_len(bucket);
            let bucket_items = if bucket != max_bucket {
                bucket_len
            } else {
                max_index + 1
            };

            let ptr: *mut T = self.buckets[bucket].load(Ordering::Relaxed);
            let slice: *mut [T] = std::ptr::slice_from_raw_parts_mut(ptr, bucket_items);
            // SAFETY:
            // - slice starts at the aligned and non-null ptr,
            // - slice is valid for reads, as all items until len have been written to
            //   before, as ensured by the Acquire load on self.len,
            // - slice is valid for writes, as this function takes an exclusive `&mut self`
            //   parameter,
            // - slice is valid for dropping, as it is a part of the leaked boxed slice of
            //   this bucket,
            // - nothing else is accessing the `slice` while `drop_in_place` is executing,
            //   as this function takes an exclusive `&mut self` parameter.
            unsafe { std::ptr::drop_in_place(slice) };

            // SAFETY:
            // - ptr has been allocated with the global allocator, as it is derived from a
            //   leaked boxed slice,
            // - T has the same alignement as what ptr was allocated with, because ptr
            //   derives from a boxed slice of Ts,
            // - ptr was allocated with T * bucket_len bytes,
            // - the length is zero, therefore lower than or equal to the capacity,
            // - the first 0 values are properly initialized values of type T,
            // - the allocated size in bytes isn't larger than isize::MAX, because that's
            //   derived from a leaked boxed slice.
            let vec: Vec<T> = unsafe { Vec::from_raw_parts(ptr, 0, bucket_len) };
            drop(vec);
        }
    }
}

impl<T> Index<usize> for AppendVec<T> {
    type Output = T;

    /// Obtain a reference to the item at the given index.
    ///
    /// # Panics
    ///
    /// The passed `index` must be lower than the size of the collection, i.e. a
    /// call to [`push()`](Self::push) that returned `index` must have
    /// happened before this function call. Otherwise, this function panics.
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len.load(Ordering::Acquire));
        let (bucket, bucket_index) = bucketize(index);

        let bucket_ptr = self.buckets[bucket].load(Ordering::Relaxed) as *const T;
        debug_assert_ne!(bucket_ptr, std::ptr::null());

        // SAFETY:
        // - the assertion at the beginning of this function, with the Acquire load on
        //   the length, ensures that an item was added at the given index before the
        //   rest of this function is executed,
        // - bucket_index * size_of::<T>() fits in an isize, as the checks within push()
        //   and push_mut() ensure that at most Self::MAX_LEN + 1 items are stored in
        //   this collection,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - ptr is properly aligned, non-null with correct provenance, because it's
        //   derived from bucket_ptr which is itself aligned as it was allocated from a
        //   boxed slice of Ts,
        // - ptr is valid for reads: the only write at the given index has happened
        //   before this read as ensured by the assertion with an Acquire load on the
        //   length at the beginning of this function.
        unsafe { &*ptr }
    }
}

/// Iterator over an [`AppendVec`].
///
/// Note that once this iterator has been created via the [`AppendVec::iter()`]
/// function, it will not iterate over items added afterwards, even on the same
/// thread. This is to minimize the number of atomic operations, and to allow
/// implementing [`ExactSizeIterator`].
pub struct AppendVecIter<'a, T> {
    inner: &'a AppendVec<T>,
    len: usize,
    index: usize,
    bucket_ptr: *const T,
}

impl<'a, T> Iterator for AppendVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.len {
            None
        } else {
            let (bucket, bucket_index) = bucketize(self.index);
            self.index += 1;

            if bucket_index == 0 {
                self.bucket_ptr = self.inner.buckets[bucket].load(Ordering::Relaxed) as *const T;
                debug_assert_ne!(self.bucket_ptr, std::ptr::null());
            }

            // SAFETY:
            // - the Acquire load in iter() ensures that all indices before self.len,
            //   including self.index, have been pushed before the iterator was created,
            // - bucket_index * size_of::<T>() fits in an isize, as the checks within push()
            //   and push_mut() ensure that at most Self::MAX_LEN + 1 items are stored in
            //   this collection,
            // - the entire range between self.bucket_ptr and ptr is derived from one
            //   allocation of bucket_len(bucket) items, as 0 <= bucket_index <
            //   bucket_len(bucket).
            let ptr = unsafe { self.bucket_ptr.add(bucket_index) };
            // SAFETY:
            // - ptr is properly aligned, non-null with correct provenance, because it's
            //   derived from self.bucket_ptr which is itself aligned as it was allocated
            //   from a boxed slice of Ts,
            // - ptr is valid for reads: the only write at the given index has happened
            //   before this read as ensured by the assertion with an Acquire load on the
            //   length in iter().
            unsafe { Some(&*ptr) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for AppendVecIter<'_, T> {}

/// Decomposes the given `index` into the bucket that contains it and the index
/// within that bucket.
///
/// This function guarantees that the returned (`bucket`, `bucket_index`)
/// satisfy:
/// - 0 <= `bucket` < [`usize::BITS`].
/// - if `bucket` == 0: 0 <= `bucket_index` < 2
/// - otherwise: 0 <= `bucket_index` < `1 << bucket`
const fn bucketize(index: usize) -> (usize, usize) {
    let bucket = (usize::BITS - 1).saturating_sub(index.leading_zeros());
    let bucket_index = if bucket == 0 {
        index
    } else {
        index - (1 << bucket)
    };
    (bucket as usize, bucket_index)
}

/// Returns the number of items held in the given `bucket`.
///
/// This is `2^bucket`, except for the first bucket which is of size 2 instead
/// of 1. This formula ensures that the sum of the sizes of all buckets before
/// bucket `n` is `2^n` (for `n > 0`).
const fn bucket_len(bucket: usize) -> usize {
    if bucket == 0 { 2 } else { 1 << bucket }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Deref;
    use std::thread;

    #[test]
    fn test_item_size_log2() {
        assert_eq!(AppendVec::<u8>::ITEM_SIZE_LOG2, 0);
        assert_eq!(AppendVec::<u16>::ITEM_SIZE_LOG2, 1);
        assert_eq!(AppendVec::<u32>::ITEM_SIZE_LOG2, 2);
        assert_eq!(AppendVec::<u64>::ITEM_SIZE_LOG2, 3);

        // The implementation uses the next power of 2.
        assert_eq!(AppendVec::<[u8; 2]>::ITEM_SIZE_LOG2, 1);
        assert_eq!(AppendVec::<[u8; 3]>::ITEM_SIZE_LOG2, 2);
        assert_eq!(AppendVec::<[u8; 4]>::ITEM_SIZE_LOG2, 2);
        assert_eq!(AppendVec::<[u8; 5]>::ITEM_SIZE_LOG2, 3);
        assert_eq!(AppendVec::<[u8; 6]>::ITEM_SIZE_LOG2, 3);
        assert_eq!(AppendVec::<[u8; 7]>::ITEM_SIZE_LOG2, 3);
    }

    #[test]
    fn test_max_len() {
        assert_eq!(AppendVec::<u8>::MAX_LEN, usize::MAX >> 1);
        assert_eq!(AppendVec::<u16>::MAX_LEN, usize::MAX >> 2);
        assert_eq!(AppendVec::<u32>::MAX_LEN, usize::MAX >> 3);
        assert_eq!(AppendVec::<u64>::MAX_LEN, usize::MAX >> 4);

        // The implementation uses the next power of 2.
        assert_eq!(AppendVec::<[u8; 2]>::MAX_LEN, usize::MAX >> 2);
        assert_eq!(AppendVec::<[u8; 3]>::MAX_LEN, usize::MAX >> 3);
        assert_eq!(AppendVec::<[u8; 4]>::MAX_LEN, usize::MAX >> 3);
        assert_eq!(AppendVec::<[u8; 5]>::MAX_LEN, usize::MAX >> 4);
        assert_eq!(AppendVec::<[u8; 6]>::MAX_LEN, usize::MAX >> 4);
        assert_eq!(AppendVec::<[u8; 7]>::MAX_LEN, usize::MAX >> 4);
    }

    #[test]
    fn test_bucketize() {
        assert_eq!(bucketize(0), (0, 0));
        assert_eq!(bucketize(1), (0, 1));
        assert_eq!(bucketize(2), (1, 0));
        assert_eq!(bucketize(3), (1, 1));
        assert_eq!(bucketize(4), (2, 0));
        assert_eq!(bucketize(5), (2, 1));
        assert_eq!(bucketize(6), (2, 2));
        assert_eq!(bucketize(7), (2, 3));
        assert_eq!(bucketize(8), (3, 0));
        assert_eq!(bucketize(9), (3, 1));
        assert_eq!(bucketize(10), (3, 2));
    }

    #[test]
    fn test_bucket_len() {
        assert_eq!(bucket_len(0), 2);
        assert_eq!(bucket_len(1), 2);
        assert_eq!(bucket_len(2), 4);
        assert_eq!(bucket_len(3), 8);
        assert_eq!(bucket_len(4), 16);
        assert_eq!(bucket_len(5), 32);
    }

    #[test]
    #[should_panic(expected = "AppendVec: requested capacity is too large for the given type")]
    fn test_with_overlarge_capacity() {
        let _ = AppendVec::<u8>::with_capacity(usize::MAX);
    }

    #[test]
    fn test_push_index() {
        let v = AppendVec::new();
        for i in 0..100 {
            assert_eq!(v.push(i), i);
        }
        for i in 0..100 {
            assert_eq!(v[i], i);
        }
    }

    #[test]
    fn test_index_concurrent_reads() {
        const NUM_READERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - 1;
                            assert_eq!(*v[last].deref(), last);
                            if len == NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            s.spawn(|| {
                for j in 0..NUM_ITEMS {
                    assert_eq!(v.push(Box::new(j)), j);
                }
            });
        });
    }

    #[test]
    fn test_index_concurrent_writes() {
        const NUM_WRITERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            s.spawn(|| {
                loop {
                    let len = v.len();
                    if len > 0 {
                        let last = len - 1;
                        assert!(*v[last].deref() <= last);
                        if len == NUM_WRITERS * NUM_ITEMS {
                            break;
                        }
                    }
                }
            });
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(v.push(Box::new(j)) >= j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_index_concurrent_readwrites() {
        const NUM_READERS: usize = 4;
        const NUM_WRITERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - 1;
                            assert!(*v[last].deref() <= last);
                            if len == NUM_WRITERS * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(v.push(Box::new(j)) >= j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_get_unchecked_concurrent_reads() {
        const NUM_READERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - 1;
                            let x = unsafe { v.get_unchecked(last) };
                            assert_eq!(*x.deref(), last);
                            if len == NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            s.spawn(|| {
                for j in 0..NUM_ITEMS {
                    assert_eq!(v.push(Box::new(j)), j);
                }
            });
        });
    }

    #[test]
    fn test_get_unchecked_concurrent_writes() {
        const NUM_WRITERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            s.spawn(|| {
                loop {
                    let len = v.len();
                    if len > 0 {
                        let last = len - 1;
                        let x = unsafe { v.get_unchecked(last) };
                        assert!(*x.deref() <= last);
                        if len == NUM_WRITERS * NUM_ITEMS {
                            break;
                        }
                    }
                }
            });
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(v.push(Box::new(j)) >= j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_get_unchecked_concurrent_readwrites() {
        const NUM_READERS: usize = 4;
        const NUM_WRITERS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ITEMS: usize = 1_000_000;
        #[cfg(miri)]
        const NUM_ITEMS: usize = 100;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - 1;
                            let x = unsafe { v.get_unchecked(last) };
                            assert!(*x.deref() <= last);
                            if len == NUM_WRITERS * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        assert!(v.push(Box::new(j)) >= j);
                    }
                });
            }
        });
    }
}
