//! A concurrent append-only container of immutable values, where reads return
//! stable references and can happen concurrently to a write.

#![forbid(
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block
)]
#![cfg_attr(not(test), forbid(clippy::undocumented_unsafe_blocks))]

mod str;

use crossbeam_utils::CachePadded;
use std::mem::MaybeUninit;
use std::ops::{Index, Range};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
pub use str::AppendStr;

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
///
/// This container is [`Send`] and [`Sync`] if and only if `T` is both [`Send`]
/// and [`Sync`]. Indeed, sharing or sending an [`AppendVec`] allows retrieving
/// const references to `T` and moving values of type `T` across threads (where
/// the value is pushed vs. where it is dropped).
pub struct AppendVec<T> {
    /// Length of the collection.
    len: CachePadded<AtomicUsize>,
    /// Base-2 logarithm of the size of the first bucket.
    bucket_offset: u32,
    /// Pointers to allocated buckets of growing size, or null for
    /// not-yet-allocated buckets. [`bucket_len()`] gives the constant size of
    /// each bucket.
    buckets: [AtomicPtr<T>; usize::BITS as usize],
    /// Lock held during [`push()`](Self::push) operations.
    write_lock: Mutex<()>,
}

// SAFETY: Sending an AppendVec allows (1) retrieving a &T on another thread and
// (2) dropping T on another thread.
unsafe impl<T: Send + Sync> Send for AppendVec<T> {}
// SAFETY: Sharing an AppendVec allows (1) retrieving a &T on another thread and
// (2) importing T from another thread, via the push() method.
unsafe impl<T: Send + Sync> Sync for AppendVec<T> {}

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
            len: CachePadded::new(AtomicUsize::new(0)),
            bucket_offset: 1,
            buckets: [const { AtomicPtr::new(std::ptr::null_mut()) }; usize::BITS as usize],
            write_lock: Mutex::new(()),
        }
    }

    /// Creates a new, empty collection, pre-allocating space for at least
    /// `capacity` items.
    ///
    /// The requested `capacity` items are guaranteed to be allocated
    /// contiguously in a single bucket.
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
        let bucket_offset = if capacity == 0 {
            1
        } else {
            let bucket_offset = capacity.next_power_of_two().ilog2();
            debug_assert_eq!(bucketize(capacity - 1, bucket_offset).0, 0);

            let bucket_len = 1 << bucket_offset;
            let allocated = Box::<[T]>::new_uninit_slice(bucket_len);
            let bucket_ptr = Box::into_raw(allocated) as *mut MaybeUninit<T> as *mut T;
            buckets[0] = bucket_ptr;

            bucket_offset
        };

        Self {
            len: CachePadded::new(AtomicUsize::new(0)),
            bucket_offset,
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
    /// Note that even though this internally appends the item at the end of the
    /// collection, and a writer lock is internally held during the operation,
    /// there is no guarantee that the returned index will remain the last one
    /// (i.e. equal to `self.len() - 1`), as a concurrent write may happen
    /// immediately afterwards. Beware of
    /// [TOCTOU](https://en.wikipedia.org/wiki/Time-of-check_to_time-of-use)
    /// bugs!
    ///
    /// See also [`push_mut()`](Self::push_mut), which is more efficient if
    /// you hold a mutable reference to this [`AppendVec`] as it avoid acquiring
    /// a write lock.
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

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        let bucket_ptr = self.get_bucket_ptr(bucket, &guard);

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

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        let bucket_ptr = self.get_bucket_ptr_mut(bucket);

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
        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);

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
    /// Note that once this iterator has been created, it will not iterate over
    /// items added afterwards, even on the same thread. This is to minimize the
    /// number of atomic operations.
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
            bucket_offset: self.bucket_offset,
            len: self.len.load(Ordering::Acquire),
            index: 0,
            bucket_ptr: std::ptr::null(),
        }
    }

    /// Obtain an iterator over contiguous chunks of items of this collection.
    ///
    /// Note that once this iterator has been created, it will not iterate over
    /// items added afterwards, even on the same thread. This is to minimize the
    /// number of atomic operations.
    ///
    /// ```
    /// use appendvec::AppendVec;
    /// use std::thread;
    ///
    /// let container = AppendVec::new();
    /// thread::scope(|s| {
    ///     s.spawn(|| {
    ///         for i in 0..42 {
    ///             let index = container.push(i);
    ///             assert_eq!(index, i);
    ///         }
    ///     });
    ///     s.spawn(|| {
    ///         loop {
    ///             let mut i = 0;
    ///             for chunk in container.iter_chunks() {
    ///                 for value in chunk {
    ///                     assert_eq!(*value, i);
    ///                     i += 1;
    ///                 }
    ///             }
    ///             if i == 42 {
    ///                 break;
    ///             }
    ///         }
    ///     });
    /// });
    /// ```
    pub fn iter_chunks(&self) -> AppendVecChunksIter<'_, T> {
        let len = self.len.load(Ordering::Acquire);
        let (bucket, (max_bucket, max_index)) = if len == 0 {
            (self.bucket_offset as usize, (0, 0))
        } else {
            (0, bucketize(len - 1, self.bucket_offset))
        };
        AppendVecChunksIter {
            inner: self,
            bucket_offset: self.bucket_offset,
            bucket,
            max_bucket,
            max_index,
        }
    }

    /// Internal function to retrieve the pointer to the given bucket,
    /// allocating it if it's null.
    fn get_bucket_ptr_mut(&mut self, bucket: usize) -> *mut T {
        // SAFETY: The caller passed a mutable reference to self, which confirms
        // exclusive access.
        unsafe { self.get_bucket_ptr_raw(bucket) }
    }

    /// Internal function to retrieve the pointer to the given bucket,
    /// allocating it if it's null.
    fn get_bucket_ptr(&self, bucket: usize, _guard: &MutexGuard<()>) -> *mut T {
        // SAFETY: The caller passed a mutex guard, which confirms exclusive access.
        unsafe { self.get_bucket_ptr_raw(bucket) }
    }

    /// Internal function to retrieve the pointer to the given bucket,
    /// allocating it if it's null.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive write access to this object (as a new
    /// bucket may be allocated), by holding a mutable reference to self or
    /// the write lock.
    unsafe fn get_bucket_ptr_raw(&self, bucket: usize) -> *mut T {
        let ptr = self.buckets[bucket].load(Ordering::Relaxed);
        if !ptr.is_null() {
            ptr
        } else {
            let bucket_len = bucket_len(bucket, self.bucket_offset);
            let allocated = Box::<[T]>::new_uninit_slice(bucket_len);
            let bucket_ptr = Box::into_raw(allocated) as *mut MaybeUninit<T> as *mut T;
            self.buckets[bucket].store(bucket_ptr, Ordering::Relaxed);
            bucket_ptr
        }
    }
}

impl<T: Default> AppendVec<T> {
    /// Moves the given contiguous slice of items to this collection, and
    /// returns the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// If you have a mutable reference to this [`AppendVec`], calling
    /// [`push_owned_slice_mut()`](Self::push_owned_slice_mut) is more efficient
    /// as it avoids acquiring a write lock.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy()`](Self::push_slice_copy) instead on a borrowed slice
    /// may be more efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![format!("{i}"); i];
    ///     let index = container.push_owned_slice(blob.clone());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_owned_slice(&self, owned_slice: Vec<T>) -> Range<usize> {
        // SAFETY: The length of an iterator on Vec is trusted.
        unsafe { self.push_iterator(owned_slice.into_iter()) }
    }

    /// Moves the given contiguous slice of items to this collection, and
    /// returns the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// Contrary to [`push_owned_slice()`](Self::push_owned_slice), no write
    /// lock is held internally because this function already takes an
    /// exclusive mutable reference to this collection.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy_mut()`](Self::push_slice_copy_mut) instead may be more
    /// efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![format!("{i}"); i];
    ///     let index = container.push_owned_slice_mut(blob.clone());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_owned_slice_mut(&mut self, owned_slice: Vec<T>) -> Range<usize> {
        // SAFETY: The length of an iterator on Vec is trusted.
        unsafe { self.push_iterator_mut(owned_slice.into_iter()) }
    }

    /// Moves the given array of items to this collection, and returns the range
    /// at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// If you have a mutable reference to this [`AppendVec`], calling
    /// [`push_array_mut()`](Self::push_array_mut) is more efficient as it
    /// avoids acquiring a write lock.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy()`](Self::push_slice_copy) instead on a borrowed slice
    /// may be more efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob: [String; 11] = std::array::from_fn(|_| format!("{i}"));
    ///     let index = container.push_array(blob.clone());
    ///     assert_eq!(&container[index], &blob);
    /// }
    /// ```
    pub fn push_array<const N: usize>(&self, array: [T; N]) -> Range<usize> {
        // SAFETY: The length of an array iterator is trusted.
        unsafe { self.push_iterator(array.into_iter()) }
    }

    /// Moves the given array of items to this collection, and returns the range
    /// at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// Contrary to [`push_array()`](Self::push_array), no write lock is held
    /// internally because this function already takes an exclusive mutable
    /// reference to this collection.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy_mut()`](Self::push_slice_copy_mut) instead may be more
    /// efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob: [String; 11] = std::array::from_fn(|_| format!("{i}"));
    ///     let index = container.push_array_mut(blob.clone());
    ///     assert_eq!(&container[index], &blob);
    /// }
    /// ```
    pub fn push_array_mut<const N: usize>(&mut self, array: [T; N]) -> Range<usize> {
        // SAFETY: The length of an array iterator is trusted.
        unsafe { self.push_iterator_mut(array.into_iter()) }
    }

    /// Adds all the items from the given iterator to this collection, and
    /// returns the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice.
    ///
    /// # Safety
    ///
    /// - This function requires the iterator length to be correct.
    unsafe fn push_iterator(&self, iter: impl ExactSizeIterator<Item = T>) -> Range<usize> {
        let iter_len = iter.len();
        if iter_len == 0 {
            return 0..0;
        }

        let guard = self.write_lock.lock().unwrap();
        let (guard, index) = self.prepare_contiguous_slice(iter_len, guard);

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(iter_len <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr(bucket, &guard);

        // SAFETY:
        // - The iterator length is trusted to be correct,
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice(),
        // - no concurrent writes happen on this AppendVec, as the write lock is held,
        // - no concurrent reads happen beyond bucket_index on this bucket, as the
        //   length is only updated below, using a Release ordering.
        unsafe { self.move_iterator(bucket_ptr, bucket_index, iter) };
        self.len.store(index + iter_len, Ordering::Release);

        drop(guard);

        index..index + iter_len
    }

    /// Adds all the items from the given iterator to this collection, and
    /// returns the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice.
    ///
    /// # Safety
    ///
    /// - This function requires the iterator length to be correct.
    unsafe fn push_iterator_mut(&mut self, iter: impl ExactSizeIterator<Item = T>) -> Range<usize> {
        let iter_len = iter.len();
        if iter_len == 0 {
            return 0..0;
        }

        let index = self.prepare_contiguous_slice_mut(iter_len);

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(iter_len <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr_mut(bucket);

        // SAFETY:
        // - The iterator length is trusted to be correct,
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr_mut(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice_mut(),
        // - no concurrent reads nor writes happen on this AppendVec, as this function
        //   holds an exclusive reference to it.
        unsafe { self.move_iterator(bucket_ptr, bucket_index, iter) };
        self.len.store(index + iter_len, Ordering::Release);

        index..index + iter_len
    }

    /// Moves items from the given iterator into indices starting at
    /// `bucket_index` in the bucket starting at `bucket_ptr`.
    ///
    /// # Safety
    ///
    /// - The iterator length `iter.len()` must be correct.
    /// - The bucket pointer must be non-null, properly aligned and point to an
    ///   allocated bucket of `bucket_len` items belonging to this
    ///   [`AppendVec`], such that `bucket_index + iter.len() <= bucket_len`.
    /// - Items starting at `bucket_index` within the bucket must not be
    ///   initialized.
    /// - No concurrent reads or writes to the items starting at `bucket_index`
    ///   happen.
    unsafe fn move_iterator(
        &self,
        bucket_ptr: *mut T,
        bucket_index: usize,
        iter: impl ExactSizeIterator<Item = T>,
    ) {
        for (i, item) in iter.enumerate() {
            // SAFETY:
            // - the entire range between bucket_ptr and ptr is derived from one allocation
            //   of bucket_len(bucket) items, as 0 <= bucket_index + slice.len() <=
            //   bucket_len(bucket).
            // - (bucket_index + i) * size_of::<T>() fits in an isize, as promised by the
            //   bucketize() function, with an input index <= Self::MAX_LEN,
            let ptr = unsafe { bucket_ptr.add(bucket_index + i) };

            // SAFETY:
            // - ptr is properly aligned, non-null with correct provenance, because it's
            //   derived from bucket_ptr which is itself aligned as it was allocated from a
            //   boxed slice of Ts,
            // - ptr points to a non-initialized memory slot, as it is beyond bucket_index,
            // - ptr is valid for exclusive writes, as no concurrent reads or writes happen
            //   beyond bucket_index.
            unsafe { std::ptr::write(ptr, item) };
        }
    }

    fn prepare_contiguous_slice_mut(&mut self, slice_len: usize) -> usize {
        let mut index = self.len.load(Ordering::Relaxed);
        assert!(
            slice_len <= Self::MAX_LEN - index,
            "AppendVec is full: cannot push"
        );

        loop {
            let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
            let bucket_len = bucket_len(bucket, self.bucket_offset);
            if slice_len <= bucket_len - bucket_index {
                break;
            }

            let bucket_ptr = self.get_bucket_ptr_mut(bucket);
            for i in bucket_index..bucket_len {
                // SAFETY:
                // - i * size_of::<T>() fits in an isize, as promised by the bucket_len()
                //   function, with an input index <= Self::MAX_LEN,
                // - the entire range between bucket_ptr and ptr is derived from one allocation
                //   of bucket_len items, as 0 <= i < bucket_len.
                let ptr = unsafe { bucket_ptr.add(i) };
                // SAFETY:
                // - ptr is properly aligned, non-null with correct provenance, because it's
                //   derived from bucket_ptr which is itself aligned as it was allocated from a
                //   boxed slice of Ts,
                // - ptr is valid for exclusive writes as this function takes an exclusive `&mut
                //   self` parameter.
                unsafe { std::ptr::write(ptr, T::default()) };
            }

            index += bucket_len - bucket_index;
        }

        index
    }

    fn prepare_contiguous_slice<'guard>(
        &self,
        slice_len: usize,
        guard: MutexGuard<'guard, ()>,
    ) -> (MutexGuard<'guard, ()>, usize) {
        let mut index = self.len.load(Ordering::Relaxed);
        if slice_len > Self::MAX_LEN - index {
            // Drop the guard before panicking to avoid poisoning the Mutex.
            drop(guard);
            panic!("AppendVec is full: cannot push");
        }

        loop {
            let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
            let bucket_len = bucket_len(bucket, self.bucket_offset);
            if slice_len <= bucket_len - bucket_index {
                break;
            }

            let bucket_ptr = self.get_bucket_ptr(bucket, &guard);
            for i in bucket_index..bucket_len {
                // SAFETY:
                // - i * size_of::<T>() fits in an isize, as promised by the bucket_len()
                //   function, with an input index <= Self::MAX_LEN,
                // - the entire range between bucket_ptr and ptr is derived from one allocation
                //   of bucket_len items, as 0 <= i < bucket_len.
                let ptr = unsafe { bucket_ptr.add(i) };
                // SAFETY:
                // - ptr is properly aligned, non-null with correct provenance, because it's
                //   derived from bucket_ptr which is itself aligned as it was allocated from a
                //   boxed slice of Ts,
                // - ptr is valid for exclusive writes:
                //   - the Release store on the length just below ensures that no other thread
                //     is reading the value at the given index (via safe APIs),
                //   - the write lock ensures that no other thread is ever obtaining the same
                //     index, i.e. no other thread is ever writing to this index.
                unsafe { std::ptr::write(ptr, T::default()) };
            }

            index += bucket_len - bucket_index;
        }

        (guard, index)
    }
}

impl<T: Clone + Default> AppendVec<T> {
    /// Adds the given contiguous slice of items to this collection, and returns
    /// the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// If you have a mutable reference to this [`AppendVec`], calling
    /// [`push_slice_mut()`](Self::push_slice_mut) is more efficient as it
    /// avoids acquiring a write lock.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy()`](Self::push_slice_copy) instead may be more
    /// efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![format!("{i}"); i];
    ///     let index = container.push_slice(blob.as_slice());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_slice(&self, slice: &[T]) -> Range<usize> {
        if slice.is_empty() {
            return 0..0;
        }

        let guard = self.write_lock.lock().unwrap();
        let (guard, index) = self.prepare_contiguous_slice(slice.len(), guard);

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(slice.len() <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr(bucket, &guard);

        // SAFETY:
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice(),
        // - no concurrent writes happen on this AppendVec, as the write lock is held,
        // - no concurrent reads happen beyond bucket_index on this bucket, as the
        //   length is only updated below, using a Release ordering.
        unsafe { self.clone_slice(bucket_ptr, bucket_index, slice) };
        self.len.store(index + slice.len(), Ordering::Release);

        drop(guard);

        index..index + slice.len()
    }

    /// Adds the given contiguous slice of items to this collection, and returns
    /// the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// Contrary to [`push_slice()`](Self::push_slice), no write lock is held
    /// internally because this function already takes an exclusive mutable
    /// reference to this collection.
    ///
    /// If `T` implements [`Copy`], calling
    /// [`push_slice_copy_mut()`](Self::push_slice_copy_mut) instead may be more
    /// efficient.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container: AppendVec<String> = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![format!("i"); i];
    ///     let index = container.push_slice_mut(blob.as_slice());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_slice_mut(&mut self, slice: &[T]) -> Range<usize> {
        if slice.is_empty() {
            return 0..0;
        }

        let index = self.prepare_contiguous_slice_mut(slice.len());

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(slice.len() <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr_mut(bucket);

        // SAFETY:
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr_mut(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice_mut(),
        // - no concurrent reads nor writes happen on this AppendVec, as this function
        //   holds an exclusive reference to it.
        unsafe { self.clone_slice(bucket_ptr, bucket_index, slice) };
        self.len.store(index + slice.len(), Ordering::Release);

        index..index + slice.len()
    }

    /// Clones the given slice into indices starting at `bucket_index` in the
    /// bucket starting at `bucket_ptr`.
    ///
    /// # Safety
    ///
    /// - The bucket pointer must be non-null, properly aligned and point to an
    ///   allocated bucket of `bucket_len` items belonging to this
    ///   [`AppendVec`], such that `bucket_index + slice.len() <= bucket_len`.
    /// - Items starting at `bucket_index` within the bucket must not be
    ///   initialized.
    /// - No concurrent reads or writes to the items starting at `bucket_index`
    ///   happen.
    unsafe fn clone_slice(&self, bucket_ptr: *mut T, bucket_index: usize, slice: &[T]) {
        for (i, item) in slice.iter().enumerate() {
            // SAFETY:
            // - the entire range between bucket_ptr and ptr is derived from one allocation
            //   of bucket_len(bucket) items, as 0 <= bucket_index + slice.len() <=
            //   bucket_len(bucket).
            // - (bucket_index + i) * size_of::<T>() fits in an isize, as promised by the
            //   bucketize() function, with an input index <= Self::MAX_LEN,
            let ptr = unsafe { bucket_ptr.add(bucket_index + i) };

            let item: T = item.clone();
            // SAFETY:
            // - ptr is properly aligned, non-null with correct provenance, because it's
            //   derived from bucket_ptr which is itself aligned as it was allocated from a
            //   boxed slice of Ts,
            // - ptr points to a non-initialized memory slot, as it is beyond bucket_index,
            // - ptr is valid for exclusive writes, as no concurrent reads or writes happen
            //   beyond bucket_index.
            unsafe { std::ptr::write(ptr, item) };
        }
    }
}

impl<T: Copy + Default> AppendVec<T> {
    /// Adds the given contiguous slice of items to this collection, and returns
    /// the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// If you have a mutable reference to this [`AppendVec`], calling
    /// [`push_slice_copy_mut()`](Self::push_slice_copy_mut) is more efficient
    /// as it avoids acquiring a write lock.
    ///
    /// If `T` only implements [`Clone`], you can call
    /// [`push_slice()`](Self::push_slice) instead.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let container = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![123; i];
    ///     let index = container.push_slice_copy(blob.as_slice());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_slice_copy(&self, slice: &[T]) -> Range<usize> {
        if slice.is_empty() {
            return 0..0;
        }

        let guard = self.write_lock.lock().unwrap();
        let (guard, index) = self.prepare_contiguous_slice(slice.len(), guard);

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(slice.len() <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr(bucket, &guard);

        // SAFETY:
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice(),
        // - no concurrent writes happen on this AppendVec, as the write lock is held,
        // - no concurrent reads happen beyond bucket_index on this bucket, as the
        //   length is only updated below, using a Release ordering.
        unsafe { self.copy_slice(bucket_ptr, bucket_index, slice) };
        self.len.store(index + slice.len(), Ordering::Release);

        drop(guard);

        index..index + slice.len()
    }

    /// Adds the given contiguous slice of items to this collection, and returns
    /// the range at which they were added.
    ///
    /// The items are guaranteed to be pushed contiguously, so that indexing the
    /// result allows to retrieve back a contiguous slice. If the current bucket
    /// doesn't have enough remaining capacity to accommodate this contiguous
    /// slice, it will be padded with default items, hence the [`Default`]
    /// requirement for `T`.
    ///
    /// Contrary to [`push_slice_copy()`](Self::push_slice_copy), no write lock
    /// is held internally because this function already takes an exclusive
    /// mutable reference to this collection.
    ///
    /// If `T` only implements [`Clone`], you can call
    /// [`push_slice_mut()`](Self::push_slice_mut) instead.
    ///
    /// # Panics
    ///
    /// This function panics if this collection has reached the maximum
    /// allocation size for items of type `T`.
    ///
    /// ```
    /// use appendvec::AppendVec;
    ///
    /// let mut container = AppendVec::new();
    /// for i in 0..42 {
    ///     let blob = vec![123; i];
    ///     let index = container.push_slice_copy_mut(blob.as_slice());
    ///     assert_eq!(&container[index], blob.as_slice());
    /// }
    /// ```
    pub fn push_slice_copy_mut(&mut self, slice: &[T]) -> Range<usize> {
        if slice.is_empty() {
            return 0..0;
        }

        let index = self.prepare_contiguous_slice_mut(slice.len());

        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);
        debug_assert!(slice.len() <= bucket_len(bucket, self.bucket_offset) - bucket_index);
        let bucket_ptr = self.get_bucket_ptr_mut(bucket);

        // SAFETY:
        // - bucket_pointer is non-null, properly aligned and points to an allocated
        //   bucket, as ensured by get_bucket_ptr_mut(),
        // - items starting at bucket_index within this bucket are not yet initialized,
        //   as ensured by prepare_contiguous_slice_mut(),
        // - no concurrent reads nor writes happen on this AppendVec, as this function
        //   holds an exclusive reference to it.
        unsafe { self.copy_slice(bucket_ptr, bucket_index, slice) };
        self.len.store(index + slice.len(), Ordering::Release);

        index..index + slice.len()
    }

    /// Copies the given slice into indices starting at `bucket_index` in the
    /// bucket starting at `bucket_ptr`.
    ///
    /// # Safety
    ///
    /// - The bucket pointer must be non-null, properly aligned and point to an
    ///   allocated bucket of `bucket_len` items belonging to this
    ///   [`AppendVec`], such that `bucket_index + slice.len() <= bucket_len`.
    /// - Items starting at `bucket_index` within the bucket must not be
    ///   initialized.
    /// - No concurrent reads or writes to the items starting at `bucket_index`
    ///   happen.
    unsafe fn copy_slice(&self, bucket_ptr: *mut T, bucket_index: usize, slice: &[T]) {
        // SAFETY:
        // - bucket_index * size_of::<T>() fits in an isize, as promised by the
        //   bucketize() function, with an input index <= Self::MAX_LEN,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - slice.as_ptr() is valid for reading slice.len() items of type T,
        // - ptr is valid for writing slice.len() items of type T, indeed bucket_index +
        //   slice.len() <= bucket_len(bucket),
        // - slice.as_ptr() is properly aligned, as it directly derives from a slice,
        // - ptr is properly aligned, as it derives from bucket_ptr which is properly
        //   aligned,
        // - the memory regions don't overlap, as the input slice is passed as an
        //   external const reference parameter,
        // - T is Copy.
        unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len()) };
    }
}

impl<T> Drop for AppendVec<T> {
    fn drop(&mut self) {
        let len = self.len.load(Ordering::Acquire);
        if len == 0 {
            return;
        }

        let (max_bucket, max_index) = bucketize(len - 1, self.bucket_offset);
        for bucket in 0..=max_bucket {
            if bucket != 0 && bucket < self.bucket_offset as usize {
                continue;
            }

            let bucket_len = bucket_len(bucket, self.bucket_offset);
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
    /// call to [`push()`](Self::push) (or its variants) that returned `index`
    /// must have happened before this function call. Otherwise, this
    /// function panics.
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len.load(Ordering::Acquire));
        let (bucket, bucket_index) = bucketize(index, self.bucket_offset);

        let bucket_ptr = self.buckets[bucket].load(Ordering::Relaxed) as *const T;
        debug_assert_ne!(bucket_ptr, std::ptr::null());

        // SAFETY:
        // - the assertion at the beginning of this function, with the Acquire load on
        //   the length, ensures that an item was added at the given index before the
        //   rest of this function is executed,
        // - bucket_index * size_of::<T>() fits in an isize, as the checks within push()
        //   and its variants ensure that at most Self::MAX_LEN + 1 items are stored in
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

impl<T> Index<Range<usize>> for AppendVec<T> {
    type Output = [T];

    /// # Panics
    ///
    /// The passed `index` range:
    /// - must be correctly ordered, i.e. `index.start <= index.end`,
    /// - must be lower than the size of the collection, i.e. a call to
    ///   [`push_slice()`](Self::push_slice) (or its variants) that returned a
    ///   superset of `index` must have happened before this function call,
    /// - must be contained in a single contiguous bucket; this is always the
    ///   case when passing a subset of a range returned by a previous call to
    ///   [`push_slice()`](Self::push_slice).
    ///
    /// Otherwise, this function panics.
    fn index(&self, index: Range<usize>) -> &Self::Output {
        if index.start == index.end {
            return &[];
        }

        assert!(index.start <= index.end);
        let index_len = index.end - index.start;

        assert!(index.end <= self.len.load(Ordering::Acquire));
        let (bucket, bucket_index) = bucketize(index.start, self.bucket_offset);
        assert!(index_len <= bucket_len(bucket, self.bucket_offset) - bucket_index);

        let bucket_ptr = self.buckets[bucket].load(Ordering::Relaxed) as *const T;
        debug_assert_ne!(bucket_ptr, std::ptr::null());

        // SAFETY:
        // - the assertion comparing `index.end` with `self.len`, with the Acquire load
        //   on the length, ensures that items were added until the given index before
        //   the rest of this function is executed,
        // - bucket_index * size_of::<T>() fits in an isize, as the checks within push()
        //   and its variants ensure that at most Self::MAX_LEN + 1 items are stored in
        //   this collection,
        // - the entire range between bucket_ptr and ptr is derived from one allocation
        //   of bucket_len(bucket) items, as 0 <= bucket_index < bucket_len(bucket).
        let ptr = unsafe { bucket_ptr.add(bucket_index) };
        // SAFETY:
        // - ptr is non-null, as this collection has an invariant that all buckets until
        //   the length are non null, futher confirmed by a debug assertion,
        //   - note that the index range isn't empty in this code branch,
        // - ptr is aligned as it derives from the aligned bucket_ptr,
        // - ptr is valid for reads of index_len items of type T, as confirmed by the
        //   assertion that bucket_index + index_len <= bucket_len(bucket),
        // - ptr points to index_len initialized values of type T, as the collection
        //   maintains the invariant that all items until `self.len` are initialized,
        // - the memory isn't mutated for the whole output lifetime, which is bound by
        //   the lifetime of this collection,
        // - index_len * size_of::<T>() fits in an isize, as the checks within push()
        //   and its variants ensure that at most Self::MAX_LEN + 1 items are stored in
        //   this collection.
        unsafe { std::slice::from_raw_parts(ptr, index_len) }
    }
}

/// Iterator over an [`AppendVec`].
///
/// Note that once this iterator has been created via the [`AppendVec::iter()`]
/// function, it will not iterate over items added afterwards, even on the same
/// thread. This is to minimize the number of atomic operations, and to allow
/// implementing [`ExactSizeIterator`].
///
/// This is is [`Send`] and [`Sync`] if and only if `T` is [`Sync`], as sharing
/// or sending this iterator allows retrieving const references to `T`.
pub struct AppendVecIter<'a, T> {
    inner: &'a AppendVec<T>,
    bucket_offset: u32,
    len: usize,
    index: usize,
    bucket_ptr: *const T,
}

// SAFETY: Sending an AppendVecIter allows retrieving a &T on another thread.
unsafe impl<T: Sync> Send for AppendVecIter<'_, T> {}
// SAFETY: One cannot do much by sharing an AppendVecIter, but at most it would
// allow retrieving a &T on another thread.
unsafe impl<T: Sync> Sync for AppendVecIter<'_, T> {}

impl<'a, T> Iterator for AppendVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.len {
            None
        } else {
            let (bucket, bucket_index) = bucketize(self.index, self.bucket_offset);
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

/// Iterator over contiguous slices contained in an [`AppendVec`].
///
/// Note that once this iterator has been created via the
/// [`AppendVec::iter_chunks()`] function, it will not iterate over items added
/// afterwards, even on the same thread. This is to minimize the number of
/// atomic operations.
///
/// Also note that if you inserted slices via
/// [`push_slice()`](AppendVec::push_slice) or
/// [`push_slice_mut()`](AppendVec::push_slice_mut), this will also iterate over
/// the potential padding items, as [`AppendVec`] doesn't keep track of these.
///
/// This is is [`Send`] and [`Sync`] if and only if `T` is [`Sync`], as sharing
/// or sending this iterator allows retrieving const references to `T`.
pub struct AppendVecChunksIter<'a, T> {
    inner: &'a AppendVec<T>,
    bucket_offset: u32,
    bucket: usize,
    max_bucket: usize,
    max_index: usize,
}

// SAFETY: Sending an AppendVecChunksIter allows retrieving a &[T] on another
// thread.
unsafe impl<T: Sync> Send for AppendVecChunksIter<'_, T> {}
// SAFETY: One cannot do much by sharing an AppendVecChunksIter, but at most it
// would allow retrieving a &[T] on another thread.
unsafe impl<T: Sync> Sync for AppendVecChunksIter<'_, T> {}

impl<'a, T> Iterator for AppendVecChunksIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.bucket > self.max_bucket {
            None
        } else {
            let bucket_len = bucket_len(self.bucket, self.bucket_offset);
            let bucket_items = if self.bucket != self.max_bucket {
                bucket_len
            } else {
                self.max_index + 1
            };

            let bucket_ptr = self.inner.buckets[self.bucket].load(Ordering::Relaxed) as *const T;
            debug_assert_ne!(bucket_ptr, std::ptr::null());

            self.bucket = if self.bucket == 0 {
                self.bucket_offset as usize
            } else {
                self.bucket + 1
            };

            // SAFETY:
            // - bucket_ptr is non-null, as this collection has an invariant that all
            //   buckets until the length are non null, futher confirmed by a debug
            //   assertion,
            // - bucket_ptr is aligned as it derives from an allocation of type [T],
            // - bucket_ptr is valid for reads of bucket_items items of type T,
            // - bucket_ptr points to bucket_items initialized values of type T, as the
            //   AppendVec maintains the invariant that all items until its length are
            //   initialized,
            // - the memory isn't mutated for the whole output lifetime, which is bound by
            //   the lifetime of the underlying AppendVec,
            // - bucket_items * size_of::<T>() fits in an isize, as the checks within push()
            //   and its variants ensure that at most Self::MAX_LEN + 1 items are stored in
            //   an AppendVec.
            unsafe { Some(std::slice::from_raw_parts(bucket_ptr, bucket_items)) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        debug_assert!(self.bucket == 0 || self.bucket >= self.bucket_offset as usize);
        let adjusted_bucket = if self.bucket == 0 {
            0
        } else {
            self.bucket + 1 - self.bucket_offset as usize
        };

        debug_assert!(self.max_bucket == 0 || self.max_bucket >= self.bucket_offset as usize);
        let adjusted_max_bucket = if self.max_bucket == 0 {
            0
        } else {
            self.max_bucket + 1 - self.bucket_offset as usize
        };

        let remaining = adjusted_max_bucket + 1 - adjusted_bucket;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for AppendVecChunksIter<'_, T> {}

/// Decomposes the given `index` into the bucket that contains it and the index
/// within that bucket.
///
/// This function guarantees that the returned (`bucket`, `bucket_index`)
/// satisfy:
/// - 0 <= `bucket` < [`usize::BITS`].
/// - if `bucket` != 0, then `bucket` >= `bucket_offset`
/// - if `bucket` == 0: 0 <= `bucket_index` < 2^bucket_offset
/// - otherwise: 0 <= `bucket_index` < `1 << bucket`
const fn bucketize(index: usize, bucket_offset: u32) -> (usize, usize) {
    let mut bucket = (usize::BITS - 1).saturating_sub(index.leading_zeros());
    if bucket < bucket_offset {
        bucket = 0;
    }

    let bucket_index = if bucket == 0 {
        index
    } else {
        index - (1 << bucket)
    };

    (bucket as usize, bucket_index)
}

/// Returns the number of items held in the given `bucket`.
///
/// This is `2^bucket`, except for the first bucket which is of size
/// `2^bucket_offset` instead of 1. This formula ensures that the sum of the
/// sizes of all buckets before bucket `n` is `2^n` (for `n >= bucket_offset`).
const fn bucket_len(bucket: usize, bucket_offset: u32) -> usize {
    if bucket == 0 {
        1 << bucket_offset
    } else if bucket < bucket_offset as usize {
        panic!("Invalid bucket between 0 and bucket_offset");
    } else {
        1 << bucket
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Deref;
    use std::{array, thread};

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
        assert_eq!(bucketize(0, 1), (0, 0));
        assert_eq!(bucketize(1, 1), (0, 1));
        assert_eq!(bucketize(2, 1), (1, 0));
        assert_eq!(bucketize(3, 1), (1, 1));
        assert_eq!(bucketize(4, 1), (2, 0));
        assert_eq!(bucketize(5, 1), (2, 1));
        assert_eq!(bucketize(6, 1), (2, 2));
        assert_eq!(bucketize(7, 1), (2, 3));
        assert_eq!(bucketize(8, 1), (3, 0));
        assert_eq!(bucketize(9, 1), (3, 1));
        assert_eq!(bucketize(10, 1), (3, 2));

        assert_eq!(bucketize(0, 2), (0, 0));
        assert_eq!(bucketize(1, 2), (0, 1));
        assert_eq!(bucketize(2, 2), (0, 2));
        assert_eq!(bucketize(3, 2), (0, 3));
        assert_eq!(bucketize(4, 2), (2, 0));
        assert_eq!(bucketize(5, 2), (2, 1));
        assert_eq!(bucketize(6, 2), (2, 2));
        assert_eq!(bucketize(7, 2), (2, 3));
        assert_eq!(bucketize(8, 2), (3, 0));
        assert_eq!(bucketize(9, 2), (3, 1));
        assert_eq!(bucketize(10, 2), (3, 2));

        assert_eq!(bucketize(0, 3), (0, 0));
        assert_eq!(bucketize(1, 3), (0, 1));
        assert_eq!(bucketize(2, 3), (0, 2));
        assert_eq!(bucketize(3, 3), (0, 3));
        assert_eq!(bucketize(4, 3), (0, 4));
        assert_eq!(bucketize(5, 3), (0, 5));
        assert_eq!(bucketize(6, 3), (0, 6));
        assert_eq!(bucketize(7, 3), (0, 7));
        assert_eq!(bucketize(8, 3), (3, 0));
        assert_eq!(bucketize(9, 3), (3, 1));
        assert_eq!(bucketize(10, 3), (3, 2));
    }

    #[test]
    fn test_bucket_len() {
        assert_eq!(bucket_len(0, 1), 2);
        assert_eq!(bucket_len(1, 1), 2);
        assert_eq!(bucket_len(2, 1), 4);
        assert_eq!(bucket_len(3, 1), 8);
        assert_eq!(bucket_len(4, 1), 16);
        assert_eq!(bucket_len(5, 1), 32);

        assert_eq!(bucket_len(0, 2), 4);
        assert_eq!(bucket_len(2, 2), 4);
        assert_eq!(bucket_len(3, 2), 8);
        assert_eq!(bucket_len(4, 2), 16);
        assert_eq!(bucket_len(5, 2), 32);

        assert_eq!(bucket_len(0, 3), 8);
        assert_eq!(bucket_len(3, 3), 8);
        assert_eq!(bucket_len(4, 3), 16);
        assert_eq!(bucket_len(5, 3), 32);
    }

    #[test]
    #[should_panic(expected = "Invalid bucket between 0 and bucket_offset")]
    fn test_invalid_bucket_len() {
        let _ = bucket_len(1, 2);
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
    fn test_push_mut_index() {
        let mut v = AppendVec::new();
        for i in 0..100 {
            assert_eq!(v.push_mut(i), i);
        }
        for i in 0..100 {
            assert_eq!(v[i], i);
        }
    }

    #[test]
    fn test_push_slice_index() {
        let v = AppendVec::new();
        for len in 0..10 {
            let mut prev = 0..0;
            for i in 0..100 {
                let data = vec![i; len];
                let index = v.push_slice(&data);
                assert!(index.start >= prev.end);
                assert_eq!(index.end - index.start, len);
                prev = index.clone();
                assert_eq!(&v[index], &data);
            }
        }
    }

    #[test]
    fn test_push_slice_mut_index() {
        let mut v = AppendVec::new();
        for len in 0..10 {
            let mut prev = 0..0;
            for i in 0..100 {
                let data = vec![i; len];
                let index = v.push_slice_mut(&data);
                assert!(index.start >= prev.end);
                assert_eq!(index.end - index.start, len);
                prev = index.clone();
                assert_eq!(&v[index], &data);
            }
        }
    }

    #[test]
    fn test_push_slice_padding() {
        for len in 1..100 {
            let v = AppendVec::new();
            let range = v.push_slice(&vec![42; len]);
            assert_eq!(range.end - range.start, len);

            let bucket_start = if len <= 2 {
                0
            } else {
                let bucket = (len - 1).ilog2() + 1;
                1 << bucket
            };
            assert_eq!(range.start, bucket_start);
            for i in 0..range.start {
                assert_eq!(v[i], 0);
            }
        }
    }

    const NUM_READERS: usize = 4;
    const NUM_WRITERS: usize = 4;
    #[cfg(not(miri))]
    const NUM_ITEMS: usize = 100_000;
    #[cfg(miri)]
    const NUM_ITEMS: usize = 100;

    #[test]
    fn test_push_index_concurrent_reads() {
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
    fn test_push_index_concurrent_writes() {
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
    fn test_push_index_concurrent_readwrites() {
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
    fn test_iter() {
        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let mut iter = v.iter();

                        let mut remaining_len = iter.len();
                        assert_eq!(iter.size_hint(), (remaining_len, Some(remaining_len)));

                        let mut i = 0;
                        while let Some(x) = iter.next() {
                            assert_eq!(i, **x);
                            i += 1;
                            remaining_len -= 1;
                            assert_eq!(iter.size_hint(), (remaining_len, Some(remaining_len)));
                        }

                        if i == NUM_ITEMS {
                            break;
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
    fn test_iter_with_capacity() {
        for capacity in [0, NUM_ITEMS / 3] {
            let v: AppendVec<Box<usize>> = AppendVec::with_capacity(capacity);
            thread::scope(|s| {
                for _ in 0..NUM_READERS {
                    s.spawn(|| {
                        loop {
                            let mut iter = v.iter();

                            let mut remaining_len = iter.len();
                            assert_eq!(iter.size_hint(), (remaining_len, Some(remaining_len)));

                            let mut i = 0;
                            while let Some(x) = iter.next() {
                                assert_eq!(i, **x);
                                i += 1;
                                remaining_len -= 1;
                                assert_eq!(iter.size_hint(), (remaining_len, Some(remaining_len)));
                            }

                            if i == NUM_ITEMS {
                                break;
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
    }

    #[test]
    fn test_iter_chunks() {
        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let mut iter = v.iter_chunks();

                        let mut remaining_chunks = iter.len();
                        assert_eq!(iter.size_hint(), (remaining_chunks, Some(remaining_chunks)));

                        let mut i = 0;
                        while let Some(chunk) = iter.next() {
                            for x in chunk {
                                assert_eq!(i, **x);
                                i += 1;
                            }

                            remaining_chunks -= 1;
                            assert_eq!(
                                iter.size_hint(),
                                (remaining_chunks, Some(remaining_chunks))
                            );
                        }

                        if i == NUM_ITEMS {
                            break;
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
    fn test_iter_chunks_with_capacity() {
        for capacity in [0, NUM_ITEMS / 3] {
            let v: AppendVec<Box<usize>> = AppendVec::with_capacity(capacity);
            thread::scope(|s| {
                for _ in 0..NUM_READERS {
                    s.spawn(|| {
                        loop {
                            let mut iter = v.iter_chunks();

                            let mut remaining_chunks = iter.len();
                            assert_eq!(
                                iter.size_hint(),
                                (remaining_chunks, Some(remaining_chunks))
                            );

                            let mut i = 0;
                            while let Some(chunk) = iter.next() {
                                for x in chunk {
                                    assert_eq!(i, **x);
                                    i += 1;
                                }

                                remaining_chunks -= 1;
                                assert_eq!(
                                    iter.size_hint(),
                                    (remaining_chunks, Some(remaining_chunks))
                                );
                            }

                            if i == NUM_ITEMS {
                                break;
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
    }

    #[test]
    fn test_push_slice_index_concurrent_reads() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - SLICE_LEN;
                            let slice = &v[last..len];
                            assert!(*slice[0] * SLICE_LEN <= last);
                            for i in 1..SLICE_LEN {
                                assert_eq!(slice[i], slice[0]);
                            }
                            if len >= SLICE_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            s.spawn(|| {
                for j in 0..NUM_ITEMS {
                    let slice: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                    assert!(v.push_slice(&slice).start >= SLICE_LEN * j);
                }
            });
        });
    }

    #[test]
    fn test_push_slice_index_concurrent_writes() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            s.spawn(|| {
                loop {
                    let len = v.len();
                    if len > 0 {
                        let last = len - SLICE_LEN;
                        let slice = &v[last..len];
                        assert!(*slice[0] * SLICE_LEN <= last);
                        for i in 1..SLICE_LEN {
                            assert_eq!(slice[i], slice[0]);
                        }
                        if len >= NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                            break;
                        }
                    }
                }
            });
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        let slice: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                        assert!(v.push_slice(&slice).start >= SLICE_LEN * j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_push_slice_index_concurrent_readwrites() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> = AppendVec::new();
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - SLICE_LEN;
                            let slice = &v[last..len];
                            assert!(*slice[0] * SLICE_LEN <= last);
                            for i in 1..SLICE_LEN {
                                assert_eq!(slice[i], slice[0]);
                            }
                            if len >= NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        let slice: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                        assert!(v.push_slice(&slice).start >= SLICE_LEN * j);
                    }
                });
            }
        });
    }

    #[test]
    fn test_push_slice_with_little_capacity() {
        const SLICE_LEN: usize = 7;

        for capacity in [0, SLICE_LEN] {
            let v: AppendVec<Box<usize>> = AppendVec::with_capacity(capacity);
            thread::scope(|s| {
                for _ in 0..NUM_READERS {
                    s.spawn(|| {
                        loop {
                            let len = v.len();
                            if len > 0 {
                                let last = len - SLICE_LEN;
                                let slice = &v[last..len];
                                assert!(*slice[0] * SLICE_LEN <= last);
                                for i in 1..SLICE_LEN {
                                    assert_eq!(slice[i], slice[0]);
                                }
                                if len >= NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                    break;
                                }
                            }
                        }
                    });
                }
                for _ in 0..NUM_WRITERS {
                    s.spawn(|| {
                        for j in 0..NUM_ITEMS {
                            let slice: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                            assert!(v.push_slice(&slice).start >= SLICE_LEN * j);
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn test_push_slice_with_enough_capacity() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> =
            AppendVec::with_capacity(NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - SLICE_LEN;
                            let slice = &v[last..len];
                            assert!(*slice[0] * SLICE_LEN <= last);
                            for i in 1..SLICE_LEN {
                                assert_eq!(slice[i], slice[0]);
                            }
                            if len == NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        let slice: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                        assert!(v.push_slice(&slice).start >= SLICE_LEN * j);
                    }
                });
            }
        });
        assert_eq!(v.len(), NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
    }

    #[test]
    fn test_push_owned_slice_with_little_capacity() {
        const SLICE_LEN: usize = 7;

        for capacity in [0, SLICE_LEN] {
            let v: AppendVec<Box<usize>> = AppendVec::with_capacity(capacity);
            thread::scope(|s| {
                for _ in 0..NUM_READERS {
                    s.spawn(|| {
                        loop {
                            let len = v.len();
                            if len > 0 {
                                let last = len - SLICE_LEN;
                                let slice = &v[last..len];
                                assert!(*slice[0] * SLICE_LEN <= last);
                                for i in 1..SLICE_LEN {
                                    assert_eq!(slice[i], slice[0]);
                                }
                                if len >= NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                    break;
                                }
                            }
                        }
                    });
                }
                for _ in 0..NUM_WRITERS {
                    s.spawn(|| {
                        for j in 0..NUM_ITEMS {
                            let vec = vec![Box::new(j); SLICE_LEN];
                            assert!(v.push_owned_slice(vec).start >= SLICE_LEN * j);
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn test_push_owned_slice_with_enough_capacity() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> =
            AppendVec::with_capacity(NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - SLICE_LEN;
                            let slice = &v[last..len];
                            assert!(*slice[0] * SLICE_LEN <= last);
                            for i in 1..SLICE_LEN {
                                assert_eq!(slice[i], slice[0]);
                            }
                            if len == NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        let vec = vec![Box::new(j); SLICE_LEN];
                        assert!(v.push_owned_slice(vec).start >= SLICE_LEN * j);
                    }
                });
            }
        });
        assert_eq!(v.len(), NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
    }

    #[test]
    fn test_push_array_with_little_capacity() {
        const SLICE_LEN: usize = 7;

        for capacity in [0, SLICE_LEN] {
            let v: AppendVec<Box<usize>> = AppendVec::with_capacity(capacity);
            thread::scope(|s| {
                for _ in 0..NUM_READERS {
                    s.spawn(|| {
                        loop {
                            let len = v.len();
                            if len > 0 {
                                let last = len - SLICE_LEN;
                                let slice = &v[last..len];
                                assert!(*slice[0] * SLICE_LEN <= last);
                                for i in 1..SLICE_LEN {
                                    assert_eq!(slice[i], slice[0]);
                                }
                                if len >= NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                    break;
                                }
                            }
                        }
                    });
                }
                for _ in 0..NUM_WRITERS {
                    s.spawn(|| {
                        for j in 0..NUM_ITEMS {
                            let array: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                            assert!(v.push_array(array).start >= SLICE_LEN * j);
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn test_push_array_with_enough_capacity() {
        const SLICE_LEN: usize = 7;

        let v: AppendVec<Box<usize>> =
            AppendVec::with_capacity(NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
        thread::scope(|s| {
            for _ in 0..NUM_READERS {
                s.spawn(|| {
                    loop {
                        let len = v.len();
                        if len > 0 {
                            let last = len - SLICE_LEN;
                            let slice = &v[last..len];
                            assert!(*slice[0] * SLICE_LEN <= last);
                            for i in 1..SLICE_LEN {
                                assert_eq!(slice[i], slice[0]);
                            }
                            if len == NUM_WRITERS * SLICE_LEN * NUM_ITEMS {
                                break;
                            }
                        }
                    }
                });
            }
            for _ in 0..NUM_WRITERS {
                s.spawn(|| {
                    for j in 0..NUM_ITEMS {
                        let array: [Box<usize>; SLICE_LEN] = array::from_fn(|_| Box::new(j));
                        assert!(v.push_array(array).start >= SLICE_LEN * j);
                    }
                });
            }
        });
        assert_eq!(v.len(), NUM_WRITERS * SLICE_LEN * NUM_ITEMS);
    }

    #[test]
    fn test_get_unchecked_concurrent_reads() {
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
