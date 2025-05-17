use core::{cmp::Ordering, ops::Index};
use num_traits::Zero;
use std::collections::BinaryHeap;

/// Handles variables/points type for which nearest neighbors can be computed.
pub trait Container: Index<usize, Output = Self::Elem> + Clone {
    type Elem;
    fn length(&self) -> usize;
}
macro_rules! impl_container {
    ($($t:ty),*) => {
        $(
            impl<T: Clone> Container for $t {
                type Elem = T;
                fn length(&self) -> usize {
                    self.len()
                }
            }
        )*
    };
}
impl_container!(
    Vec<T>,
    nalgebra::Vector1<T>,
    nalgebra::Vector2<T>,
    nalgebra::Vector3<T>,
    nalgebra::Vector4<T>,
    nalgebra::Vector5<T>,
    nalgebra::Vector6<T>
);

/// Type alias for the set of k nearest neighbors of a point.
pub type KNearestNeighbors<X, T> = BinaryHeap<KthNearestNeighbor<X, T>>;

/// Represents a nearest neighbor point
#[derive(Debug, PartialEq, Clone)]
pub struct KthNearestNeighbor<P, D> {
    /// Id/value of this point.
    pub point: P,
    /// Distance from a target point.
    pub dist: D,
}
impl<P: PartialEq, D: Order> PartialOrd for KthNearestNeighbor<P, D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: PartialEq, D: Order> Ord for KthNearestNeighbor<P, D> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}
impl<P: PartialEq, D: PartialEq> Eq for KthNearestNeighbor<P, D> {}

/// Handles ordering of float-pointing and integer like types.
pub trait Order: PartialOrd + Copy + Zero {
    fn total_cmp(&self, other: &Self) -> Ordering;
}

macro_rules! impl_int_order {
    ($($t:ty),*) => {
        $(
            impl Order for $t {
                fn total_cmp(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}
impl_int_order!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

macro_rules! impl_float_order {
    ($($t:ty),*) => {
        $(
            impl Order for $t {
                fn total_cmp(&self, other: &Self) -> Ordering {
                    self.total_cmp(other)
                }
            }
        )*
    };
}
impl_float_order!(f32, f64);
