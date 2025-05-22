use core::{cmp::Ordering, f64::consts::PI};
use num_traits::Zero;
use std::collections::BinaryHeap;

use crate::function::gamma::gamma;

use super::kd_tree::KdTree;

pub fn knn_density(k: usize, x: f64, samples: Vec<f64>) -> Option<f64> {
    let n_samples = samples.len() as f64;
    let d = 1.;
    return KdTree::from(samples)
        .k_nearest_neighbors(&x, k, |a, b| (a - b).abs())
        .map(|mut neighbors| {
            let radius = neighbors.pop().unwrap().dist;
            (k as f64 / n_samples) * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d)))
        });
}

/// Handles variable/point types for which nearest neighbors can be computed.
pub trait Container: Clone {
    type Elem;
    fn length(&self) -> usize;
    fn get(&self, index: usize) -> Self::Elem;
}
macro_rules! impl_container_for_num {
    ($($t:ty),*) => {
        $(
            impl Container for $t {
                type Elem = $t;
                fn length(&self) -> usize {
                    1
                }
                fn get(&self, _index: usize) -> Self::Elem {
                    *self
                }
            }
        )*
    };
}
impl_container_for_num!(f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

macro_rules! impl_container {
    ($($t:ty),*) => {
        $(
            impl<T: Copy> Container for $t {
                type Elem = T;
                fn length(&self) -> usize {
                    self.len()
                }
                fn get(&self, index: usize) -> Self::Elem {
                    self[index]
                }
            }
        )*
    };
}
impl_container!(
    [T; 1],
    [T; 2],
    [T; 3],
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

#[cfg(test)]
mod tests {
    use crate::{
        distribution::{Continuous, Normal},
        statistics::Statistics,
    };
    use rand::distributions::Distribution;

    use super::*;

    #[test]
    fn test_knn_density() {
        let law = Normal::new(1., 1.).unwrap();
        let mut rng = rand::thread_rng();
        let samples = (0..10000).map(|_| law.sample(&mut rng)).collect::<Vec<_>>();
        let x = 0.001;
        let density = knn_density(1000, x, samples.clone());
        println!("Density: {:?}", density.unwrap());
        println!("Mean: {:?}", samples.clone().mean());
        println!("Std: {:?}", samples.clone().std_dev());
        println!("Pdf: {:?}", law.pdf(x));
    }
}
