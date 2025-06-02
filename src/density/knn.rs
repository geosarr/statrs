use core::{
    cmp::Ordering,
    f64::consts::PI,
    iter::Sum,
    ops::{Div, Sub},
};
use num_traits::{NumOps, Zero};
use std::collections::BinaryHeap;

use crate::tree::kd_tree::KdTree;
use crate::{density::kde::Kernel, function::gamma::gamma};

fn orava_optimal_k(n_samples: f64) -> f64 {
    // Adapted from K-nearest neighbour kernel density estimation, the choice of optimal k; Jan Orava 2012
    (0.587 * n_samples.powf(4.0 / 5.0)).round().max(1.)
}

fn base_pdf<D, S, X>(
    x: &X,
    samples: S,
    distance: D,
) -> (f64, f64, KNearestNeighbors<usize, f64>, KdTree<X>)
where
    S: IntoIterator<Item = X> + Container,
    X: Container<Elem = f64> + PartialEq,
    D: Fn(&X, &X) -> X::Elem,
{
    let n_samples = samples.length() as f64;
    let k = orava_optimal_k(n_samples);
    let tree = KdTree::from(samples);
    let neighbors = tree.k_nearest_neighbors_raw(x, k as usize, distance);
    (n_samples, k, neighbors, tree)
}

/// Computes the k-nearest neighbor density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using Orava's formula.
///
/// Returns `None` when `samples` is empty.
pub fn knn_pdf<S, X>(x: X, samples: S) -> Option<f64>
where
    S: IntoIterator<Item = X> + Container,
    X: Container<Elem = f64> + PartialEq,
{
    let (n_samples, k, mut neighbors, tree) =
        base_pdf(&x, samples, |a, b| a.squared_l2_distance(b).sqrt());
    if neighbors.is_empty() {
        None
    } else {
        let radius = neighbors.pop().unwrap().dist;
        let d = tree.data()[0].length() as f64;
        Some((k / n_samples) * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d))))
    }
}

/// Computes the kernel density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using Orava's formula.
///
/// Returns `None` when `samples` is empty.
pub fn kde_pdf<S, X>(x: X, samples: S) -> Option<f64>
where
    S: IntoIterator<Item = X> + Container,
    X: Container<Elem = f64> + PartialEq + Div<X::Elem, Output = X>,
    for<'a> &'a X: Sub<&'a X, Output = X>,
{
    let (n_samples, _, mut neighbors, tree) =
        base_pdf(&x, samples, |a, b| a.squared_l2_distance(b).sqrt());
    if neighbors.is_empty() {
        None
    } else {
        let radius = neighbors.pop().unwrap().dist;
        // let dim = tree.data()[0].length() as i32;
        // let kernel = Kernel::Gaussian { sigma: 1., dim };
        let kernel = Kernel::Epanechnikov;
        Some(
            (1. / (n_samples * radius))
                * tree
                    .data()
                    .iter()
                    .map(|xi| kernel.evaluate(x.squared_l2_distance(xi).sqrt() / radius))
                    .sum::<f64>(),
        )
    }
}

/// Computes the kernel density estimate for a given one dimensional point `x`
/// using the samples provided and a specified kernel.
///
/// The optimal `k` is computed using Orava's formula.
///
/// Returns `None` when `samples` is empty.
pub fn kde_pdf_1d<S, X>(x: X, samples: S, kernel: Kernel) -> Option<f64>
where
    S: IntoIterator<Item = X> + Container,
    X: Container<Elem = f64> + PartialEq + Div<X::Elem, Output = X::Elem>,
    for<'a> &'a X: Sub<&'a X, Output = X>,
{
    let (n_samples, _, mut neighbors, tree) =
        base_pdf(&x, samples, |a, b| a.squared_l2_distance(b));
    if neighbors.is_empty() {
        None
    } else {
        let radius = neighbors.pop().unwrap().dist.sqrt();
        Some(
            (1. / (n_samples * radius))
                * tree
                    .data()
                    .iter()
                    .map(|xi| kernel.evaluate((&x - xi) / radius))
                    .sum::<f64>(),
        )
    }
}

/// Handles variable/point types for which nearest neighbors can be computed.
pub trait Container: Clone {
    type Elem;
    fn length(&self) -> usize;
    fn get(&self, index: usize) -> Self::Elem;
    fn squared_l2_distance(&self, other: &Self) -> Self::Elem
    where
        Self::Elem: NumOps + Sum + Copy,
    {
        (0..self.length())
            .map(|i| {
                let elem = self.get(i);
                let other_elem = other.get(i);
                (elem - other_elem) * (elem - other_elem)
            })
            .sum::<Self::Elem>()
    }
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
                fn squared_l2_distance(&self, other: &Self) -> Self::Elem {
                    (self - other) * (self - other)
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
    use crate::distribution::Normal;
    use nalgebra::Vector2;
    use rand::distributions::Distribution;

    use super::*;

    #[test]
    fn test_knn_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = rand::thread_rng();
        let samples = (0..100000)
            .map(|_| Vector2::new(law.sample(&mut rng), law.sample(&mut rng)))
            .collect::<Vec<_>>();
        let x = Vector2::new(0., 0.);
        let knn_density = knn_pdf(x, samples.clone());
        let kde_density = kde_pdf(x, samples.clone()); // { sigma: 1. }
        println!("Density with kkn estimator: {:?}", knn_density.unwrap());
        println!("Density with kde estimator: {:?}", kde_density.unwrap());
        // println!("Pdf: {:?}", law.pdf(x));
    }
}
