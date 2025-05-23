use crate::density::knn::{Container, KNearestNeighbors, KthNearestNeighbor, Order};
use core::ops::Sub;
use num_traits::Zero;
use std::collections::BinaryHeap;

#[derive(Clone, Debug, PartialEq)]
struct Node<K> {
    key: K,
    coordinate: Option<usize>,
    left: Option<Box<Node<K>>>,
    right: Option<Box<Node<K>>>,
}

impl<K> Node<K> {
    pub fn new(key: K, coordinate: Option<usize>) -> Self {
        Self {
            key,
            coordinate,
            left: None,
            right: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KdTree<K> {
    root: Option<Box<Node<usize>>>,
    len: usize,
    data: Option<Vec<K>>,
}

impl<K> KdTree<K> {
    /// Creates an empty tree instance.
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let tree = KdTree::<Vec<f64>>::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            root: None,
            len: 0,
            data: None,
        }
    }

    /// Gives the number of keys in the tree.
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let mut tree = KdTree::<Vec<f32>>::from([vec![1.0, 2.0], vec![1.0, -1.0]]);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let tree = KdTree::<Vec<f32>>::from([vec![0.0, 0.0]]);
    /// assert!(!tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the data in the tree
    pub fn data(&self) -> Option<&Vec<K>> {
        self.data.as_ref()
    }
}

impl<K: Container> KdTree<K>
where
    K::Elem: Order,
{
    /// Inserts `keys` in the tree.
    ///
    /// The caller must make sure that the `keys` does not have duplicate
    /// elements to avoid errors when retrieving nearest neighbors.
    ///
    /// This method is useful for offline tree construction, when all the
    /// points to insert are available at the same time. Building the
    /// tree with this mehod guarantees an (almost) balanced tree, so the caller
    /// may expect faster nearest neighbors retrieval, at the cost of
    /// slower tree construction.
    ///
    /// The caller must make sure that the `keys` have the same dimension.
    ///
    /// Adapted from the paper: [Parallel k Nearest Neighbor Graph Construction
    /// Using Tree-Based Data Structures][paper].
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// use nalgebra::Vector2;
    /// let a = Vector2::new(5., 4.);
    /// let b = Vector2::new(2., 6.);
    /// let c = Vector2::new(13., 3.);
    /// let d = Vector2::new(3., 1.);
    /// let e = Vector2::new(10., 2.);
    /// let f = Vector2::new(8., 7.);
    /// let mut bt = KdTree::from([a, b, c, d, e, f]);
    /// assert_eq!(bt.len(), 6);
    /// ```
    ///
    /// [paper]: http://dx.doi.org/10.5821/hpgm15.1
    pub fn from<Keys>(keys: Keys) -> Self
    where
        Keys: IntoIterator<Item = K>,
        K::Elem: Sub<Output = K::Elem>,
    {
        let mut keys = keys.into_iter().collect::<Vec<_>>();
        if keys.is_empty() {
            return Self::new();
        }
        let dimension = keys[0].length();
        let len = keys.len();
        let dim_max_spread = find_dim_max_spread(&keys, 0, len, dimension);
        let median = len / 2;
        keys[0..len].sort_by(|a, b| a.get(dim_max_spread).total_cmp(&b.get(dim_max_spread)));
        let mut root = Some(Box::new(Node::new(median, Some(dim_max_spread))));
        build_tree(&mut root, Children::Left, &mut keys, 0, median, dimension);
        build_tree(
            &mut root,
            Children::Right,
            &mut keys,
            median + 1,
            len,
            dimension,
        );
        Self {
            root,
            len,
            data: Some(keys),
        }
    }

    /// Searches the nearest neighbors of a `key` in the tree.
    ///
    /// The distance metric is provided by the caller.
    ///
    /// It returns a max oriented binary heap collecting the k nearest neighbors
    /// of `key` located in the tree. It means that the top element of the heap
    /// (whose reference is accessible in O(1) running time) is the furthest
    /// from `key`.
    ///
    /// Adapted from the paper: [Parallel k Nearest Neighbor Graph Construction
    /// Using Tree-Based Data Structures][paper].
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// use nalgebra::Vector2;
    /// let a = Vector2::new(5., 4.);
    /// let b = Vector2::new(2., 6.);
    /// let c = Vector2::new(13., 3.);
    /// let d = Vector2::new(3., 1.);
    /// let e = Vector2::new(10., 2.);
    /// let f = Vector2::new(8., 7.);
    /// let mut bt = KdTree::from([a, b, c, d, e, f]);
    /// let mut knn = bt
    ///     .k_nearest_neighbors(&Vector2::new(9., 4.), 4, |pa, pb| (pa - pb).norm())
    ///     .unwrap();
    /// assert_eq!(c, knn.pop().unwrap().point);
    /// assert_eq!(a, knn.pop().unwrap().point);
    /// assert_eq!(f, knn.pop().unwrap().point);
    /// assert_eq!(e, knn.pop().unwrap().point);
    /// ```
    ///
    /// [paper]: http://dx.doi.org/10.5821/hpgm15.1
    pub fn k_nearest_neighbors<D>(
        &self,
        key: &K,
        k: usize,
        distance: D,
    ) -> Option<KNearestNeighbors<K, K::Elem>>
    where
        K: PartialEq,
        K::Elem: Sub<Output = K::Elem> + Order,
        D: Fn(&K, &K) -> K::Elem,
    {
        if self.root.is_none() | (k == 0) | self.data.is_none() {
            return None;
        }
        let heap = BinaryHeap::with_capacity(k + 1);
        let data = self.data.as_ref().unwrap();
        Some(
            k_nearest_neighbors(&self.root, data, key, heap, k, &distance)
                .iter()
                .map(|neighbor| KthNearestNeighbor {
                    point: data[neighbor.point].clone(),
                    dist: neighbor.dist,
                })
                .collect(),
        )
    }
}

fn k_nearest_neighbors<K, D>(
    node: &Option<Box<Node<usize>>>,
    data: &[K],
    key: &K,
    mut the_bests: KNearestNeighbors<usize, K::Elem>,
    k: usize,
    distance: &D,
) -> KNearestNeighbors<usize, K::Elem>
where
    K: Container + PartialEq,
    K::Elem: Order + Sub<Output = K::Elem>,
    D: Fn(&K, &K) -> K::Elem,
{
    if let Some(nod) = node {
        let dist = distance(&data[nod.key], key);
        if the_bests.len() < k {
            the_bests.push(KthNearestNeighbor {
                point: nod.key,
                dist,
            });
        } else if dist < the_bests.peek().unwrap().dist {
            // .unwrap() is safe here as long as k >= 1 because when k >= 1 the heap is not
            // empty, which guaranties the existence of a maximum.
            the_bests.pop();
            the_bests.push(KthNearestNeighbor {
                point: nod.key,
                dist,
            });
        }
        if let Some(coordinate) = nod.coordinate {
            let (node_val, key_val) = (data[nod.key].get(coordinate), key.get(coordinate));
            let (next, other, dist) = if key.get(coordinate) < data[nod.key].get(coordinate) {
                (&nod.left, &nod.right, node_val - key_val)
            } else {
                (&nod.right, &nod.left, key_val - node_val)
            };
            the_bests = k_nearest_neighbors(next, data, key, the_bests, k, distance);
            if (dist <= the_bests.peek().unwrap().dist) | (the_bests.len() < k) {
                the_bests = k_nearest_neighbors(other, data, key, the_bests, k, distance);
            }
        }
    }
    the_bests
}

#[derive(Debug)]
enum Children {
    Left,
    Right,
}
fn build_tree<K>(
    node: &mut Option<Box<Node<usize>>>,
    children: Children,
    keys: &mut Vec<K>,
    start: usize,
    end: usize,
    dimension: usize,
) where
    K: Container,
    K::Elem: Order + Sub<Output = K::Elem>,
{
    if start >= end {
        return;
    }
    if end == start + 1 {
        let new_node = Some(Box::new(Node::new(start, None)));
        match children {
            Children::Left => {
                if let Some(ref mut nod) = node {
                    nod.left = new_node;
                }
            }
            Children::Right => {
                if let Some(ref mut nod) = node {
                    nod.right = new_node;
                }
            }
        };
        return;
    }
    let dim_max_spread = find_dim_max_spread(&*keys, start, end, dimension);
    keys[start..end].sort_by(|a, b| a.get(dim_max_spread).total_cmp(&b.get(dim_max_spread)));
    let median = start + (end - start) / 2;
    match children {
        Children::Left => {
            if let Some(ref mut nod) = node {
                nod.left = Some(Box::new(Node::new(median, Some(dim_max_spread))));
                build_tree(
                    &mut nod.left,
                    Children::Left,
                    keys,
                    start,
                    median,
                    dimension,
                );
                build_tree(
                    &mut nod.left,
                    Children::Right,
                    keys,
                    median + 1,
                    end,
                    dimension,
                );
            }
        }
        Children::Right => {
            if let Some(ref mut nod) = node {
                nod.right = Some(Box::new(Node::new(median, Some(dim_max_spread))));
                build_tree(
                    &mut nod.right,
                    Children::Left,
                    keys,
                    start,
                    median,
                    dimension,
                );
                build_tree(
                    &mut nod.right,
                    Children::Right,
                    keys,
                    median + 1,
                    end,
                    dimension,
                );
            }
        }
    };
}

fn find_dim_max_spread<K>(keys: &[K], start: usize, end: usize, dimension: usize) -> usize
where
    K: Container,
    K::Elem: Order + Sub<Output = K::Elem>,
{
    let (mut dim_max_spread, mut max_spread) = (0, K::Elem::zero());
    for dim in 0..dimension {
        let (mut min, mut max) = (keys[start].get(dim), keys[start].get(dim));
        for key in keys.iter().take(end).skip(start + 1) {
            let val = key.get(dim);
            if val > max {
                max = val;
            }
            if val < min {
                min = val;
            }
        }
        let spread = max - min;
        if spread > max_spread {
            dim_max_spread = dim;
            max_spread = spread;
        }
    }
    dim_max_spread
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector1;

    use crate::density::kd_tree::KdTree;

    #[test]
    fn test_kdtree() {
        let keys = [
            Vector1::new(-1.),
            Vector1::new(3.),
            Vector1::new(5.),
            Vector1::new(5.),
            Vector1::new(7.),
            Vector1::new(9.),
        ];
        let tree = KdTree::from(keys);
        assert!(!tree.is_empty());
        let knn = tree
            .k_nearest_neighbors(&Vector1::new(6.5f64), 5, |a, b| (a - b).norm())
            .unwrap()
            .into_sorted_vec();
        assert_eq!(knn[0].point, Vector1::new(7.));
        assert_eq!(knn[1].point, Vector1::new(5.));
        assert_eq!(knn[2].point, Vector1::new(5.));
        assert_eq!(knn[3].point, Vector1::new(9.));
    }
}
