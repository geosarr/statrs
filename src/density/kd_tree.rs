use crate::density::knn::{Container, KNearestNeighbors, KthNearestNeighbor, Order};
use core::{
    cmp::Ordering,
    ops::{Index, Sub},
};
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
    root: Option<Box<Node<K>>>,
    len: usize,
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
        Self { root: None, len: 0 }
    }

    /// Gives the number of keys in the tree.
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let mut tree = KdTree::<Vec<f32>>::new();
    /// tree.insert(vec![1.0, 2.0]);
    /// tree.insert(vec![1.0, -1.0]);
    /// assert_eq!(tree.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether or not the tree is empty.
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let mut tree = KdTree::<Vec<f32>>::new();
    /// tree.insert(vec![0.0, 0.0]);
    /// assert!(!tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Container> KdTree<K>
where
    K::Elem: Order,
{
    fn put(
        node: &mut Option<Box<Node<K>>>,
        key: K,
        coordinate: usize,
    ) -> Option<&mut Box<Node<K>>> {
        match node {
            None => *node = Some(Box::new(Node::new(key, Some(coordinate)))),
            Some(ref mut nod) => match key[coordinate].total_cmp(&nod.key[coordinate]) {
                Ordering::Less => {
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.left, key, coordinate);
                }
                Ordering::Greater => {
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.right, key, coordinate);
                }
                Ordering::Equal => {
                    // Used to overwrite the current node's value, but doing so would change
                    // (possibly) the value of the current node's key, which changes the
                    // label/target of the predictor in the nearest neighbors algorithms.
                    // nod.value = value;
                    // return Some(nod);

                    // Possibility to put key in the left branch also ?
                    let coordinate = (nod.coordinate.unwrap() + 1) % key.length();
                    return Self::put(&mut nod.right, key, coordinate);
                }
            },
        }
        None
    }

    /// Inserts a `key` in the tree.
    ///
    /// The caller must make sure that the tree does not already contain `key`,
    /// to avoid duplicates and error when retrieving nearest neighbors. The caller
    /// must also make sure that the inserted `key`'s have the same dimension.
    ///
    /// This method is may be useful for online tree construction, when all the
    /// points to insert are not available at the same time. Building the
    /// tree with this mehod does not guarantee a balanced tree, so the
    /// caller should expect poor performance when retrieving the nearest
    /// neighbors, however inserting an element is fast on average.
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// let mut tree = KdTree::<Vec<i32>>::new();
    /// tree.insert(vec![1, 1]);
    /// tree.insert(vec![-3, 4]);
    /// tree.insert(vec![-3, 4]); // duplicate
    /// tree.insert(vec![5, -7]);
    /// assert_eq!(tree.len(), 4);
    /// ```
    pub fn insert(&mut self, key: K) {
        Self::put(&mut self.root, key, 0);
        self.len += 1;
    }
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
    /// Adapted from the paper: [Parallel k Nearest Neighbor Graph Construction
    /// Using Tree-Based Data Structures][paper].
    ///
    /// # Example
    /// ```
    /// use statrs::density::kd_tree::KdTree;
    /// use nalgebra::Vector2;
    /// let mut bt = KdTree::<_>::new();
    /// let a = Vector2::new(5., 4.);
    /// let b = Vector2::new(2., 6.);
    /// let c = Vector2::new(13., 3.);
    /// let d = Vector2::new(3., 1.);
    /// let e = Vector2::new(10., 2.);
    /// let f = Vector2::new(8., 7.);
    /// bt.insert(a.clone());
    /// bt.insert(b.clone());
    /// bt.insert(c.clone());
    /// bt.insert(d.clone());
    /// bt.insert(e.clone());
    /// bt.insert(f.clone());
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
    pub fn from<Keys>(keys: Keys) -> Option<Self>
    where
        Keys: IntoIterator<Item = K>,
        K::Elem: Sub<Output = K::Elem>,
    {
        let mut keys = keys.into_iter().collect::<Vec<_>>();
        if keys.is_empty() {
            return None;
        }
        let dimension = keys[0].length();
        let len = keys.len();
        let dim_max_spread = find_dim_max_spread(&keys, 0, len, dimension);
        let median = len / 2;
        keys[0..len].sort_by(|a, b| a[dim_max_spread].total_cmp(&b[dim_max_spread]));
        let root_key = keys[median].clone();
        let mut root = Some(Box::new(Node::new(root_key, Some(dim_max_spread))));
        build_tree(&mut root, Children::Left, &mut keys, 0, median, dimension);
        build_tree(
            &mut root,
            Children::Right,
            &mut keys,
            median + 1,
            len,
            dimension,
        );
        Some(Self { root, len })
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
    /// let mut bt = KdTree::<_>::new();
    /// let a = Vector2::new(5., 4.);
    /// let b = Vector2::new(2., 6.);
    /// let c = Vector2::new(13., 3.);
    /// let d = Vector2::new(3., 1.);
    /// let e = Vector2::new(10., 2.);
    /// let f = Vector2::new(8., 7.);
    /// bt.insert(a.clone());
    /// bt.insert(b.clone());
    /// bt.insert(c.clone());
    /// bt.insert(d.clone());
    /// bt.insert(e.clone());
    /// bt.insert(f.clone());
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
        if self.root.is_none() | (k == 0) {
            return None;
        }
        let heap = BinaryHeap::with_capacity(k + 1);
        Some(k_nearest_neighbors(&self.root, key, heap, k, &distance))
    }
}

fn k_nearest_neighbors<K, D>(
    node: &Option<Box<Node<K>>>,
    key: &K,
    mut the_bests: KNearestNeighbors<K, K::Elem>,
    k: usize,
    distance: &D,
) -> KNearestNeighbors<K, K::Elem>
where
    K: Container + PartialEq,
    K::Elem: Order + Sub<Output = K::Elem>,
    D: Fn(&K, &K) -> K::Elem,
{
    if let Some(nod) = node {
        let dist = distance(&nod.key, key);
        if the_bests.len() < k {
            the_bests.push(KthNearestNeighbor {
                point: nod.key.clone(),
                dist,
            });
        } else if dist < the_bests.peek().unwrap().dist {
            // .unwrap() is safe here as long as k >= 1 because when k >= 1 the heap is not
            // empty, which guaranties the existence of a maximum.
            the_bests.pop();
            the_bests.push(KthNearestNeighbor {
                point: nod.key.clone(),
                dist,
            });
        }
        if let Some(coordinate) = nod.coordinate {
            let (next, other, dist) = if key[coordinate] < nod.key[coordinate] {
                (&nod.left, &nod.right, nod.key[coordinate] - key[coordinate])
            } else {
                (&nod.right, &nod.left, key[coordinate] - nod.key[coordinate])
            };
            the_bests = k_nearest_neighbors(next, key, the_bests, k, distance);
            if (dist <= the_bests.peek().unwrap().dist) | (the_bests.len() < k) {
                the_bests = k_nearest_neighbors(other, key, the_bests, k, distance);
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
    node: &mut Option<Box<Node<K>>>,
    children: Children,
    keys: &mut Vec<K>,
    start: usize,
    end: usize,
    dimension: usize,
) where
    K: Clone + Index<usize>,
    K::Output: Order + Sub<K::Output, Output = K::Output>,
{
    if start >= end {
        return;
    }
    if end == start + 1 {
        let key = &keys[start];
        let new_node = Some(Box::new(Node::new(key.clone(), None)));
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
    keys[start..end].sort_by(|a, b| a[dim_max_spread].total_cmp(&b[dim_max_spread]));
    let median = start + (end - start) / 2;
    match children {
        Children::Left => {
            if let Some(ref mut nod) = node {
                nod.left = Some(Box::new(Node::new(
                    keys[median].clone(),
                    Some(dim_max_spread),
                )));
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
                nod.right = Some(Box::new(Node::new(
                    keys[median].clone(),
                    Some(dim_max_spread),
                )));
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
    K: Index<usize>,
    K::Output: Order + Sub<Output = K::Output>,
{
    let (mut dim_max_spread, mut max_spread) = (0, <K::Output as Zero>::zero());
    for dim in 0..dimension {
        let (mut min, mut max) = (keys[start][dim], keys[start][dim]);
        for key in keys.iter().take(end).skip(start + 1) {
            let val = key[dim];
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
        let tree = KdTree::from(keys).unwrap();
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
