extern crate criterion;
extern crate rand;
extern crate statrs;

use criterion::{criterion_group, criterion_main, Criterion};
use statrs::tree::kd_tree::KdTree;

fn generate_1d(n_samples: usize) -> Vec<f64> {
    (0..n_samples).map(|_| rand::random()).collect()
}
fn generate_3d(n_samples: usize) -> Vec<[f64; 3]> {
    (0..n_samples).map(|_| rand::random()).collect()
}
fn squared_l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
}

fn bench_kd_tree(c: &mut Criterion) {
    let points = generate_3d(1000);
    let mut group = c.benchmark_group("kd_tree");
    group.bench_function("build_1k_3d_points_tree", |b| {
        b.iter(|| {
            let _tree = KdTree::from(points.clone());
        });
    });
    let tree = KdTree::from(points.clone());
    group.bench_function("search_1point_150_neighbors", |b| {
        b.iter(|| {
            let _neighbors = tree.k_nearest_neighbors(&points[0], 150, |a, b| {
                squared_l2_distance(a.as_slice(), b.as_slice())
            });
        });
    });
}

fn bench_1d_density(c: &mut Criterion) {
    let samples = generate_1d(100_000);
    let mut group = c.benchmark_group("density");
    group.bench_function("knn_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf(0., samples.clone());
        });
    });
    group.bench_function("kde_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::kde_pdf_1d(0., samples.clone(), Default::default());
        })
    });
}

criterion_group!(benches, bench_kd_tree, bench_1d_density);

criterion_main!(benches);
