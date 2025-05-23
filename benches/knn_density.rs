#![feature(test)]
extern crate rand;
extern crate statrs;
extern crate test;

use rand::distributions::Distribution;
use test::Bencher;

fn generate_gaussian_1d(n_samples: usize) -> Vec<f64> {
    let law = statrs::distribution::Normal::new(0., 1.).unwrap();
    let mut rng = rand::thread_rng();
    (0..n_samples)
        .map(|_| law.sample(&mut rng))
        .collect::<Vec<_>>()
}

#[bench]
fn bench_knn_pdf_1d(bench: &mut Bencher) {
    let samples = generate_gaussian_1d(100_000);
    bench.iter(|| {
        let _f = statrs::density::knn::knn_pdf(0., samples.clone());
    })
}

#[bench]
fn bench_kde_pdf_1d(bench: &mut Bencher) {
    let samples = generate_gaussian_1d(100_000);
    bench.iter(|| {
        let _f = statrs::density::knn::kde_pdf(
            0.,
            samples.clone(),
            statrs::density::kde::Kernel1d::Epanechnikov,
        );
    })
}
