use crate::error::Error;
use crate::box_properties::BoxProperties;

use ndarray::{Array3,ArrayView3};
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

pub fn white_noise(bp: &BoxProperties) -> Array3<f64> {
    let n = bp.logical as usize;
    let mut result = Array3::zeros([n, n, n]);
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    for x in result.iter_mut() {
        *x = normal.sample(&mut rng);
    }
    result
}

