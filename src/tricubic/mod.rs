use crate::stencil;
use crate::numeric::{Vec3};

use ndarray::{ArrayView3, indices, Ix3};
use lru_cache::LruCache;

mod table;
use table::COEFFICIENTS;

pub trait CoefficientFn
{
    fn new() -> Self;
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64];
}

pub struct Tricubic<'a, C: CoefficientFn> {
    view: ArrayView3<'a,f64>,
    cache: LruCache<[usize;3], [f64;64]>,
    _c: C
}

#[inline]
fn offset(a: [usize;3], b: (usize, usize, usize), s: Ix3) -> [usize;3]
{
    [ (a[0] + b.0) % s[0], (a[1] + b.1) % s[1], (a[2] + b.2) % s[2] ]
}

struct SecondOrder;

impl CoefficientFn for SecondOrder
{
    fn new() -> Self { SecondOrder {} }
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64] {
        let s = a.raw_dim();
        let y = stencil::flat_2x2x2(a, i);

        let mut c = [0.0; 64];
        for (j, dx) in indices([2, 2, 2]).into_iter().enumerate() {
            let idx = offset(i, dx, s);
            c[j]    = y[j];
            c[8+j]  = derivative!(a, 2, idx);
            c[16+j] = derivative!(a, 1, idx);
            c[24+j] = derivative!(a, 0, idx);
            c[32+j] = derivative!(a, 1, 2, idx);
            c[40+j] = derivative!(a, 0, 1, idx);
            c[48+j] = derivative!(a, 0, 2, idx);
            c[56+j] = derivative!(a, 0, 1, 2, idx);
        }

        let mut result = [0.0; 64];
        for (i, j, m) in COEFFICIENTS {
            result[i] += c[j] * m;
        }
        result
    }
}

struct FirstOrder;

impl CoefficientFn for FirstOrder
{
    fn new() -> Self { Self {} }
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64] {
        let s = a.raw_dim();
        let mut c = [0.0; 64];

        for (j, dx) in indices([2, 2, 2]).into_iter().enumerate() {
            let idx = offset(i, dx, s);
            let y = stencil::flat_3x3x3(a, idx);

            c[j]    = y[13];
            c[ 8+j] = (y[14] - y[12]) / 2.;
            c[16+j] = (y[16] - y[10]) / 2.;
            c[24+j] = (y[22] - y[ 4]) / 2.;
            c[32+j] = (y[17] - y[15] - y[11] + y[ 9]) / 4.;
            c[40+j] = (y[25] - y[19] - y[ 7] + y[ 1]) / 4.;
            c[48+j] = (y[23] - y[21] - y[ 5] + y[ 3]) / 4.;
            c[56+j] = (y[26] - y[24] - y[20] + y[18]
                    -  y[ 8] + y[ 6] + y[ 2] - y[ 0]) / 8.;
        }
        let mut result = [0.0; 64];
        for (i, j, m) in COEFFICIENTS {
            result[i] += c[j] * m;
        }
        result
    }
}

impl<'a, C: CoefficientFn> Tricubic<'a, C> {
    fn new(view: ArrayView3<'a,f64>, n_cache: usize) -> Self {
        Self {
            view: view,
            cache: LruCache::new(n_cache),
            _c: C::new()
        }
    }

    fn interpolate(&mut self, x: Vec3) -> f64 {
        let i = x.0.map(|j| j.floor() as usize);
        let f = x.0.zip(i).map(|(j, k)| j - (k as f64));

        if !self.cache.contains_key(&i) {
            self.cache.insert(i, C::coefficients(&self.view, i));
        }

        let b = self.cache.get_mut(&i).unwrap();
        let mut result: f64 = 0.0;
        for (j, dx) in indices([4, 4, 4]).into_iter().enumerate() {
            result += b[j] * f[0].powi(dx.0 as i32)
                           * f[1].powi(dx.1 as i32)
                           * f[2].powi(dx.2 as i32);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grf::{GaussianRandomField};
    use crate::box_properties::{BoxProperties};

    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_tricubic() {
        let mut file = File::create("splinetest.dat").unwrap();
        let bp = BoxProperties{ physical: 1.0, logical: 8 };
        let mut f = bp.white_noise();
        bp.apply_power_spectrum(&mut f, |k| k.powf(-5.0)).unwrap();

        let mut t = Tricubic::<FirstOrder>::new(f.view(), 256);
        for i in indices([64,64,64]).into_iter() {
            let x = Vec3([i.0 as f64 / 8., i.1 as f64 / 8., i.2 as f64 / 8.]);
            write!(&mut file, "{} ", t.interpolate(x)).unwrap();
        }
        write!(&mut file, "\n").unwrap();

        let mut t = Tricubic::<SecondOrder>::new(f.view(), 256);
        for i in indices([64,64,64]).into_iter() {
            let x = Vec3([i.0 as f64 / 8., i.1 as f64 / 8., i.2 as f64 / 8.]);
            write!(&mut file, "{} ", t.interpolate(x)).unwrap();
        }
        write!(&mut file, "\n").unwrap();

        for i in f.iter() {
            write!(&mut file, "{} ", i).unwrap();
        }
    }
}
