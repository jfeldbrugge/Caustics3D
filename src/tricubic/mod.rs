use crate::stencil;
use crate::numeric::{Vec3};

use ndarray::{ArrayBase,ArrayView3, Ix3, RawData, Data};
use lru_cache::LruCache;

mod table;
use table::COEFFICIENTS;

use std::cell::RefCell;

pub trait CoefficientFn
{
    fn new() -> Self;
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64];
}

pub struct Tricubic<S: Data + RawData<Elem=f64>, C: CoefficientFn> {
    pub array: ArrayBase<S,Ix3>,
    cache: RefCell<LruCache<[usize;3], [f64;64]>>,
    _c: C
}

#[inline]
fn offset(a: [usize;3], b: (usize, usize, usize), s: Ix3) -> [usize;3]
{
    [ (a[0] + b.0) % s[0], (a[1] + b.1) % s[1], (a[2] + b.2) % s[2] ]
}

pub struct SecondOrder;

impl CoefficientFn for SecondOrder
{
    fn new() -> Self { SecondOrder {} }
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64] {
        let s = a.raw_dim();
        let y = stencil::flat_2x2x2(a, i);

        let mut c = [0.0; 64];
        for j in 0..8 {
            let dx = ((j & 4) >> 2, (j & 2) >> 1, j & 1);
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

pub struct FirstOrder;

impl CoefficientFn for FirstOrder
{
    fn new() -> Self { Self {} }
    fn coefficients(a: &ArrayView3<f64>, i: [usize;3]) -> [f64;64] {
        let s = a.raw_dim();
        let mut c = [0.0; 64];

        // for (j, dx) in indices([2, 2, 2]).into_iter().enumerate() {
        for j in 0..8 {
            let dx = ((j & 4) >> 2, (j & 2) >> 1, j & 1);
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

impl<S: Data + RawData<Elem=f64>, C: CoefficientFn> Tricubic<S, C> {
    pub fn new(array: ArrayBase<S,Ix3>, n_cache: usize) -> Self {
        Self {
            array: array,
            cache: RefCell::new(LruCache::new(n_cache)),
            _c: C::new()
        }
    }

    pub fn f(&self, x: Vec3) -> f64 {
        let mut cache = self.cache.borrow_mut();
        let i = x.0.map(|j| j.floor() as usize);
        let f = x.0.zip(i).map(|(j, k)| j - (k as f64));

        if !cache.contains_key(&i) {
            cache.insert(i, C::coefficients(&self.array.view(), i));
        }

        let b = cache.get_mut(&i).unwrap();
        let mut result: f64 = 0.0;
        let mut j = 0;
        for j in 0..64 {
            let w = j & 3;
            let v = (j >> 2) & 3;
            let u = (j >> 4) & 3;
            result += b[j] * f[0].powi(u as i32)
                           * f[1].powi(v as i32)
                           * f[2].powi(w as i32);
        }
        result
    }

    pub fn df(&self, k: usize, x: Vec3) -> f64 {
        let mut cache = self.cache.borrow_mut();
        let i = x.0.map(|j| j.floor() as usize);
        let f = x.0.zip(i).map(|(j, k)| j - (k as f64));

        if !cache.contains_key(&i) {
            cache.insert(i, C::coefficients(&self.array.view(), i));
        }

        let b = cache.get_mut(&i).unwrap();
        let mut result: f64 = 0.0;
        let i = (k + 1) % 3;
        let j = (k + 2) % 3;
        for j in 0..64 {
            let dx = [(j >> 4) & 3, (j >> 2) & 3, j & 3];
            if dx[k] == 0 { continue; }
            result += b[j] * (dx[k] as f64)
                   * f[dx[i]].powi(dx[i] as i32)
                   * f[dx[j]].powi(dx[j] as i32)
                   * f[dx[k]].powi(dx[k] as i32 - 1);
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
    use ndarray::{ViewRepr, indices};

    #[test]
    fn test_tricubic() {
        let mut file = File::create("splinetest.dat").unwrap();
        let bp = BoxProperties{ physical: 1.0, logical: 8 };
        let mut f = bp.white_noise();
        bp.apply_power_spectrum(&mut f, |k| k.powf(-5.0)).unwrap();

        let t = Tricubic::<ViewRepr<&f64>, FirstOrder>::new(f.view(), 256);
        for i in indices([64,64,64]).into_iter() {
            let x = Vec3([i.0 as f64 / 8., i.1 as f64 / 8., i.2 as f64 / 8.]);
            write!(&mut file, "{} ", t.f(x)).unwrap();
        }
        write!(&mut file, "\n").unwrap();

        let t = Tricubic::<ViewRepr<&f64>, SecondOrder>::new(f.view(), 256);
        for i in indices([64,64,64]).into_iter() {
            let x = Vec3([i.0 as f64 / 8., i.1 as f64 / 8., i.2 as f64 / 8.]);
            write!(&mut file, "{} ", t.f(x)).unwrap();
        }
        write!(&mut file, "\n").unwrap();

        for i in f.iter() {
            write!(&mut file, "{} ", i).unwrap();
        }
    }
}
