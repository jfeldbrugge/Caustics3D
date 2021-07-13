use crate::error::Error;
use crate::box_properties::BoxProperties;

use ndarray::{Array3,ArrayBase,Ix3,DataMut,RawData};
// use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

pub trait GaussianRandomField {
    fn white_noise(&self) -> Array3<f64>;
    fn apply_power_spectrum<D, F>(&self, f: &mut ArrayBase<D, Ix3>, power_spectrum: F) -> Result<(), Error>
        where D: DataMut + RawData<Elem=f64>,
              F: Fn(f64) -> f64;
}

impl GaussianRandomField for BoxProperties {
    fn white_noise(&self) -> Array3<f64> {
        let n = self.logical as usize;
        let mut result = Array3::zeros([n, n, n]);
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        for x in result.iter_mut() {
            *x = normal.sample(&mut rng);
        }
        result
    }

    fn apply_power_spectrum<D, F>(&self, f: &mut ArrayBase<D, Ix3>, power_spectrum: F) -> Result<(), Error>
        where D: DataMut + RawData<Elem=f64>,
              F: Fn(f64) -> f64
    {
        let n = self.logical as usize;

        let mut fft: R2CPlan64 = R2CPlan::aligned(&[n, n, n], Flag::ESTIMATE)?;
        let mut ifft: C2RPlan64 = C2RPlan::aligned(&[n, n, n], Flag::ESTIMATE)?;

        let mut real_buffer = AlignedVec::<f64>::new(n.pow(3));
        let mut freq_buffer = AlignedVec::<c64>::new(n.pow(2) * (n / 2 + 1));
        
        let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
        f.assign_to(real_view);
        
        fft.r2c(&mut real_buffer, &mut freq_buffer)?;
        let mut freq_space = ndarray::ArrayViewMut::from_shape([n, n, n / 2 + 1], &mut freq_buffer)?;

        for (idx, v) in freq_space.indexed_iter_mut() {
            let k_sqr = self.freq_sqr(idx);
            *v *= power_spectrum(k_sqr.sqrt()).sqrt();
        }
        freq_space[[0,0,0]] = c64::new(0.0, 0.0);
        ifft.c2r(&mut freq_buffer, &mut real_buffer)?;

        let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
        real_view.assign_to(f);
        Ok(())
    }
}
