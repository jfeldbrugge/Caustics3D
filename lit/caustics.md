# Caustics

``` {.rust file=src/caustics.rs}
use crate::error::{Error};
use crate::box_properties::{BoxProperties, tuple3_idx};

use clap::{ArgMatches};
use ndarray::{Array3, Ix3, indices};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

use std::str::FromStr;
use std::f64::consts::PI;

fn read_box_properties(file: &hdf5::File) -> Result<BoxProperties, Error> {
    let pars = file.group("parameters")?;
    let logical: u64 = pars.attr("grid-size")?.read_scalar()?;
    let physical: f64 = pars.attr("box-size")?.read_scalar()?;

    Ok(BoxProperties {
        logical: logical,
        physical: physical
    })
}

fn compute_hessian(file: &hdf5::File, target: &hdf5::Group,  scale: Option<f64>) -> Result<(), Error> {
    let ics = file.group("ics")?;
    let pot_ds = ics.dataset("potential")?;

    let bp = read_box_properties(file)?;
    let n: usize = bp.logical as usize;
    let size: f64 = n.pow(3) as f64;

    let mut real_buffer = AlignedVec::<f64>::new(n.pow(3));
    let mut pot_f_buffer = AlignedVec::<c64>::new(n.pow(2) * (n / 2 + 1));
    let mut hessian_f_buffer = AlignedVec::<c64>::new(n.pow(2) * (n / 2 + 1));

    let mut fft: R2CPlan64 = R2CPlan::aligned(&[n, n, n], Flag::ESTIMATE)?;
    let mut ifft: C2RPlan64 = C2RPlan::aligned(&[n, n, n], Flag::ESTIMATE)?;

    let pot = pot_ds.read::<f64, Ix3>()?;
    let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
    pot.assign_to(real_view);

    fft.r2c(&mut real_buffer, &mut pot_f_buffer)?;
    let mut pot_f = ndarray::ArrayViewMut::from_shape([n, n, n / 2 + 1], &mut pot_f_buffer)?;

    if let Some(s) = scale {
        for (idx, v) in pot_f.indexed_iter_mut() {
            let k_sqr = bp.freq_sqr(idx);
            *v *= (- s * s * k_sqr / 2.0).exp();
        }
    }

    for i in 0..3 {
        for j in 0..3 {
            if j < i { continue; }
            let mut hessian_f = ndarray::ArrayViewMut::from_shape([n, n, n / 2 + 1], &mut hessian_f_buffer)?;
            for (idx, v) in pot_f.indexed_iter() {
                let ki = bp.freq(tuple3_idx(idx, i));
                let kj = bp.freq(tuple3_idx(idx, j));
                hessian_f[idx] = v * c64::new(-ki * kj / size, 0.0);
            }
            ifft.c2r(&mut hessian_f_buffer, &mut real_buffer)?;
            let name = format!("H{}{}", i, j);
            let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
            target.new_dataset::<f64>().shape([n,n,n]).create(name.as_str())?.write(real_view.view())?;
        }
    }

    Ok(())
}

#[derive(Clone,Debug)]
struct Vec3 ([f64; 3]);
#[derive(Clone,Debug)]
struct Sym3 ([f64; 6]);

impl std::ops::Index<(u8, u8)> for Sym3 {
    type Output = f64;
    fn index(&self, idx: (u8, u8)) -> &Self::Output {
        let (i, j) = idx;
        if i > j {
            self.index((j, i))
        } else {
            let k: usize = (((7 - i) * i) / 2 + j) as usize;
            let Sym3(d) = self;
            &d[k]
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Self) -> Self {
        let Vec3(a) = self;
        let Vec3(b) = other;
        let mut c = [0.0; 3];
        for i in 0..3 { c[i] = a[i] + b[i]; }
        Vec3(c)
    }
}

impl num_traits::identities::Zero for Vec3 {
    fn zero() -> Self {
        Vec3([0.0, 0.0, 0.0])
    }

    fn is_zero(&self) -> bool {
        let Vec3(d) = self;
        d.iter().all(|x| x.is_zero())
    }
}

unsafe impl hdf5::H5Type for Vec3 {
    fn type_descriptor() -> hdf5::types::TypeDescriptor {
        use hdf5::types::{TypeDescriptor,FloatSize};
        TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Float(FloatSize::U8)), 3)
    }
}

fn tuple3_sort(x: (f64, f64, f64)) -> (f64, f64, f64) {
    let (a, b, c) = x;

    if a > b {
        if b > c {
            (a, b, c)
        } else if c > a {
            (c, a, b)
        } else {
            (a, c, b)
        }
    } else {
        if a > c {
            (b, a, c)
        } else if c > b {
            (c, b, a)
        } else {
            (b, c, a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple3_sort() {
        assert_eq!(tuple3_sort((1., 2., 3.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((1., 3., 2.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((2., 1., 3.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((2., 3., 1.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((3., 1., 2.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((3., 2., 1.)), (3., 2., 1.));
    }
}

impl Sym3 {
    fn trace(&self) -> f64 {
        let Sym3(d) = self;
        d[0] + d[3] + d[5]
    }

    fn square_trace(&self) -> f64 {
        let Sym3(d) = self;
        d[0]*d[0] + 2.*d[1]*d[1] + 2.*d[2]*d[2]
                  +    d[3]*d[3] + 2.*d[4]*d[4]
                                 +    d[5]*d[5]
    }

    fn eigenvalues(&self) -> (f64, f64, f64) {
        let Sym3(d) = self;
        let q = self.trace() / 3.;
        let mut f: [f64; 6] = d.clone();
        f[0] -= q; f[3] -= q; f[5] -=q;

        let p = (self.square_trace() / 6.).sqrt();
        for i in 0..6 { f[i] /= p; }
        let phi = (Sym3(f).trace() / 2.).acos() / 3.;

        let a = q + 2. * p * phi.cos();
        let b = q + 2. * p * (phi + 2. * PI / 3.).cos();
        let c = q + 2. * p * (phi + 4. * PI / 3.).cos();

        tuple3_sort((a, b, c))
    }

    fn eigenvector(&self, l: f64) -> Vec3 {
        let Sym3(d) = self;
        Vec3([ (l - d[3]) * d[2] + d[4] * d[1]
             , (l - d[0]) * d[4] + d[3] * d[1]
             , (l - d[0]) * (l - d[1]) - d[0] * d[0] ])
    }
}

pub fn compute_eigenvalues(file: &hdf5::File, target: &hdf5::Group) -> Result<(), Error> {
    let bp = read_box_properties(file)?;
    let n: usize = bp.logical as usize;

    let h00 = target.dataset("H00")?.read::<f64,Ix3>()?;
    let h01 = target.dataset("H01")?.read::<f64,Ix3>()?;
    let h02 = target.dataset("H02")?.read::<f64,Ix3>()?;
    let h11 = target.dataset("H11")?.read::<f64,Ix3>()?;
    let h12 = target.dataset("H12")?.read::<f64,Ix3>()?;
    let h22 = target.dataset("H22")?.read::<f64,Ix3>()?;

    let mut lambda = (
        Array3::<f64>::zeros([n, n, n]),
        Array3::<f64>::zeros([n, n, n]),
        Array3::<f64>::zeros([n, n, n]));

    for idx in indices([n, n, n]) {
        let h = Sym3([h00[idx], h01[idx], h02[idx], h11[idx], h12[idx], h22[idx]]);
        let (a, b, c) = h.eigenvalues();
        lambda.0[idx] = a;
        lambda.1[idx] = b;
        lambda.2[idx] = c;
    }

    let mut ev = Array3::<Vec3>::zeros([n, n, n]);
    for k in 0..3 {
        let lambda = match k {
            0 => &lambda.0, 1 => &lambda.1, 2 => &lambda.2,
            _ => panic!("system error") };

        for idx in indices([n, n, n]) {
            let h = Sym3([h00[idx], h01[idx], h02[idx], h11[idx], h12[idx], h22[idx]]);
            ev[idx] = h.eigenvector(lambda[idx]);
        }

        let name = format!("lambda{}", k);
        let group = target.create_group(name.as_str())?;
        group.new_dataset::<f64>().shape([n, n, n]).create("eigenvalue")?.write(lambda)?;
        group.new_dataset::<Vec3>().shape([n, n, n]).create("eigenvector")?.write(ev.view())?;
    }

    Ok(())
}


pub fn run_caustics(args: &ArgMatches) -> Result<(), Error> {
    let filename = args.value_of("file").unwrap();
    let file = hdf5::File::open_rw(filename)?;
    let scale = args.value_of("scale").map(|s| f64::from_str(s).unwrap());

    let target_name = args.value_of("name").unwrap_or("caustics");
    let target = file.create_group(target_name)?;
    target.new_attr::<f64>().create("scale")?.write_scalar(&scale.unwrap_or(0.0))?;

    compute_hessian(&file, &target, scale)?;
    compute_eigenvalues(&file, &target)?;

    Ok(())
}
```
