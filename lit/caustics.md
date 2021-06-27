# Caustics

``` {.rust file=src/caustics.rs}
use crate::error::{Error};
use crate::box_properties::{BoxProperties};
use crate::numeric::{Vec3, Sym3, tuple3_idx};

use clap::{ArgMatches};
use ndarray::{Array3, Ix3, indices};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

use std::str::FromStr;

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
