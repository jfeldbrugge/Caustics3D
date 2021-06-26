// ~\~ language=Rust filename=src/caustics.rs
// ~\~ begin <<lit/caustics.md|src/caustics.rs>>[0]
use crate::error::{Error};
use crate::box_properties::{BoxProperties, tuple3_idx};

use clap::{ArgMatches};
use ndarray::{Ix3};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

use std::str::FromStr;
// use std::f64::consts::PI;

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
                hessian_f[idx] = v * c64::new(-ki * kj, 0.0);
            }
            ifft.c2r(&mut hessian_f_buffer, &mut real_buffer)?;
            let name = format!("H{}{}", i, j);
            let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
            target.new_dataset::<f64>().shape([n,n,n]).create(name.as_str())?.write(real_view.view())?;
        }
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
    Ok(())
}
// ~\~ end
