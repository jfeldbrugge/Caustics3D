# Caustics

``` {.rust file=src/caustics/a3.rs}
use crate::stencil;
use crate::numeric::{Vec3};
use crate::marching_tetrahedra;

use ndarray::{arr1, ArrayView3, Ix3, indices, IntoDimension};

const FIR: [f64;5] = [1./12., -2./3., 0., 2./3., -1./12.];

pub fn discrete_gradient<D>(f: &ArrayView3<f64>, x: D) -> Vec3
    where D: IntoDimension<Dim=Ix3>
{
    let fir = arr1(&FIR);
    let i = x.into_dimension();
    let u = stencil::pencil_5_x(f, i).dot(&fir);
    let v = stencil::pencil_5_y(f, i).dot(&fir);
    let w = stencil::pencil_5_z(f, i).dot(&fir);
    Vec3([u, v, w])
}

#[inline]
fn to_array(i: Ix3) -> [usize;3] {
    [i[0], i[1], i[2]]
}

#[inline]
fn grid_pos(ix: [isize;3]) -> Vec3 {
    Vec3([ix[0] as f64, ix[1] as f64, ix[2] as f64])
}

#[inline]
fn make_rel(a: [usize;3], b: [usize;3], shape: [usize;3]) -> [isize;3] {
    let mut result: [isize;3] = [0;3];
    for i in 0..3 {
        let d = b[i] as isize - a[i] as isize;
        result[i] = if d < -(shape[i] as isize)/2 {
            d + shape[i] as isize
        } else if d > (shape[i] as isize)/2 {
            d - shape[i] as isize
        } else {
            d
        };
    }
    result
}

pub struct EigenSolution<'a> {
    pub value: ArrayView3<'a, f64>,
    pub vector: ArrayView3<'a, Vec3>
}

impl<'a> marching_tetrahedra::Oracle for EigenSolution<'a> {
    fn grid_shape(&self) -> [usize;3] {
        to_array(self.value.raw_dim())
    }

    fn stencil(&self, x: [usize;3]) -> [f64;8] {
        let s = self.grid_shape();
        let mut result = [0.0;8];
        let e_ref = &self.vector[x];

        for (j, k) in indices([2, 2, 2]).into_iter().enumerate() {
            let other = [(x[0] + k.0) % s[0], (x[1] + k.1) % s[1], (x[2] + k.2) % s[2]];
            let sign = e_ref.dot(&self.vector[other]).signum();
            result[j] = sign * discrete_gradient(&self.value, other).dot(&self.vector[other]);
        }
        result
    }

    fn intersect(&self, y: f64, a: [usize;3], b: [usize;3]) -> Vec3 {
        let sign = self.vector[a].dot(&self.vector[b]).signum();
        let y_a = discrete_gradient(&self.value, a).dot(&self.vector[a]);
        let y_b = sign * discrete_gradient(&self.value, b).dot(&self.vector[b]);

        if y_a * y_b > 0.0 {
            eprintln!("warning: difficult point between {:?} and {:?}", a, b);
            if y_a.abs() < y_b.abs() {
                let a_rel = a.map(|i| i as isize);
                return grid_pos(a_rel);
            } else {
                let b_rel = b.map(|i| i as isize);
                return grid_pos(b_rel);
            }
        }

        let loc = (y - y_a) / (y_b - y_a);
        let a_rel = a.map(|i| i as isize);
        grid_pos(a_rel) + grid_pos(make_rel(a, b, self.grid_shape())) * loc
    }
}
```

``` {.rust file=src/caustics/mod.rs}
use crate::error::{Error};
use crate::box_properties::{BoxProperties};
use crate::numeric::{Vec3, Sym3, tuple3_idx};
use crate::marching_tetrahedra::{level_set, bound_level_set};
use crate::stencil;

use clap::{ArgMatches};
use ndarray::{Array3, ArrayView3, Ix3, indices, Array1, arr1, s};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

use std::str::FromStr;

mod a3;

#[inline]
fn get_or_create_group<S>(parent: &hdf5::Group, name: S) -> hdf5::Result<hdf5::Group>
    where /* T: Deref<Target=hdf5::Group>, */
          S: Into<String>
{
    let n: String = name.into();
    if !parent.member_names()?.contains(&n) {
        parent.create_group(n.as_str())
    } else {
        parent.group(n.as_str())
    }
}

macro_rules! group {
    ($home:expr, $name:expr) => {
        get_or_create_group($home, $name)?
    };
    ($home:expr, $name:expr, $($rest:tt),*) => {
        group!(&get_or_create_group($home, $name)?, $($rest),*)
    };
}

macro_rules! dataset {
    ($home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.dataset(name.as_str())?.read()? }
    };
    ($home:expr, $name:expr, $($rest:tt),*) => {
        dataset!($home.group($name)?, $($rest),*)
    };
}

macro_rules! write_dataset {
    ($array:ident: $type:ty => $home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.new_dataset::<$type>().shape($array.shape()).create(name.as_str())?.write($array.view())? }
    };
    ($array:ident: $type:ty => $home:expr, $name:expr, $($rest:tt),*) => {
        write_dataset!($array: $type => get_or_create_group($home, $name)?, $($rest),*)
    };
}

macro_rules! write_attribute {
    ($type:ty; $value:expr => $home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.new_attr::<$type>().create(name.as_str())?.write_scalar($value)? }
    };
    ($type:ty; $value:expr => $home:expr, $name:expr, $($rest:tt),*) => {
        write_attribute!($type; $value => get_or_create_group($home, $name)?, $($rest),*)
    };
}

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

    let vel_data = target.new_dataset::<f64>().shape([n, n, n, 3]).create("v")?;
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

            let real_view = ndarray::ArrayViewMut::from_shape([n, n, n], &mut real_buffer)?;
            write_dataset!(real_view: f64 => target, format!("H{}{}", i, j));
        }

        let mut hessian_f = ndarray::ArrayViewMut::from_shape([n, n, n / 2 + 1], &mut hessian_f_buffer)?;
        for (idx, v) in pot_f.indexed_iter() {
            let ki = bp.freq(tuple3_idx(idx, i));
            hessian_f[idx] = v * c64::new(0.0, ki / size);
        }
        ifft.c2r(&mut hessian_f_buffer, &mut real_buffer)?;
        let real_view = ndarray::ArrayView::from_shape([n, n, n], &real_buffer)?;
        vel_data.write_slice(real_view, s![..,..,..,i])?;
    }

    Ok(())
}

pub fn compute_eigenvalues(file: &hdf5::File, target: &hdf5::Group) -> Result<(), Error> {
    let bp = read_box_properties(file)?;
    let n: usize = bp.logical as usize;

    let h00: Array3<f64> = dataset!(target, "H00");
    let h01: Array3<f64> = dataset!(target, "H01");
    let h02: Array3<f64> = dataset!(target, "H02");
    let h11: Array3<f64> = dataset!(target, "H11");
    let h12: Array3<f64> = dataset!(target, "H12");
    let h22: Array3<f64> = dataset!(target, "H22");

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
            ev[idx] = h.eigenvector(lambda[idx]).normalize();
        }

        let name = format!("lambda{}", k);
        write_dataset!(lambda: f64 => target, &name, "eigenvalue");
        write_dataset!(ev: Vec3 => target, &name, "eigenvector");
    }

    Ok(())
}


pub fn run_eigen(args: &ArgMatches) -> Result<(), Error> {
    let filename = args.value_of("file").unwrap();
    let file = hdf5::File::open_rw(filename)?;
    let scale = args.value_of("scale").map(|s| f64::from_str(s).unwrap());

    let target_name = args.value_of("name").unwrap_or("0");
    let target = file.create_group(target_name)?;
    write_attribute!(f64; &scale.unwrap_or(0.0) => &file, target_name, "scale");

    compute_hessian(&file, &target, scale)?;
    compute_eigenvalues(&file, &target)?;

    Ok(())
}

fn time_tag(t: f64) -> String {
    format!("{:04}", (t * 1000.0).round() as u64)
}

pub fn run_a2(args: &ArgMatches) -> Result<(), Error> {
    let filename = args.value_of("file").unwrap();
    let file = hdf5::File::open_rw(filename)?;
    let name = args.value_of("name").unwrap_or("0");
    let group = file.group(name)?;
    let time = f64::from_str(args.value_of("growing-mode").unwrap())?;
    let tag = time_tag(time);
    let target = group.create_group(tag.as_str())?;
    write_attribute!(f64; &time => target, "growing-mode");

    let alpha: Array3<f64> = dataset!(file, name, "lambda0", "eigenvalue");
    let mesh = level_set(&alpha.view(), 1.0 / time);
    mesh.write_hdf5(&target.create_group("lambda0")?)?;
    Ok(())
}

pub fn run_a3(args: &ArgMatches) -> Result<(), Error> {
    use a3::EigenSolution;

    let filename = args.value_of("file").unwrap();
    let name = args.value_of("name").unwrap_or("0");
    let file = hdf5::File::open_rw(filename)?;
    let bp = read_box_properties(&file)?;
    let target = group!(&file, name, "a3");

    let alpha: Array3<f64> = dataset!(file, name, "lambda0", "eigenvalue");
    let e_alpha: Array3<Vec3> = dataset!(file, name, "lambda0", "eigenvector");
    let eigen_solution = EigenSolution { value: alpha.view(), vector: e_alpha.view() };
    let mesh = bound_level_set(&eigen_solution, 0.0, &alpha.view(), 1.0);

    if let Some(filename) = args.value_of("obj") {
        mesh.write_obj_file(&filename, bp.logical as f64)?;
    }

    mesh.write_hdf5(&target)?;
    Ok(())
}
```
