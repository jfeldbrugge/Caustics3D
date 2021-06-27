use crate::cosmology::{Cosmology};
use crate::box_properties::{BoxProperties};
use crate::error::{Error};
use crate::numeric::{tuple3_idx};

use clap::{ArgMatches};
use ndarray::{AsArray, Array3, Array2, Ix2, Ix3};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

struct GadgetData {
    props: BoxProperties,
    cosmology: Cosmology,
    velocities: Array2<f64>
}

fn read_gadget_data(filename: &str) -> Result<GadgetData, hdf5::Error> {
    let input_file = hdf5::File::open(filename)?;
    let particles = input_file.group("PartType1")?;
    // let positions_ds = particles.dataset("Coordinates")?;
    let velocities_ds = particles.dataset("Velocities")?;

    let parameters = input_file.group("Parameters")?;
    let header = input_file.group("Header")?;

    let n: u64 = parameters.attr("GridSize")?.read_scalar()?;
    let l: f64 = header.attr("BoxSize")?.read_scalar()?;
    let h: f64 = parameters.attr("HubbleParam")?.read_scalar()?;
    let omega_m: f64 = parameters.attr("Omega0")?.read_scalar()?;
    let omega_l: f64 = parameters.attr("OmegaLambda")?.read_scalar()?;
    let cosmology = Cosmology { h: h, omega_m: omega_m, omega_l: omega_l };

    // let positions = positions_ds.read::<f64, Ix2>()?;
    let velocities = velocities_ds.read::<f64, Ix2>()?;

    let unit_velocity: f64 = parameters.attr("UnitVelocity_in_cm_per_s")?.read_scalar()?;
    let unit_length: f64 = parameters.attr("UnitLength_in_cm")?.read_scalar()?;
    let unit_time = unit_length / unit_velocity;
    let scale_factor: f64 = header.attr("Time")?.read_scalar()?;

    // we express velocities as the linearly extrapolated displacement at t=t0
    // or in other words u = dx / da = v / (D' a)
    // Gadget stores velocities with a sqrt(a) scaling
    let factor = h * scale_factor.sqrt() / (cosmology.da(scale_factor) * scale_factor * unit_time);
    eprintln!("Scaling velocities to growing mode solution, factor: {}", factor);
    let displacement = velocities.map(|x| x * factor);

    Ok(GadgetData {
        props: BoxProperties { logical: n, physical: l },
        cosmology: cosmology,
        velocities: displacement
    })
}

fn write_metadata(file: &hdf5::File, gadget_data: &GadgetData) -> Result<(), hdf5::Error> {
    let pars = file.create_group("parameters")?;
    pars.new_attr::<u64>().create("grid-size")?.write_scalar(&gadget_data.props.logical)?;
    pars.new_attr::<f64>().create("box-size")?.write_scalar(&gadget_data.props.physical)?;
    pars.new_attr::<f64>().create("hubble-constant")?.write_scalar(&gadget_data.cosmology.h)?;
    pars.new_attr::<f64>().create("omega-matter")?.write_scalar(&gadget_data.cosmology.omega_m)?;
    pars.new_attr::<f64>().create("omega-lambda")?.write_scalar(&gadget_data.cosmology.omega_l)?;
    Ok(())
}

fn write_potential<'a, A: AsArray<'a, f64, Ix3>>(file: &hdf5::File, pot: A) -> Result<(), hdf5::Error> {
    let ics = if file.member_names()?.iter().any(|x| x == "ics")
        { file.group("ics") } else { file.create_group("ics") }?;
    let pot_view = pot.into();
    ics.new_dataset::<f64>().shape(pot_view.shape()).create("potential")?.write(pot_view)?;
    Ok(())
}

pub fn run_gadget_ics(args: &ArgMatches) -> Result<(), Error> {
    let input_filename = args.value_of("input").unwrap();
    let output_filename = args.value_of("output").unwrap();

    let gadget_data = read_gadget_data(input_filename)?;
    let cosmos = &gadget_data.cosmology;
    eprintln!("Cosmology: Ωm={}, Ωl={}, h={}", cosmos.omega_m, cosmos.omega_l, cosmos.h);
    let bp = &gadget_data.props;
    eprintln!("Box: N={} pix L={} Mpc", bp.logical, bp.physical);
    let n = bp.logical as usize;

    let mut forward: R2CPlan64 = R2CPlan::aligned(&[n, n, n], Flag::ESTIMATE)
        .map_err(|e| Error::Fftw(e.to_string()))?;
    let mut inverse: C2RPlan64 = C2RPlan::aligned(&[n, n, n], Flag::ESTIMATE)
        .map_err(|e| Error::Fftw(e.to_string()))?;
    let mut a = AlignedVec::<f64>::new(n.pow(3));
    let mut b = AlignedVec::<c64>::new(n.pow(2) * (n / 2 + 1));
    let mut pot_k = Array3::<c64>::zeros([n, n, n/2 + 1]);

    for i in 0..3 {
        let v_k = gadget_data.velocities.slice(ndarray::s![.., i]);
        let a_view : ndarray::ArrayViewMut1<f64> = 
            ndarray::ArrayViewMut::from_shape(n.pow(3), &mut a).unwrap();
        v_k.assign_to(a_view);
        forward.r2c(&mut a, &mut b)?;
        
        let b_view : ndarray::ArrayView3<c64> =
            ndarray::ArrayView::from_shape([n, n, n/2 + 1], &b).unwrap();
        for (idx, v) in b_view.indexed_iter() {
            let ki = bp.freq(tuple3_idx(idx, i));
            let k_sqr = bp.freq_sqr(idx);
            pot_k[idx] += v * c64::new(0.0, ki / k_sqr);
        }
    }

    pot_k[(0,0,0)] = c64::new(0.0, 0.0);
    let b_view : ndarray::ArrayViewMut3<c64> =
        ndarray::ArrayViewMut::from_shape([n, n, n/2 + 1], &mut b).unwrap();
    pot_k.assign_to(b_view);
    inverse.c2r(&mut b, &mut a)?;

    let mut a_view : ndarray::ArrayViewMut3<f64> =
        ndarray::ArrayViewMut::from_shape([n, n, n], &mut a).unwrap();
    let n_elem: f64 = (n as f64).powi(3);
    a_view.mapv_inplace(|x| x / n_elem);

    let output_file = hdf5::File::create(output_filename)?;
    write_metadata(&output_file, &gadget_data)?;
    write_potential(&output_file, a_view.view())?;

    Ok(())
}

