extern crate clap;
extern crate hdf5;
extern crate ndarray;
extern crate fftw;

use clap::{Arg, App, SubCommand, ArgMatches};
use ndarray::{AsArray, ArrayView, Array3, Array2, Ix2, Ix3};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

mod cosmology;
use cosmology::{Cosmology,EDS_COSMOLOGY,PLANCK_COSMOLOGY,SEC_PER_GYR};

use std::str::FromStr;
use std::ops::{Deref};
use std::f64::consts::{PI};

#[derive(Debug, Clone)]
enum Error {
    ArgumentError(String),
    Hdf5(hdf5::Error),
    Fftw(String)
}

impl From<hdf5::Error> for Error {
    fn from(e: hdf5::Error) -> Error {
        Error::Hdf5(e)
    }
}

fn run_cosmology(args: &ArgMatches) -> Result<(), Error> {
    let mut c: Cosmology = if args.is_present("eds") {
        EDS_COSMOLOGY.clone()
    } else {
        PLANCK_COSMOLOGY.clone()
    };

    if let Some(h_str) = args.value_of("hubble-constant") {
        let h = f64::from_str(h_str).map_err(
            |e| Error::ArgumentError(format!("could not parse Hubble constant: {}", e)))?;
        c.h = h;
    }
    if let Some(omega_m_str) = args.value_of("omega-m") {
        let omega_m = f64::from_str(omega_m_str).map_err(
            |e| Error::ArgumentError(format!("could not parse Omega matter: {}", e)))?;
        c.omega_m = omega_m;
    }
    if let Some(omega_l_str) = args.value_of("omega-l") {
        let omega_l = f64::from_str(omega_l_str).map_err(
            |e| Error::ArgumentError(format!("could not parse Omega Lambda: {}", e)))?;
        c.omega_l = omega_l;
    }

    println!("# t0 = {} Gyr", c.t0() / SEC_PER_GYR);
    println!("# a      D");
    for i in 0..30 {
        let a = f64::from(i) / 20.0;
        println!("{} {}", a, c.growing_mode(a));
    }
    Ok(())
}

struct BoxProperties {
    logical: u64,
    physical: f64
}

impl BoxProperties {
    fn half_size(&self) -> usize {
        self.logical as usize / 2
    }

    fn freq(&self, i: usize) -> f64 {
        let a: f64 = 2.0 * PI / self.physical;
        if i > self.half_size() {
            (i as f64) * a
        } else {
            ((i as i64 - self.logical as i64) as f64) * a
        }
    }
}

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
    // let ics = file.group("ics").or(file.create_group("ics"))?;
    let pot_view = pot.into();
    ics.new_dataset::<f64>().shape(pot_view.shape()).create("potential")?.write(pot_view)?;
    Ok(())
}

fn tuple3_idx<T>(x: (T, T, T), idx: usize) -> T {
    match idx {
        0 => x.0,
        1 => x.1,
        2 => x.2,
        _ => panic!("index out of range in tuple3_idx")
    }
}

fn run_gadget_ics(args: &ArgMatches) -> Result<(), Error> {
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
        forward.r2c(&mut a, &mut b).unwrap();
        
        let b_view : ndarray::ArrayView3<c64> =
            ndarray::ArrayView::from_shape([n, n, n/2 + 1], &b).unwrap();
        for (idx, v) in b_view.indexed_iter() {
            let ki = bp.freq(tuple3_idx(idx, i));
            let mut k_sqr: f64 = 0.0;
            for j in 0..3 { k_sqr += bp.freq(tuple3_idx(idx, j)).powi(2); }
            pot_k[idx] += v * c64::new(0.0, ki / k_sqr);
        }
    }

    pot_k[(0,0,0)] = c64::new(0.0, 0.0);
    let b_view : ndarray::ArrayViewMut3<c64> =
        ndarray::ArrayViewMut::from_shape([n, n, n/2 + 1], &mut b).unwrap();
    pot_k.assign_to(b_view);
    inverse.c2r(&mut b, &mut a).unwrap();

    let a_view : ndarray::ArrayView3<f64> =
        ndarray::ArrayView::from_shape([n, n, n], &a).unwrap();

    let output_file = hdf5::File::create(output_filename)?;
    write_metadata(&output_file, &gadget_data)?;
    write_potential(&output_file, a_view)?;

    Ok(())
}

fn main() -> Result<(), Error> {
    let args = App::new("Caustic Skeleton 3D")
        .version("0.1.0")
        .author("Feldbrugge and Hidding")
        .about("Compute caustic skeleton for 3d gradient deformation fields.")
        .subcommand(SubCommand::with_name("cosmology")
                    .about("computes properties of a given cosmology")
                    .arg(Arg::with_name("eds")
                         .long("eds")
                         .help("Einstein-de-Sitter universe (Ωm=1)"))
                    .arg(Arg::with_name("planck")
                         .long("planck")
                         .help("Planck universe (Ωm=0.308, Ωl=0.692, h=0.678)"))
                    .arg(Arg::with_name("hubble-constant")
                         .long("hubble")
                         .takes_value(true)
                         .help("Hubble constant H_0/100."))
                    .arg(Arg::with_name("omega-m")
                         .long("omega-m")
                         .takes_value(true)
                         .help("Ωm - matter density"))
                    .arg(Arg::with_name("omega-l")
                         .long("omega-l")
                         .takes_value(true)
                         .help("Ωl - dark energy density")))
        .subcommand(SubCommand::with_name("gadget-ics")
                    .about("recompute initial density field from Gadget4 ICs")
                    .arg(Arg::with_name("input")
                         .help("Input file, HDF5 with Gadget4 format.")
                         .required(true)
                         .index(1))
                    .arg(Arg::with_name("output")
                         .help("Output file, HDF5 archive.")
                         .long("output")
                         .short("o")
                         .takes_value(true)
                         .required(true)))
        .get_matches();

    match args.subcommand() {
        ("cosmology",  Some(args)) => run_cosmology(args),
        ("gadget-ics", Some(args)) => run_gadget_ics(args),
        _                          => Ok(())
    }
}
