extern crate clap;
extern crate hdf5;
extern crate ndarray;
extern crate fftw;

use clap::{Arg, App, SubCommand, ArgMatches};
use ndarray::{Array0, Array3, Array2, Ix2};
use fftw::types::{Flag, c64};
use fftw::plan::{R2CPlan, R2CPlan64, C2RPlan, C2RPlan64};
use fftw::array::{AlignedVec};

mod cosmology;
use cosmology::{Cosmology,EDS_COSMOLOGY,PLANCK_COSMOLOGY,SEC_PER_GYR};

use std::str::FromStr;
use std::ops::{Deref};

#[derive(Debug, Clone)]
enum Error {
    ArgumentError(String),
    Hdf5(hdf5::Error),
    Fftw(String)
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

struct BoxSize {
    logical: u64,
    physical: f64
}

struct GadgetData {
    size: BoxSize,
    cosmology: Cosmology,
    velocities: Array2<f64>
}

fn read_gadget_data(filename: &str) -> Result<GadgetData, hdf5::Error> {
    let input_file = hdf5::File::open(filename)?;
    let particles = input_file.group("PartType1")?;
    let positions_ds = particles.dataset("Coordinates")?;
    let velocities_ds = particles.dataset("Velocities")?;

    let parameters = input_file.group("Parameters")?;
    let header = input_file.group("Header")?;

    let n: u64 = parameters.attr("GridSize")?.read_scalar()?;
    let l: f64 = header.attr("BoxSize")?.read_scalar()?;
    eprintln!("box: n={}, l={}", n, l);

    let h: f64 = parameters.attr("HubbleParam")?.read_scalar()?;
    let omega_m: f64 = parameters.attr("Omega0")?.read_scalar()?;
    let omega_l: f64 = parameters.attr("OmegaLambda")?.read_scalar()?;
    let cosmology = Cosmology { h: h, omega_m: omega_m, omega_l: omega_l };
    eprintln!("cosmology: h={}, Ωm={}, Ωl={}", h, omega_m, omega_l);

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
        size: BoxSize { logical: n, physical: l },
        cosmology: cosmology,
        velocities: displacement
    })
}

fn run_gadget_ics(args: &ArgMatches) -> Result<(), Error> {
    let input_filename = args.value_of("input").unwrap();
    let gadget_data = read_gadget_data(input_filename).map_err(Error::Hdf5)?;
    let n = gadget_data.size.logical as usize;

    let mut forward: R2CPlan64 = R2CPlan::aligned(&[n, n, n], Flag::ESTIMATE)
        .map_err(|e| Error::Fftw(e.to_string()))?;
    let mut inverse: C2RPlan64 = C2RPlan::aligned(&[n, n, n], Flag::ESTIMATE)
        .map_err(|e| Error::Fftw(e.to_string()))?;
    let mut a = AlignedVec::<f64>::new(n.pow(3));
    let mut b = AlignedVec::<c64>::new(n.pow(3) / 2);

    for k in 0..3 {
        let v_k = gadget_data.velocities.slice(ndarray::s![.., k]);
        let mut a_view : ndarray::ArrayViewMut1<f64> = 
            ndarray::ArrayViewMut::from_shape(n.pow(3), &mut a).unwrap();
        v_k.assign_to(a_view);
        forward.r2c(&mut a, &mut b).unwrap();
        

    }

    let potential = Array3::<f64>::zeros([n, n, n]);
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
