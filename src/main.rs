#![feature(array_map)]
#![feature(array_zip)]

extern crate clap;
extern crate hdf5;
extern crate ndarray;
extern crate fftw;
extern crate num_traits;

use clap::{Arg, App, SubCommand, ArgMatches};
#[macro_use] mod stencil;
#[macro_use] mod util;
mod grf;
mod cosmology;
mod box_properties;
mod gadget_data;
mod error;
mod caustics;
mod numeric;
mod marching_tetrahedra;
mod marching_triangles;
mod tricubic;
mod mesh;

use cosmology::{Cosmology,EDS_COSMOLOGY,PLANCK_COSMOLOGY,SEC_PER_GYR};
use error::{Error};

use std::str::FromStr;

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
                    .about("recompute initial potential field from Gadget4 ICs")
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
        .subcommand(SubCommand::with_name("eigen")
                    .about("compute eigenvalues")
                    .arg(Arg::with_name("file")
                         .help("HDF5 archive to work on")
                         .required(true)
                         .index(1))
                    .arg(Arg::with_name("scale")
                         .help("set smoothing scale")
                         .long("scale")
                         .short("s")
                         .takes_value(true))
                    .arg(Arg::with_name("name")
                         .help("name of output group in HDF5 file (default: 0)")
                         .long("name")
                         .short("n")
                         .takes_value(true)))
        .subcommand(SubCommand::with_name("a2")
                    .about("compute A2 surfaces")
                    .arg(Arg::with_name("file")
                         .help("HDF5 archive to work on")
                         .required(true)
                         .index(1))
                    .arg(Arg::with_name("name")
                         .help("output group")
                         .long("name")
                         .short("n")
                         .takes_value(true))
                    .arg(Arg::with_name("growing-mode")
                         .help("value of growing-mode solution (aka time)")
                         .long("growing-mode")
                         .short("D")
                         .takes_value(true)
                         .required(true)))
        .subcommand(SubCommand::with_name("a3")
                    .about("compute A3 bigcaustic")
                    .arg(Arg::with_name("file")
                         .help("HDF5 archive to work on")
                         .required(true)
                         .index(1))
                    .arg(Arg::with_name("name")
                         .help("output group")
                         .long("name")
                         .short("n")
                         .takes_value(true)
                         .required(true))
                    .arg(Arg::with_name("obj")
                         .help("write mesh to obj file")
                         .long("dump-obj")
                         .takes_value(true)))
        .get_matches();

    match args.subcommand() {
        ("cosmology",  Some(args)) => run_cosmology(args),
        ("gadget-ics", Some(args)) => gadget_data::run_gadget_ics(args),
        ("eigen",      Some(args)) => caustics::run_eigen(args),
        ("a2",         Some(args)) => caustics::run_a2(args),
        // ("a3",         Some(args)) => caustics::run_a3(args),
        _                          => Ok(())
    }
}
