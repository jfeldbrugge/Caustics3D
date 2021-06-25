extern crate clap;
use clap::{Arg, App, SubCommand, ArgMatches};

mod cosmology;
use cosmology::{Cosmology,EDS_COSMOLOGY,PLANCK_COSMOLOGY,SEC_PER_GYR};

use std::str::FromStr;

#[derive(Debug, Clone)]
enum Error {
    ArgumentError(String)
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
            |e| Error::ArgumentError(format!("could not parse Omega matter: {}", e)))?;
        c.omega_l = omega_l;
    }

    println!("# t0 = {} Gyr", c.t0() / SEC_PER_GYR);
    println!("# a      D");
    for i in 0..200 {
        let a = f64::from(i) / 100.0;
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
        .subcommand(SubCommand::with_name("prepare-gadget-ics")
                    .about("recompute initial density field from Gadget4 ICs")
                    .arg(Arg::with_name("INPUT")
                         .help("Input file, HDF5 with density or Gadget4 format.")
                         .required(true)
                         .index(1)))
        .get_matches();

    match args.subcommand() {
        ("cosmology", Some(args)) => run_cosmology(args),
        _                         => Ok(())
    }
}
