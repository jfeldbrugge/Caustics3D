extern crate clap;
use clap::{Arg, App};

mod cosmology;

fn main() {
    let matches = App::new("Caustic Skeleton 3D")
        .version("0.1.0")
        .author("Feldbrugge and Hidding")
        .about("Compute caustic skeleton for 3d gradient deformation fields.")
        .arg(Arg::with_name("INPUT")
             .help("Input file, HDF5 with density or Gadget4 format.")
             .required(true)
             .index(1))
        .get_matches();

    println!("Using input file: {}", matches.value_of("INPUT").unwrap());
    println!("Hello, world!");
}
