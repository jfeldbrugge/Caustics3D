# Caustics 3D

Caustics codes for the filament project.

## Rust
The new version of the code is written in Rust. To build and run using `cargo`:

    cargo build --release
    cargo run --release -- help

The `--release` flag makes the executable a factor ~10 faster, which is very desirable even for smaller grid sizes.
Run tests with:

    cargo test

### Usage
This program is designed around command-line use and HDF5 files. The reason for this is to make it easy to get started with.
Every sub-command reads previous results from the HDF5 file and writes results to the same file, creating a single archive for each session.

- `gadget-ics` reads initial conditions from a Gadget4 run. You need to ask Gadget explicitely to write these files for you with optional flag `6`:

      mpirun ./Gadget4 param.txt 6
  
  You should be able to do this even after the simulation ran, since the random seed is stored in `param.txt`.
  The file written out is probably called something like `snapshot_ics_000.hdf5`.
  The `gadget-ics` command reads this file and computes the initial velocity potential from the velocities in the
  Gadget file. This potential and some cosmological parameters are stored in the output HDF5 archive.

- `caustics` expects a HDF5 archive with initial potential and cosmological parameters present. From this, the
  deformation tensor, its eigenvalues and vectors are computed, and the caustic skeleton is extracted. All are stored
  in a new group inside the HDF5 archive, so you can run this command multiple times with (for instance) different smoothing
  scales.

