
#[derive(Debug, Clone)]
pub enum Error {
    ArgumentError(String),
    Hdf5(hdf5::Error),
    Fftw(String)
}

