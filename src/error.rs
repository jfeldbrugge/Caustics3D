
#[derive(Debug, Clone)]
pub enum Error {
    ArgumentError(String),
    Hdf5(hdf5::Error),
    Fftw(String),
    Internal(String),
    IO(String)
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::IO(format!("{}", e))
    }
}

impl From<hdf5::Error> for Error {
    fn from(e: hdf5::Error) -> Error {
        Error::Hdf5(e)
    }
}

impl From<fftw::error::Error> for Error {
    fn from(e: fftw::error::Error) -> Error {
        Error::Fftw(format!("{}", e))
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(e: ndarray::ShapeError) -> Error {
        Error::Internal(format!("{}", e))
    }
}

