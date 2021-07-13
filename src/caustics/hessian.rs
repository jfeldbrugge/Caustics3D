use crate::numeric::{Sym3,Vec3};
use crate::tricubic::{Tricubic,SecondOrder};
use crate::error::Error;

use ndarray::{Array3, ArrayView3, OwnedRepr};
use hdf5;

pub struct Hessian {
    data: [Tricubic<OwnedRepr<f64>,SecondOrder>;6]
}

pub struct EigenSystem<'a> {
    pub value: ArrayView3<'a, f64>,
    pub vector: ArrayView3<'a, Vec3>
}

impl Hessian {
    pub fn from_hdf5(target: &hdf5::Group) -> Result<Self, Error> {
        let data: [Array3<f64>;6] =
            [ dataset!(target, "H00")
            , dataset!(target, "H01")
            , dataset!(target, "H02")
            , dataset!(target, "H11")
            , dataset!(target, "H12")
            , dataset!(target, "H22") ];

        Ok(Self {
            data: data.map(|d| Tricubic::new(d, 1024))
        })
    }

    pub fn call(&self, x: Vec3) -> Sym3 {
        let mut m = [0.0;6];
        for k in 0..6 {
            m[k] = self.data[k].f(x.clone());
        }
        Sym3(m)
    }

    pub fn eigen_system(&self, x: Vec3, k: usize) -> (f64, Vec3) {
        let m = self.call(x);
        let (a, b, c) = m.eigenvalues();
        match k {
            0 => (a, m.eigenvector(a)),
            1 => (b, m.eigenvector(b)),
            2 => (c, m.eigenvector(c)),
            _ => panic!("only three eigenvalues")
        }
    }
}

