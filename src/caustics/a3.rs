// ~\~ language=Rust filename=src/caustics/a3.rs
// ~\~ begin <<lit/caustics.md|src/caustics/a3.rs>>[0]
use crate::stencil;
use crate::numeric::{Vec3};
use crate::marching_tetrahedra;

use ndarray::{arr1, ArrayView3, Ix3, indices, IntoDimension};

const FIR: [f64;5] = [1./12., -2./3., 0., 2./3., -1./12.];

pub fn discrete_gradient<D>(f: &ArrayView3<f64>, x: D) -> Vec3
    where D: IntoDimension<Dim=Ix3>
{
    let fir = arr1(&FIR);
    let i = x.into_dimension();
    let u = stencil::pencil_5_x(f, i).dot(&fir);
    let v = stencil::pencil_5_y(f, i).dot(&fir);
    let w = stencil::pencil_5_z(f, i).dot(&fir);
    Vec3([u, v, w])
}

#[inline]
fn to_array(i: Ix3) -> [usize;3] {
    [i[0], i[1], i[2]]
}

#[inline]
fn grid_pos(ix: [usize;3]) -> Vec3 {
    Vec3([ix[0] as f64, ix[1] as f64, ix[2] as f64])
}

pub struct EigenSolution<'a> {
    pub value: ArrayView3<'a, f64>,
    pub vector: ArrayView3<'a, Vec3>
}

impl<'a> marching_tetrahedra::Oracle for EigenSolution<'a> {
    fn grid_shape(&self) -> [usize;3] {
        to_array(self.value.raw_dim())
    }

    fn stencil(&self, x: [usize;3]) -> [f64;8] {
        let s = self.grid_shape();
        let mut result = [0.0;8];
        let e_ref = &self.vector[x];

        for (j, k) in indices([2, 2, 2]).into_iter().enumerate() {
            let other = [(x[0] + k.0) % s[0], (x[1] + k.1) % s[1], (x[2] + k.2) % s[2]];
            let sign = e_ref.dot(&self.vector[other]).signum();
            result[j] = sign * discrete_gradient(&self.value, other).dot(&self.vector[other]);
        }
        result
    }

    fn intersect(&self, y: f64, a: [usize;3], b: [usize;3]) -> Vec3 {
        let sign = self.vector[a].dot(&self.vector[b]).signum();
        let y_a = discrete_gradient(&self.value, a).dot(&self.vector[a]);
        let y_b = discrete_gradient(&self.value, b).dot(&self.vector[b]);
        let loc = (y - y_a) / (y_b - y_a);
        grid_pos(a) + (grid_pos(b) - grid_pos(a)) * loc
    }
}
// ~\~ end
