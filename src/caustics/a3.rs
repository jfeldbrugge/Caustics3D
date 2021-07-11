// ~\~ language=Rust filename=src/caustics/a3.rs
// ~\~ begin <<lit/caustics.md|src/caustics/a3.rs>>[0]
use crate::stencil;
use crate::numeric::{Vec3};
use crate::marching_tetrahedra;

use ndarray::{ArrayView3, Ix3, indices};

#[inline]
fn to_array(i: Ix3) -> [usize;3] {
    [i[0], i[1], i[2]]
}

#[inline]
fn grid_pos(ix: [isize;3]) -> Vec3 {
    Vec3([ix[0] as f64, ix[1] as f64, ix[2] as f64])
}

#[inline]
fn make_rel(a: [usize;3], b: [usize;3], shape: [usize;3]) -> [isize;3] {
    let mut result: [isize;3] = [0;3];
    for i in 0..3 {
        let d = b[i] as isize - a[i] as isize;
        result[i] = if d < -(shape[i] as isize)/2 {
            d + shape[i] as isize
        } else if d > (shape[i] as isize)/2 {
            d - shape[i] as isize
        } else {
            d
        };
    }
    result
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
            result[j] = sign * stencil::discrete_gradient(&self.value, other).dot(&self.vector[other]);
        }
        result
    }

    fn intersect(&self, y: f64, a: [usize;3], b: [usize;3]) -> Vec3 {
        let sign = self.vector[a].dot(&self.vector[b]).signum();
        let y_a = stencil::discrete_gradient(&self.value, a).dot(&self.vector[a]);
        let y_b = sign * stencil::discrete_gradient(&self.value, b).dot(&self.vector[b]);

        if y_a * y_b > 0.0 {
            eprintln!("warning: difficult point between {:?} and {:?}", a, b);
            if y_a.abs() < y_b.abs() {
                let a_rel = a.map(|i| i as isize);
                return grid_pos(a_rel);
            } else {
                let b_rel = b.map(|i| i as isize);
                return grid_pos(b_rel);
            }
        }

        let loc = (y - y_a) / (y_b - y_a);
        let a_rel = a.map(|i| i as isize);
        grid_pos(a_rel) + grid_pos(make_rel(a, b, self.grid_shape())) * loc
    }
}
// ~\~ end
