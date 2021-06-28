use crate::numeric::{Vec3, Sym3};

#[macro_use]
use ndarray::{ArrayView, Array, Ix3, Ix2, Ix1, Axis, s};

trait Stencil<T, DIn, DOut>
    : Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>
{}
impl<T, DIn, DOut, S: Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>>
Stencil<T, DIn, DOut> for S {}

pub fn stencil_3x3x3(x: &ArrayView<f64,Ix3>, i: Ix3) -> Array<f64, Ix3>
{
    let s = x.shape();
    if i[0] > 0 && i[0] < (s[0] - 1) && i[0] > 0
    && i[1] < (s[1] - 1) && i[2] > 0 && i[2] < (s[2] - 1) {
        x.slice(s![ (i[0] - 1) .. (i[0] + 2)
                  , (i[1] - 1) .. (i[1] + 2)
                  , (i[2] - 1) .. (i[2] + 2) ]).into_owned()
    } else {
        let mut result = Array::<f64, Ix3>::zeros([3, 3, 3]);
        for (j, v) in result.indexed_iter_mut() {
            *v = x[((i[0] + j.0 - 1) % s[0]
                  , (i[1] + j.1 - 1) % s[1]
                  , (i[2] + j.2 - 1) % s[2])];
        }
        result
    }
}

pub fn stencil_pencil(k: u8) -> impl Stencil<f64, Ix3, Ix1> {
    match k {
        0 => |x: &ArrayView<f64,Ix3>, i: Ix3| -> Array<f64, Ix1> {
            let s = x.shape()[0];
            if i[0] > 0 && i[0] < s {
                x.index_axis(Axis(1), i[1])
                 .index_axis(Axis(1), i[2])
                 .slice(s![(i[0] - 1) .. (i[0] + 2)]).into_owned()
            } else {
                let mut result = Array::<f64,Ix1>::zeros([3]);
                for (j, v) in result.indexed_iter_mut() {
                    *v = x[((i[0] + j - 1) % s, i[1], i[2])];
                }
                result
            }
        },
        1 => |x: &ArrayView<f64,Ix3>, i: Ix3| -> Array<f64, Ix1> {
            let s = x.shape()[1];
            if i[1] > 0 && i[1] < s {
                x.index_axis(Axis(0), i[0])
                 .index_axis(Axis(1), i[2])
                 .slice(s![(i[1] - 1) .. (i[1] + 2)]).into_owned()
            } else {
                let mut result = Array::<f64,Ix1>::zeros([3]);
                for (j, v) in result.indexed_iter_mut() {
                    *v = x[(i[0], (i[1] + j - 1) % s, i[2])];
                }
                result
            }
        },
        2 => |x: &ArrayView<f64,Ix3>, i: Ix3| -> Array<f64, Ix1> {
            let s = x.shape()[0];
            if i[0] > 0 && i[0] < s {
                x.index_axis(Axis(0), i[0])
                 .index_axis(Axis(0), i[1])
                 .slice(s![(i[2] - 1) .. (i[2] + 2)]).into_owned()
            } else {
                let mut result = Array::<f64,Ix1>::zeros([3]);
                for (j, v) in result.indexed_iter_mut() {
                    *v = x[(i[0], i[1], (i[2] + j - 1) % s)];
                }
                result
            }
        },
        _ => panic!("index error")
    }
}

