use ndarray::{ArrayView, ArrayView3, Array, Ix3, Ix1, Axis, s, indices, arr1};
use crate::numeric::{Vec3};

pub trait Stencil<T, DIn, DOut>
    : Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>
{}
impl<T, DIn, DOut, S: Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>>
Stencil<T, DIn, DOut> for S {}

/// Stencil of a 2x2x2 area. This is used to iterate over volume
/// elements in the grid.
pub fn stencil_2x2x2(x: &ArrayView<f64,Ix3>, i: Ix3) -> Array<f64, Ix3>
{
    let s = x.shape();
    if i[0] < (s[0] - 1) && i[1] < (s[1] - 1) && i[2] < (s[2] - 1) {
        x.slice(s![ i[0] .. (i[0] + 2)
                  , i[1] .. (i[1] + 2)
                  , i[2] .. (i[2] + 2) ]).into_owned()
    } else {
        let mut result = Array::<f64, Ix3>::zeros([2, 2, 2]);
        for (j, v) in result.indexed_iter_mut() {
            *v = x[((i[0] + j.0) % s[0]
                  , (i[1] + j.1) % s[1]
                  , (i[2] + j.2) % s[2])];
        }
        result
    }
}

pub fn flat_2x2x2(x: &ArrayView<f64,Ix3>, i: [usize;3]) -> [f64;8]
{
    let s = x.shape();
    if i[0] < (s[0] - 1) && i[1] < (s[1] - 1) && i[2] < (s[2] - 1) {
        let mut result = [0.0; 8];
        x.slice(s![ i[0] .. (i[0] + 2)
                  , i[1] .. (i[1] + 2)
                  , i[2] .. (i[2] + 2) ])
         .iter().enumerate()
         .for_each(|(j, v)| { result[j] = *v; });
        result
    } else {
        let mut result = [0.0; 8];
        for (j, k) in indices([2, 2, 2]).into_iter().enumerate() {
            result[j] = x[((i[0] + k.0) % s[0]
                         , (i[1] + k.1) % s[1]
                         , (i[2] + k.2) % s[2])];
        }
        result
    }
}

/// Stencil of a 3x3x3 area. Could be useful for computing Laplacians
/// for instance.
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
            *v = x[((i[0] + j.0 + s[0] - 1) % s[0]
                  , (i[1] + j.1 + s[1] - 1) % s[1]
                  , (i[2] + j.2 + s[2] - 1) % s[2])];
        }
        result
    }
}

macro_rules! pencil_select {
    ( 0, $x:expr, $i:expr ) => {
        $x.index_axis(Axis(1), $i[1])
          .index_axis(Axis(1), $i[2])
    };
    ( 1, $x:expr, $i:expr ) => {
        $x.index_axis(Axis(0), $i[0])
          .index_axis(Axis(1), $i[2])
    };
    ( 2, $x:expr, $i:expr ) => {
        $x.index_axis(Axis(0), $i[0])
          .index_axis(Axis(0), $i[1])
    };
}

macro_rules! pencil_index {
    ( $k:tt, $x:expr, $i:expr, $half:expr ) => {
        pencil_select!($k, $x, $i)
          .slice(s![($i[$k] - $half) .. ($i[$k] + $half + 1)]).into_owned()
    };
}

macro_rules! pencil_collect {
    ( 0, $v:expr, $x:expr, $i:expr, $j:expr, $s:expr, $half:expr ) => {
        *$v = $x[(($i[0] + $j + $s - $half) % $s, $i[1], $i[2])];
    };
    ( 1, $v:expr, $x:expr, $i:expr, $j:expr, $s:expr, $half:expr ) => {
        *$v = $x[($i[0], ($i[1] + $j + $s - $half) % $s, $i[2])];
    };
    ( 2, $v:expr, $x:expr, $i:expr, $j:expr, $s:expr, $half:expr ) => {
        *$v = $x[($i[0], $i[1], ($i[2] + $j + $s - $half) % $s)];
    };
}

/// Macro for defining pencil beam stencils, or pencils for short.
/// These are used to compute derivatives on the grid. For instance,
/// the five-point pencil can be used to compute derivatives to the
/// second order of precision, using a [1/12, -2/3, 0, 2/3, -1/12]
/// FIR filter.
#[macro_export]
macro_rules! define_pencil {
    ( $vis:vis $name:ident, $k:tt, $w:expr ) => {
        $vis fn $name(x: &ArrayView<f64,Ix3>, i: Ix3) -> Array<f64,Ix1> {
            const W: usize = $w;
            const HALF: usize = $w/2;

            let s = x.shape()[$k];
            if i[$k] > HALF-1 && i[$k] < s-HALF {
                pencil_index!($k, x, i, HALF)
            } else {
                let mut result = Array::<f64,Ix1>::zeros([W]);
                for (j, v) in result.indexed_iter_mut() {
                    pencil_collect!($k, v, x, i, j, s, HALF);
                }
                result
            }
        }
    };
}

define_pencil!(pub pencil_3_x, 0, 3);
define_pencil!(pub pencil_3_y, 1, 3);
define_pencil!(pub pencil_3_z, 2, 3);
define_pencil!(pub pencil_5_x, 0, 5);
define_pencil!(pub pencil_5_y, 1, 5);
define_pencil!(pub pencil_5_z, 2, 5);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_stencil_2x2x2() {
        let x = Array::range(0.0, 64.0, 1.0).into_shape([4,4,4]).unwrap();
        let y = ArrayView::from(
                &[63.0, 60.0, 51.0, 48.0,
                  15.0, 12.0,  3.0,  0.0])
            .into_shape([2, 2, 2]).unwrap();

        assert_eq!(stencil_2x2x2(&x.view(), Ix3(1, 1, 1)),
                   x.slice(s![1..3,1..3,1..3]).into_owned());
        assert_eq!(stencil_2x2x2(&x.view(), Ix3(3, 3, 3)), y);
    }

    #[test]
    fn test_stencil_3x3x3() {
        let x = Array::range(0.0, 64.0, 1.0).into_shape([4,4,4]).unwrap();
        let y = ArrayView::from(
                &[63.0, 60.0, 61.0, 51.0, 48.0, 49.0, 55.0, 52.0, 53.0,
                  15.0, 12.0, 13.0,  3.0,  0.0,  1.0,  7.0,  4.0,  5.0,
                  31.0, 28.0, 29.0, 19.0, 16.0, 17.0, 23.0, 20.0, 21.0])
            .into_shape([3, 3, 3]).unwrap();

        assert_eq!(stencil_3x3x3(&x.view(), Ix3(1, 1, 1)),
                   x.slice(s![0..3,0..3,0..3]).into_owned());
        assert_eq!(stencil_3x3x3(&x.view(), Ix3(0, 0, 0)), y);
    }

    #[test]
    fn test_pencil_3() {
        let x = Array::range(0.0, 64.0, 1.0).into_shape([4,4,4]).unwrap();

        assert_eq!(pencil_3_x(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[48.0, 0.0, 16.0]));
        assert_eq!(pencil_3_y(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[12.0, 0.0,  4.0]));
        assert_eq!(pencil_3_z(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[ 3.0, 0.0,  1.0]));
        assert_eq!(pencil_3_x(&x.view(), Ix3(2, 2, 2)),
                   arr1(&[ 26.0, 42.0, 58.0]));
        assert_eq!(pencil_3_z(&x.view(), Ix3(2, 2, 2)),
                   arr1(&[ 41.0, 42.0, 43.0]));

        assert_eq!(pencil_5_x(&x.view(), Ix3(1, 2, 0)),
                   arr1(&[ 56.0, 8.0, 24.0, 40.0, 56.0 ]));
        assert_eq!(pencil_5_y(&x.view(), Ix3(1, 2, 0)),
                   arr1(&[ 16.0, 20.0, 24.0, 28.0, 16.0 ]));
        assert_eq!(pencil_5_z(&x.view(), Ix3(1, 2, 0)),
                   arr1(&[ 26.0, 27.0, 24.0, 25.0, 26.0 ]));
    }
}

