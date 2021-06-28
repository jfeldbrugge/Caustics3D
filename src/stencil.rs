use ndarray::{ArrayView, Array, Ix3, Ix1, Axis, s};

pub trait Stencil<T, DIn, DOut>
    : Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>
{}
impl<T, DIn, DOut, S: Fn (&ArrayView<T, DIn>, DIn) -> Array<T, DOut>>
Stencil<T, DIn, DOut> for S {}

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

pub fn pencil_3(k: u8) -> impl Stencil<f64, Ix3, Ix1> {
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
                    *v = x[((i[0] + j + s - 1) % s, i[1], i[2])];
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
                    *v = x[(i[0], (i[1] + j + s - 1) % s, i[2])];
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
                    *v = x[(i[0], i[1], (i[2] + j + s - 1) % s)];
                }
                result
            }
        },
        _ => panic!("index error")
    }
}

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

        assert_eq!(pencil_3(0)(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[48.0, 0.0, 16.0]));
        assert_eq!(pencil_3(1)(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[12.0, 0.0,  4.0]));
        assert_eq!(pencil_3(2)(&x.view(), Ix3(0, 0, 0)),
                   arr1(&[ 3.0, 0.0,  1.0]));
        assert_eq!(pencil_3(0)(&x.view(), Ix3(2, 2, 2)),
                   arr1(&[ 26.0, 42.0, 58.0]));
        assert_eq!(pencil_3(2)(&x.view(), Ix3(2, 2, 2)),
                   arr1(&[ 41.0, 42.0, 43.0]));
    }
}

