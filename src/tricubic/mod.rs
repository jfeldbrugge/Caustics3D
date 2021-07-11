use crate::stencil;
use crate::numeric::{Vec3};

use ndarray::{ArrayView3, indices, Ix3};

mod table;
use table::COEFFICIENTS;

pub struct Tricubic<'a> (ArrayView3<'a,f64>);

#[inline]
fn offset(a: [usize;3], b: (usize, usize, usize), s: Ix3) -> [usize;3]
{
    [ (a[0] + b.0) % s[0], (a[1] + b.1) % s[1], (a[2] + b.2) % s[2] ]
}

impl<'a> Tricubic<'a> {
    fn interpolate(&self, x: Vec3) -> f64 {
        let s = self.0.raw_dim();
        let i = x.0.map(|j| j.floor() as usize);
        let f = x.0.zip(i).map(|(j, k)| j - (k as f64));
        let y = stencil::flat_2x2x2(&self.0, i);

        let mut c = [0.0; 64];
        for (j, dx) in indices([2, 2, 2]).into_iter().enumerate() {
            let idx = offset(i, dx, s);
            c[j]    = y[j];
            c[8+j]  = derivative!(&self.0, 0, idx);
            c[16+j] = derivative!(&self.0, 1, idx);
            c[24+j] = derivative!(&self.0, 2, idx);
            c[32+j] = derivative!(&self.0, 0, 1, idx);
            c[40+j] = derivative!(&self.0, 1, 2, idx);
            c[48+j] = derivative!(&self.0, 0, 2, idx);
            c[56+j] = derivative!(&self.0, 0, 1, 2, idx);
        }

        let mut b = [0.0; 64];
        for (i, j, m) in COEFFICIENTS {
            b[i] += c[j] * m;
        }

        let mut v: f64 = 0.0;
        for (j, dx) in indices([4, 4, 4]).into_iter().enumerate() {
            v += b[j] * f[0].powi(dx.0 as i32) * f[1].powi(dx.1 as i32) * f[2].powi(dx.2 as i32);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grf;
    use crate::box_properties::{BoxProperties};

    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_tricubic() {
        let mut file = File::create("splinetest.dat").unwrap();
        let f = grf::white_noise(&BoxProperties{ physical: 1.0, logical: 8 });
        let t = Tricubic(f.view());
        for i in indices([64,64,64]).into_iter() {
            let x = Vec3([i.0 as f64 / 8., i.1 as f64 / 8., i.2 as f64 / 8.]);
            write!(&mut file, "{} ", t.interpolate(x)).unwrap();
        }
    }
}
