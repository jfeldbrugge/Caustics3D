use std::f64::consts::PI;

pub trait Modulo<RHS=Self> {
    type Output;

    fn modulo(self, rhs: RHS) -> Self::Output;
}

impl<A: num_traits::Num + Ord + Copy> Modulo for A {
    type Output = Self;

    #[inline]
    fn modulo(self, rhs: Self) -> Self {
        if self < Self::zero() {
            self % rhs + rhs
        } else {
            self % rhs
        }
    }
}

#[derive(Clone,Debug)]
pub struct Vec3 (pub [f64; 3]);
#[derive(Clone,Debug)]
pub struct Sym3 (pub [f64; 6]);

impl std::ops::Index<(u8, u8)> for Sym3 {
    type Output = f64;
    fn index(&self, idx: (u8, u8)) -> &Self::Output {
        let (i, j) = idx;
        if i > j {
            self.index((j, i))
        } else {
            let k: usize = (((7 - i) * i) / 2 + j) as usize;
            let Sym3(d) = self;
            &d[k]
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Self) -> Self {
        let Vec3(a) = self;
        let Vec3(b) = other;
        let mut c = [0.0; 3];
        for i in 0..3 { c[i] = a[i] + b[i]; }
        Vec3(c)
    }
}

impl num_traits::identities::Zero for Vec3 {
    fn zero() -> Self {
        Vec3([0.0, 0.0, 0.0])
    }

    fn is_zero(&self) -> bool {
        let Vec3(d) = self;
        d.iter().all(|x| x.is_zero())
    }
}

unsafe impl hdf5::H5Type for Vec3 {
    fn type_descriptor() -> hdf5::types::TypeDescriptor {
        use hdf5::types::{TypeDescriptor,FloatSize};
        TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Float(FloatSize::U8)), 3)
    }
}

pub fn tuple3_idx<T>(x: (T, T, T), idx: usize) -> T {
    match idx {
        0 => x.0,
        1 => x.1,
        2 => x.2,
        _ => panic!("index out of range in tuple3_idx")
    }
}

fn tuple3_sort(x: (f64, f64, f64)) -> (f64, f64, f64) {
    let (a, b, c) = x;

    if a > b {
        if b > c {
            (a, b, c)
        } else if c > a {
            (c, a, b)
        } else {
            (a, c, b)
        }
    } else {
        if a > c {
            (b, a, c)
        } else if c > b {
            (c, b, a)
        } else {
            (b, c, a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple3_sort() {
        assert_eq!(tuple3_sort((1., 2., 3.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((1., 3., 2.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((2., 1., 3.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((2., 3., 1.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((3., 1., 2.)), (3., 2., 1.));
        assert_eq!(tuple3_sort((3., 2., 1.)), (3., 2., 1.));
    }
}

impl Sym3 {
    pub fn trace(&self) -> f64 {
        let Sym3(d) = self;
        d[0] + d[3] + d[5]
    }

    pub fn square_trace(&self) -> f64 {
        let Sym3(d) = self;
        d[0]*d[0] + 2.*d[1]*d[1] + 2.*d[2]*d[2]
                  +    d[3]*d[3] + 2.*d[4]*d[4]
                                 +    d[5]*d[5]
    }

    pub fn eigenvalues(&self) -> (f64, f64, f64) {
        let Sym3(d) = self;
        let q = self.trace() / 3.;
        let mut f: [f64; 6] = d.clone();
        f[0] -= q; f[3] -= q; f[5] -=q;

        let p = (self.square_trace() / 6.).sqrt();
        for i in 0..6 { f[i] /= p; }
        let phi = (Sym3(f).trace() / 2.).acos() / 3.;

        let a = q + 2. * p * phi.cos();
        let b = q + 2. * p * (phi + 2. * PI / 3.).cos();
        let c = q + 2. * p * (phi + 4. * PI / 3.).cos();

        tuple3_sort((a, b, c))
    }

    pub fn eigenvector(&self, l: f64) -> Vec3 {
        let Sym3(d) = self;
        Vec3([ (l - d[3]) * d[2] + d[4] * d[1]
             , (l - d[0]) * d[4] + d[3] * d[1]
             , (l - d[0]) * (l - d[1]) - d[0] * d[0] ])
    }
}

