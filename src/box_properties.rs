use std::f64::consts::{PI};

pub fn tuple3_idx<T>(x: (T, T, T), idx: usize) -> T {
    match idx {
        0 => x.0,
        1 => x.1,
        2 => x.2,
        _ => panic!("index out of range in tuple3_idx")
    }
}

pub struct BoxProperties {
    pub logical: u64,
    pub physical: f64
}

impl BoxProperties {
    pub fn half_size(&self) -> usize {
        self.logical as usize / 2
    }

    pub fn freq(&self, i: usize) -> f64 {
        let a: f64 = 2.0 * PI / self.physical;
        if i <= self.half_size() {
            (i as f64) * a
        } else {
            ((i as i64 - self.logical as i64) as f64) * a
        }
    }

    pub fn freq_sqr(&self, idx: (usize, usize, usize)) -> f64 {
        let mut k_sqr: f64 = 0.0;
        for j in 0..3 { k_sqr += self.freq(tuple3_idx(idx, j)).powi(2); }
        k_sqr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_freq() {
        let bp = BoxProperties { physical: 16.0, logical: 16 };
        assert_approx_eq!(bp.freq(8), PI, 1e-8);
    }
}
