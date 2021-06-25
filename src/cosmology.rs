// ~\~ language=Rust filename=src/cosmology.rs
// ~\~ begin <<lit/cosmology.md|src/cosmology.rs>>[0]
extern crate rgsl;

pub const KM_PER_MPC: f64 = 3.24077929e-20;
pub const SEC_PER_GYR: f64 = 3.15576e16;

// ~\~ begin <<lit/cosmology.md|cosmology-parameters>>[0]
#[derive(Debug, Clone)]
pub struct Cosmology {
    pub h: f64,
    pub omega_m: f64,
    pub omega_l: f64
}
// ~\~ end

pub const PLANCK_COSMOLOGY: Cosmology = Cosmology {
    h: 0.678,
    omega_m: 0.308,
    omega_l: 0.692
};

pub const EDS_COSMOLOGY: Cosmology = Cosmology {
    h: 0.7,
    omega_m: 1.0,
    omega_l: 0.0
};

fn euler_method<F1, Pred>(dy: F1, t0: f64, y0: f64, dt: f64, stop: Pred) -> (f64, f64)
    where F1: Fn(f64) -> f64,
          Pred: Fn(f64, f64) -> bool
{
    let mut t = t0;
    let mut y = y0;
    while !stop(t, y) {
        y += dt * dy(y);
        t += dt;
    }
    (t, y)
}

impl Cosmology {
    // ~\~ begin <<lit/cosmology.md|friedman-equation>>[0]
    fn friedman_eqn(&self, a: f64) -> f64 {
        self.omega_m * a.powf(-3.0) + (1.0 - self.omega_m - self.omega_l) * a.powf(-2.0) + self.omega_l
    }
    // ~\~ end
    // ~\~ begin <<lit/cosmology.md|scale-factor-da>>[0]
    pub fn da(&self, a: f64) -> f64 {
        self.friedman_eqn(a).sqrt() * 100.0 * self.h * a * KM_PER_MPC
    }
    // ~\~ end
    // ~\~ begin <<lit/cosmology.md|scale-factor-limit>>[0]
    pub fn limit_a(&self, t: f64) -> f64 {
        (self.omega_m.sqrt() * 3.0 * t * 100.0 * self.h * KM_PER_MPC / 2.0).powf(2.0/3.0)
    }

    #[cfg(test)]
    pub fn limit_t(&self, a: f64) -> f64 {
        a.powf(3.0 / 2.0) * 2.0 / (self.omega_m.sqrt() * 3.0 * 100.0 * self.h * KM_PER_MPC)
    }
    // ~\~ end
    // ~\~ begin <<lit/cosmology.md|growing-mode>>[0]
    pub fn growing_mode(&self, a: f64) -> f64 {
        let integrant = |a| {
            (self.friedman_eqn(a).sqrt() * a).powf(-3.0)
        };

        const A_SMALL: f64 = 0.001;
        if a < A_SMALL {
            a
        } else {
            let int_init  = 0.4 * A_SMALL.powf(2.5) * self.omega_m.powf(-1.5);
            let (int_rest, _, _, _) = rgsl::integration::qk15(integrant, A_SMALL, a);
            let hubble_factor = self.friedman_eqn(a).sqrt();
            (int_init + int_rest) * hubble_factor * 2.5 * self.omega_m
        }
    }
    // ~\~ end
    pub fn t0(&self) -> f64 {
        let t0 = 0.01 * SEC_PER_GYR;
        let dt = 0.01 * SEC_PER_GYR;
        let a0 = self.limit_a(t0);
        let (age, _) = euler_method(
            |a| self.da(a),
            t0, a0, dt, |_, a| { a >= 1.0 });
        age
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_age_of_universe() {
        assert_approx_eq!(PLANCK_COSMOLOGY.t0() / SEC_PER_GYR, 13.8, 0.1);
    }

    #[test]
    fn test_limit_fn() {
        let t0 = 0.01;
        let a0 = PLANCK_COSMOLOGY.limit_a(t0);
        assert_approx_eq!(t0, PLANCK_COSMOLOGY.limit_t(a0), 1e-6);
    }
}
// ~\~ end
