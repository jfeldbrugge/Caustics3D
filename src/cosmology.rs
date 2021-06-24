// ~\~ language=Rust filename=src/cosmology.rs
// ~\~ begin <<lit/cosmology.md|src/cosmology.rs>>[0]
pub const KM_PER_MPC: f64 = 3.24077929e-20;
pub const SEC_PER_GYR: f64 = 3.15576e16;

pub struct Cosmology {
    h: f64,
    omega_m: f64,
    omega_l: f64
}

pub const PLANCK_COSMOLOGY: Cosmology = Cosmology {
    h: 0.678,
    omega_m: 0.308,
    omega_l: 0.692
};

impl Cosmology {
    fn friedman_eqn(&self, a: f64) -> f64 {
        self.omega_m * a.powf(-3.0) + (1.0 - self.omega_m - self.omega_l) * a.powf(-2.0) + self.omega_l
    }

    fn da(&self, a: f64) -> f64 {
        self.friedman_eqn(a).sqrt() * 100.0 * self.h * a * KM_PER_MPC
    }

    fn limit_a(&self, t: f64) -> f64 {
        (self.omega_m.sqrt() * 3.0 * t * 100.0 * self.h * KM_PER_MPC / 2.0).powf(2.0/3.0)
    }

    fn limit_t(&self, a: f64) -> f64 {
        a.powf(3.0 / 2.0) * 2.0 / (self.omega_m.sqrt() * 3.0 * 100.0 * self.h * KM_PER_MPC)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn euler_method(dy: fn(f64) -> f64, t0: f64, y0: f64, dt: f64, stop: fn(f64, f64) -> bool) -> (f64, f64) {
        let mut t = t0;
        let mut y = y0;
        while !stop(t, y) {
            y += dt * dy(y);
            t += dt;
        }
        (t, y)
    }

    #[test]
    fn test_age_of_universe() {
        let t0 = 0.01 * SEC_PER_GYR;
        let dt = 0.01 * SEC_PER_GYR;
        let a0 = PLANCK_COSMOLOGY.limit_a(t0);
        let (age, _) = euler_method(
            |a| PLANCK_COSMOLOGY.da(a),
            t0, a0, dt, |_, a| { a >= 1.0 });
        println!("Age of the Universe: {} Gyr", age / SEC_PER_GYR);
        assert_approx_eq!(age / SEC_PER_GYR, 13.8, 0.1);
    }

    #[test]
    fn test_limit_fn() {
        let t0 = 0.01;
        let a0 = PLANCK_COSMOLOGY.limit_a(t0);
        assert_approx_eq!(t0, PLANCK_COSMOLOGY.limit_t(a0), 1e-6);
    }
}
// ~\~ end
