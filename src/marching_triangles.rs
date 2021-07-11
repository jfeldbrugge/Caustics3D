use crate::error::Error;
use crate::mesh::Mesh;
use crate::numeric::{Vec3};

struct Curve {
    pub vertices: Vec<Vec3>,
    pub edges: Vec<[usize;2]>
}

fn level_set_curve<F>(mesh: &Mesh, f: F) -> Curve
    where F: Fn(Vec3, Vec3) -> Option<Vec3>
{
    Curve {
        vertices: Vec::new(),
        edges: Vec::new()
    }
}

