use crate::stencil::{flat_2x2x2};
use crate::numeric::{Vec3};
use crate::stencil;
use crate::error::{Error};

use ndarray::{Ix3, ArrayView3, indices};
// use num_traits::identities::{Zero};

use std::fs::{File};
use std::io::Write;

type Edge = (usize, usize);

#[derive(Debug)]
pub struct Mesh {
    vertices: Vec<Vec3>,
    triangles: Vec<[usize;3]>
}

impl Mesh {
    pub fn write_obj_file(&self, filename: &str) -> Result<(), Error> {
        let mut file = File::create(filename)?;
        for Vec3(v) in self.vertices.iter() {
            writeln!(&mut file, "v {} {} {}", v[0], v[1], v[2])?;
        }
        for t in self.triangles.iter() {
            writeln!(&mut file, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)?;
        }
        Ok(())
    }

    pub fn write_hdf5(&self, group: &hdf5::Group) -> Result<(), Error> {
        group.new_dataset::<Vec3>()
            .shape([self.vertices.len()])
            .create("vertices")?
            .write(&self.vertices)?;
        group.new_dataset::<[usize;3]>()
            .shape([self.triangles.len()])
            .create("triangles")?
            .write(&self.triangles)?;
        Ok(())
    }
}

/// Decomposition of the cube into six tetrahedra. The
/// vertices are numbered to their binary coordinates.
/// Each tetrahedron has the same shape, their edges lining
/// up with neighbouring volume elements.
const CUBE_CELLS: [[usize;4];6] =
    [ [ 2, 1, 0, 5 ]
    , [ 2, 0, 4, 5 ]
    , [ 2, 4, 6, 5 ]
    , [ 2, 6, 7, 5 ]
    , [ 2, 7, 3, 5 ]
    , [ 2, 3, 1, 5 ] ];

pub trait Oracle {
    fn grid_shape(&self) -> [usize;3];
    fn stencil(&self, x: [usize;3]) -> [f64;8];
    fn intersect(&self, y: f64, a: [usize;3], b: [usize;3]) -> Vec3;
}

/* fn grid_pos(ix: [usize;3]) -> Vec3 {
    Vec3([ix[0] as f64, ix[1] as f64, ix[2] as f64])
} */

fn grid_pos(ix: [usize;3]) -> Vec3 {
    Vec3([ix[0] as f64, ix[1] as f64, ix[2] as f64])
}

impl<'a> Oracle for ArrayView3<'a, f64> {
    fn grid_shape(&self) -> [usize;3] {
        let s = self.shape();
        [s[0], s[1], s[2]]
    }
    fn stencil(&self, x: [usize;3]) -> [f64;8] { 
        stencil::flat_2x2x2(self, x)
    }
    fn intersect(&self, y: f64, a: [usize;3], b: [usize;3]) -> Vec3 {
        let loc = (y - self[a]) / (self[b] - self[a]);
        grid_pos(a) + (grid_pos(b) - grid_pos(a)) * loc
    }
}

fn intersect_tetrahedron(fx: &[f64;8], y: f64, vertices: &[usize;4], triangles: &mut Vec<[Edge;3]>)
{
    let mut push_triangle = |a1: usize, a2: usize, b1: usize, b2: usize, c1: usize, c2: usize| {
        triangles.push([ (vertices[a1], vertices[a2])
                       , (vertices[b1], vertices[b2])
                       , (vertices[c1], vertices[c2]) ]);
    };

    let mut case: u8 = 0x00;

    if fx[vertices[0]] > y { case |= 0x01; }
    if fx[vertices[1]] > y { case |= 0x02; }
    if fx[vertices[2]] > y { case |= 0x04; }
    if fx[vertices[3]] > y { case |= 0x08; }

    match case {
        0x00 | 0x0F => {},
        0x01 | 0x0E => { push_triangle(0, 1, 0, 2, 0, 3); },
        0x02 | 0x0D => { push_triangle(1, 0, 1, 2, 1, 3); },
        0x03 | 0x0C => { push_triangle(0, 2, 0, 3, 1, 2);
                         push_triangle(1, 3, 1, 2, 0, 3); },
        0x04 | 0x0B => { push_triangle(2, 1, 2, 0, 2, 3); },
        0x05 | 0x0A => { push_triangle(0, 1, 0, 3, 1, 2);
                         push_triangle(2, 3, 2, 1, 0, 3); },
        0x06 | 0x09 => { push_triangle(1, 0, 1, 3, 2, 0); 
                         push_triangle(2, 3, 2, 0, 1, 3); },
        0x07 | 0x08 => { push_triangle(3, 1, 3, 2, 3, 0); },
        _           => panic!("system error")
    }
}

fn level_set_element(fx: &[f64;8], y: f64) -> Vec<[Edge;3]> {
    let mut triangles = Vec::<[Edge;3]>::new();
    for tet in CUBE_CELLS {
        intersect_tetrahedron(fx, y, &tet, &mut triangles);
    }
    triangles
}

type ProtoVertex = ([usize;3], [usize;3]);

fn offset_edge_tripple(shape: [usize;3], ix: [usize;3], et: [Edge;3]) -> [ProtoVertex;3] {
    let offset = |i: usize| -> [usize;3] {
        [ (ix[0] + ((i & 0x4) >> 2)) % shape[0]
        , (ix[1] + ((i & 0x2) >> 1)) % shape[1]
        , (ix[2] + ((i & 0x1)     )) % shape[2] ]
    };

    et.map(|e| (offset(e.0), offset(e.1)))
/*    let mut result = [([0;3], [0;3]);3];
    for k in 0..3 {
        result[k] = (offset(et[k].0), offset(et[k].1));
    }
    result */
}

/*
fn intersect_edge(f: &ArrayView3<f64>, y: f64, a: [usize;3], b: [usize;3]) -> Vec3 {
    let loc = (y - f[a]) / (f[b] - f[a]);
    grid_pos(a) + (grid_pos(b) - grid_pos(a)) * loc
}*/

pub fn level_set<O: Oracle>(f: &O, y: f64) -> Mesh {
    use std::collections::BTreeMap;

    let mut proto_triangles = Vec::<[ProtoVertex;3]>::new(); // <[ProtoVertex;3]>::new();

    for ix in indices(f.grid_shape()) {
        let index = [ix.0, ix.1, ix.2];
        let fx = f.stencil(index); // stencil::flat_2x2x2(f, index);
        level_set_element(&fx, y).iter().for_each(
            |e| proto_triangles.push(offset_edge_tripple(f.grid_shape(), index, *e)));
    }

    let mut proto_vertices = BTreeMap::new();
    let mut triangles = Vec::with_capacity(proto_triangles.len());
    let mut get_index = |edge: ProtoVertex| -> usize {
        let s = proto_vertices.len();
        if proto_vertices.contains_key(&edge) {
            proto_vertices[&edge]
        } else {
            proto_vertices.insert(edge, s);
            s
        }
        // proto_vertices.try_insert(edge, s).unwrap_or_else(|e| e.value)
    };

    for [a, b, c] in proto_triangles {
        triangles.push([get_index(a), get_index(b), get_index(c)]);
    }

    let mut vertices = Vec::new();
    vertices.resize(proto_vertices.len(), Vec3([0.,0.,0.]));
    for ((a, b), i) in proto_vertices.iter() {
        vertices[*i] = f.intersect(y, *a, *b);
    }

    Mesh {
        triangles: triangles,
        vertices: vertices
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_set() {
        use ndarray::arr3;
        let f = arr3(&[[[0., 1., 0.],
                        [1., 2., 1.],
                        [0., 1., 0.]],
                       [[1., 2., 1.],
                        [2., 4., 2.],
                        [1., 2., 1.]],
                       [[0., 1., 0.],
                        [1., 2., 1.],
                        [0., 1., 0.]]]);
        let m = level_set(&f.view(), 3.0);
        // m.write_obj_file("test.obj");
        println!("{:?}", m);
    }
}

