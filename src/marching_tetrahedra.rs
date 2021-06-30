use crate::stencil::{flat_2x2x2};
use crate::numeric::{Vec3};

use ndarray::{Array, ArrayView, Ix1, Ix3};

struct Mesh {
    vertices: Array<Vec3, Ix1>,
    triangles: Array<Ix3, Ix1>
}

/// Decomposition of the cube into six tetrahedra. The
/// vertices are numbered to their binary coordinates.
/// Each tetrahedron has the same shape, their edges lining
/// up with neighbouring volume elements.
const CUBE_CELLS: [[usize;3];6] =
    [ [ 2, 0, 1, 5 ]
    , [ 2, 4, 0, 5 ]
    , [ 2, 6, 4, 5 ]
    , [ 2, 7, 6, 5 ]
    , [ 2, 3, 7, 5 ]
    , [ 2, 1, 3, 5 ] ];

const CUBE_EDGES: [[usize;2];19] =
    [ /*  0 */ [ 0, 1 ], [ 0, 2 ], [ 0, 4 ], [ 0, 5 ]
    , /*  4 */ [ 1, 2 ], [ 1, 3 ], [ 1, 5 ]
    , /*  7 */ [ 2, 3 ], [ 2, 4 ], [ 2, 5 ], [ 2, 6 ], [ 2, 7 ]
    , /* 12 */ [ 3, 5 ], [ 3, 7 ],
    , /* 14 */ [ 4, 5 ], [ 4, 6 ],
    , /* 16 */ [ 5, 6 ], [ 5, 7 ],
    , /* 18 */ [ 6, 7 ] ];

const CUBE_CELL_EDGES: [[usize;6];6] =
    [ [ 0,  1,  3,  4,  6,  9 ]
    , [ 1,  2,  3,  8,  9, 14 ]
    , [ 8,  9, 10, 14, 15, 16 ]
    , [ 9, 10, 11, 16, 17, 18 ]
    , [ 7,  9, 11, 12, 13, 17 ]
    , [ 4,  5,  6,  7,  9, 12 ] ];

const CUBE_VERTICES: [Vec3;8] =
    [ Vec3([ 0.0, 0.0, 0.0 ])
    , Vec3([ 1.0, 0.0, 0.0 ])
    , Vec3([ 0.0, 1.0, 0.0 ])
    , Vec3([ 1.0, 1.0, 0.0 ])
    , Vec3([ 0.0, 0.0, 1.0 ])
    , Vec3([ 1.0, 0.0, 1.0 ])
    , Vec3([ 0.0, 1.0, 1.0 ])
    , Vec3([ 1.0, 1.0, 1.0 ]) ];

fn intersect_segment(f: &[f64;8], y: f64, a: usize, b: usize) -> Option<Vec3> {
    if f[a] < y && f[b] < y || f[a] >= y && f[b] >= y {
        None
    } else {
        let loc = (y - f[a]) / (f[b] - f[a]);
        Some(CUBE_VERTICES[a] + (CUBE_VERTICES[b] - CUBE_VERTICES[a]) * loc)
    }
}

fn level_set_element(f: &[f64;8], y: f64) -> Mesh {
    
}

fn level_set(f: &ArrayView3<f64>, y: f64) -> Mesh {

}

