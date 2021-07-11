use crate::numeric::{Vec3};
use crate::error::{Error};

use std::fs::{File};
use std::io::Write;

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[usize;3]>
}

impl Mesh {
    pub fn write_obj_file(&self, filename: &str, box_size: f64) -> Result<(), Error> {
        let mut file = File::create(filename)?;
        for v in self.vertices.iter() {
            writeln!(&mut file, "v {} {} {}", v.0[0], v.0[1], v.0[2])?;
        }
        'next_triangle: for t in self.triangles.iter() {
            for k in 0..3 {
                for j in 0..k {
                    if (self.vertices[t[k]].clone() - self.vertices[t[j]].clone()).0
                        .iter().any(|a| (*a).abs() > box_size/2.) {
                        continue 'next_triangle;
                    }
                }
            }
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

