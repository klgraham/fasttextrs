
use std::ops::{Index, IndexMut, Mul};
use vector::Vector;


struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f32>
}


impl Matrix {
    pub fn new(nrows: usize, ncols: usize) -> Matrix {
        let mut data: Vec<f32> = Vec::with_capacity(nrows * ncols);
        for _ in 0..nrows*ncols {
            data.push(0.0);
        }

        Matrix { nrows, ncols, data }
    }

//    pub fn zero(&mut self) {
//
//    }
//
    // row i of matrix times v
    pub fn dot_row(&self, v: Vector, i: usize) -> f32 {
        assert!(i >= 0usize && i < self.nrows, "Index out of bounds.");
        assert_eq!(v.len(), self.ncols, "Matrix and Vector dimensions are incompatible.");

        (0..self.ncols).fold(0f32, |acc, j| acc + self[(i,j)] * v[j])
    }

    // row i of matrix plus a*v
    pub fn add_vector_and_scale_row(&mut self, v: Vector, i: usize, a: f32) {
        assert!(i >= 0usize && i < self.nrows, "Index out of bounds.");
        assert_eq!(v.len(), self.ncols, "Matrix and Vector dimensions are incompatible.");

        (0..self.ncols).for_each(|j| self[(i,j)] += a * v[j]);
    }
}


impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        assert!(i < self.nrows && j < self.ncols, "Matrix subscript out of bounds.");
        &(self.data[i * self.ncols + j])
    }
}


impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f32 {
        assert!(i < self.nrows && j < self.ncols, "Matrix subscript out of bounds.");
        &mut (self.data[i * self.ncols + j])
    }
}


#[cfg(test)]
mod tests {
    use super::Matrix;
    use vector::Vector;

    #[test]
    fn can_create_matrix() {
        let m = Matrix::new(3, 2);
        assert_eq!(3, m.nrows);
        assert_eq!(2, m.ncols);
    }

    #[test]
    fn can_index_into_matrix() {
        let mut m = Matrix::new(3, 2);
        m[(2,1)] = 3.4;
        assert_eq!(3.4, m[(2,1)]);
    }

    #[test]
    fn can_dot_with_vector() {
        let mut v = Vector::new(3);
        (0..v.len()).for_each(|i| v[i] = 3.0);

        let mut m = Matrix::new(3, 3);
        m[(0,0)] = -1.0;
        m[(1,1)] = -1.0;
        m[(2,2)] = -1.0;

        let dot = m.dot_row(v, 0);
        assert_eq!(dot, -3.0);
    }

    #[test]
    fn can_add_vector_and_scale_row() {
        let mut v = Vector::new(3);
        (0..v.len()).for_each(|i| v[i] = 3.0);

        let mut m = Matrix::new(3, 3);
        m[(0,0)] = -1.0;
        m[(0,1)] = 2.0;

        m.add_vector_and_scale_row(v, 0, 7.0);
        assert_eq!(m[(0,0)], 20.0);
        assert_eq!(m[(0,1)], 23.0);
    }
}