
use std::ops::{Index, IndexMut, Add, AddAssign, Mul, MulAssign};
use std::clone::Clone;

#[derive(Clone)]
pub struct Vector {
    pub m: usize,
    data: Vec<f32>
}


impl Vector {
    pub fn new(m: usize) -> Vector {
        let mut data: Vec<f32> = Vec::with_capacity(m);
        for _ in 0..m {
            data.push(0.0);
        }
        Vector { m, data }
    }

    pub fn zero(&mut self) {
        for x in &mut self.data {
            *x = 0f32;
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn norm(&self) -> f32 {
        self.data.iter().fold(0f32, |acc, x| acc + (x * x)).sqrt()
    }

    pub fn add_vector(&mut self, v: Vector, a: f32) {
        if self.m != v.m {
            panic!("Cannot add vectors with mismatched dimensions");
        }

        for i in 0..v.len() {
            self.data[i] += a * v[i];
        }
    }

    pub fn argmax(&self) -> usize {
        let mut max = self[0];
        let mut imax = 0usize;

        for i in 0..self.m {
            if self[i] > max {
                imax = i;
                max = self[i];
            }
        }
        imax
    }
}


impl Index<usize> for Vector {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &f32 {
        &(self.data[index])
    }
}


impl IndexMut<usize> for Vector {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut(self.data[index])
    }
}


impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Vector) -> Vector {
        assert_eq!(self.m, rhs.m, "Cannot add vectors with mismatched dimensions");


        let mut data = Vec::with_capacity(self.m);
        for i in 0..self.len() {
            data.push(self[i] + rhs[i]);
        }

        Vector { m: self.len(), data }
    }
}


impl AddAssign for Vector {
    fn add_assign(&mut self, other: Vector) {
        if self.m != other.m {
            panic!("Cannot add vectors with mismatched dimensions");
        }

        for i in 0..self.len() {
            (*self).data[i] += other[i];
        }
    }
}


impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f32) -> Vector {
        let mut data = Vec::with_capacity(self.m);
        for i in 0..self.len() {
            data.push(self[i] * rhs);
        }

        Vector { m: self.len(), data }
    }
}


impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..self.len() {
            (*self).data[i] *= rhs;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::Vector;

    #[test]
    fn can_make_vector() {
        let v = Vector::new(3);
        assert_eq!(3, v.m);
    }

    #[test]
    fn can_index_into_vector() {
        let mut v = Vector::new(3);

        for i in 0..v.len() {
            v[i] = (i as f32) + 1.0;
        }

        assert_eq!(1.0, v[0]);
        assert_eq!(2.0, v[1]);
        assert_eq!(3.0, v[2]);

        let y = v;

        assert_eq!(1.0, y[0]);
        assert_eq!(2.0, y[1]);
        assert_eq!(3.0, y[2]);
    }

    #[test]
    fn can_scale_vector() {
        let mut u = Vector::new(3);
        &u.data.iter_mut().for_each(|x| *x = 3.0);
        let mut u1 = u.clone();

        let v = u * 4.0;
        assert_eq!(12.0, v[0]);
        assert_eq!(12.0, v[1]);
        assert_eq!(12.0, v[2]);

        u1 *= 5.0;
        assert_eq!(15.0, u1[0]);
        assert_eq!(15.0, u1[1]);
        assert_eq!(15.0, u1[2]);
    }

    #[test]
    fn can_get_norm() {
        let mut v = Vector::new(2);
        v[0] = 3.0;
        v[1] = 4.0;

        assert_eq!(5.0, v.norm());
    }

    #[test]
    fn can_add_vectors() {
        let mut v1 = Vector::new(3);
        v1.data.iter_mut().for_each(|x| *x = 1.0);
        let mut v3 = v1.clone();

        let mut v2 = Vector::new(3);
        v2.data.iter_mut().for_each(|x| *x = 2.0);
        let v4 = v2.clone();

        let v12 = v1 + v2;
        assert_eq!(3.0, v12[1]);

        v3 += v4;
        assert_eq!(3.0, v3[1]);
    }

    #[test]
    fn can_add_and_scale_vectors() {
        let mut v1 = Vector::new(3);
        v1.data.iter_mut().for_each(|x| *x = 1.0);

        let mut v2 = Vector::new(3);
        v2.data.iter_mut().for_each(|x| *x = 2.0);

        let v12 = v1 + v2;
        assert_eq!(3.0, v12[1]);
    }

    #[test]
    #[should_panic(expected = "Cannot add vectors with mismatched dimensions")]
    fn cannot_add_vectors_of_mismatched_lengths() {
        let v1 = Vector::new(3);
        let v2 = Vector::new(4);
        let _v12 = v1 + v2;
    }

    #[test]
    fn can_get_argmax() {
        let mut v = Vector::new(3);
        v[0] = 8.0;
        v[1] = -3.5;
        v[2] = 45.0;

        assert_eq!(2, v.argmax());
    }
}
