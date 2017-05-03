use matrix::Matrix;

pub trait LeastSquares {
    fn least_squares(&self, y: &Matrix) -> Matrix;
}

impl LeastSquares for Matrix {
    fn least_squares(&self, y: &Matrix) -> Matrix {
        assert_eq!(self.nrows(), y.nrows());
        assert_eq!(y.ncols(), 1);

        let xaug = Matrix::ones(self.nrows(), 1).hcat(self);

        xaug.gram_solve(&(xaug.t() * y))
    }
}



#[cfg(test)]
mod tests {
    use std::path::{PathBuf};
    use std::f64;

    use etl::DataFrame;

    use matrix::SubMatrix;

    use super::*;

    #[test]
    fn test_regression() {
        let data_dir_pathbuf = PathBuf::from(file!()) // current file
                .parent().unwrap() // "src" directory
                .parent().unwrap() // crate root directory;
                .join("test_data");

        let data_file_path = data_dir_pathbuf.join("matrix_test.csv");
        let config_file_path = data_dir_pathbuf.join("matrix_test.yaml");

        let (_, df) = DataFrame::load(&config_file_path, &data_file_path).unwrap();
        assert_eq!(df.nrows(), 100);

        let (fieldnames, mat) = df.as_matrix().unwrap();
        println!("{:?}", fieldnames);
        assert_eq!(fieldnames.len(), 2);
        assert_eq!(mat.nrows(), 100);
        assert_eq!(mat.ncols(), 2);
        // TODO: find a better way to extract based on column names
        let posx = fieldnames.iter().position(|s| *s == "x").unwrap();
        let posy = fieldnames.iter().position(|s| *s == "y").unwrap();
        let x = mat.subm(.., posx).unwrap();
        let y = mat.subm(.., posy).unwrap();
        println!("{:?}", x);
        println!("{:?}", y);

        let soln = x.least_squares(&y);

        println!("{:?}", soln);
        assert_eq!(soln.nrows(), 2);
        assert_eq!(soln.ncols(), 1);
        // TODO: standardize approx asserts
        assert!((soln - mat![-0.82637; 1.66524]).iter()
            .fold(f64::NEG_INFINITY, |acc, f| acc.max(f.abs())) < 0.00001);

    }

}
