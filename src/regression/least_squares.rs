use matrix::Matrix;
use matrix::GramSolve;

use errors::*;

pub trait LeastSquares {
    fn least_squares(&self, y: &Matrix) -> Result<Matrix>;
}

impl LeastSquares for Matrix {
    fn least_squares(&self, y: &Matrix) -> Result<Matrix> {
        assert_eq!(self.nrows(), y.nrows());
        assert_eq!(y.ncols(), 1);

        let xaug = Matrix::ones(self.nrows(), 1).hcat(self);

        xaug.gram_solve(&(xaug.t() * y)).chain_err(|| "Failure in solving system of equations")
    }
}



#[cfg(test)]
mod tests {
    use std::path::{PathBuf};

    use etl::DataFrame;

    use super::*;

    #[test]
    fn test_regression() {
        let data_dir_pathbuf = PathBuf::from(file!()) // current file
                .parent().unwrap() // "regression" directory
                .parent().unwrap() // "src" directory
                .parent().unwrap() // crate root directory;
                .join("test_data");

        let data_file_path = data_dir_pathbuf.join("matrix_test.csv");
        let config_file_path = data_dir_pathbuf.join("matrix_test.yaml");

        let (_, df) = DataFrame::load(&config_file_path, &data_file_path).unwrap();
        assert_eq!(df.nrows(), 100);

        let (fnx, x) = df.sub(vec!["x"]).expect("dataframe sub failed").as_matrix().unwrap();
        let (fny, y) = df.sub(vec!["y"]).expect("dataframe sub failed").as_matrix().unwrap();
        println!("{:?} {:?}", fnx, fny);
        assert_eq!(fnx.len(), 1);
        assert_eq!(fny.len(), 1);
        assert_eq!((x.nrows(), x.ncols()), (100, 1));
        assert_eq!((y.nrows(), y.ncols()), (100, 1));

        println!("{:?}", x);
        println!("{:?}", y);

        let soln = x.least_squares(&y).expect("solve failed");

        println!("{:?}", soln);
        assert_eq!(soln.nrows(), 2);
        assert_eq!(soln.ncols(), 1);
        assert_fpvec_eq!(soln, mat![-0.82637; 1.66524], 1e-5);
    }
}
