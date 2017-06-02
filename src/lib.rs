#[macro_use] extern crate error_chain;

#[allow(unused_imports)] #[macro_use] extern crate matrix;
#[allow(unused_imports)] #[macro_use] extern crate unittest;
extern crate etl;

mod errors {
    error_chain! {
    }
}


pub mod regression;
