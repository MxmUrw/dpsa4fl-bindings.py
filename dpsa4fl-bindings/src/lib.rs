
use pyo3::prelude::*;
use dpsa4fl::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b + b).to_string())
}

#[pyfunction]
fn call_main()
{
    dpsa4fl::main();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dpsa4fl_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(call_main, m)?)?;
    Ok(())
}

