
use std::ffi::CString;

use pyo3::{prelude::*, types::PyCapsule};
use dpsa4fl::{*, controller::{api__new_controller_state, ControllerState}, core::{CommonState_Parametrization, Locations}};
use url::Url;
use anyhow::Result;

#[pyfunction]
fn controller_api__new_controller_state() -> Result<Py<PyCapsule>>
{
    let p = CommonState_Parametrization {
        location: Locations {
            leader: Url::parse("localhost:9991")?, // .map_err(|e| PyErr::new(e.to_string()))?,
            helper: Url::parse("localhost:9992")?, // .map_err(|e| PyErr::new(e.to_string()))?,
        },
        gradient_len: 16,
    };
    let s = api__new_controller_state(p);
    let name = CString::new("foo").unwrap();
    let capsule : Py<PyCapsule> = Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, s, &name);
        capsule.map(|c| c.into())
    })?;

    // let s = PyCapsule::new()
    Ok(capsule)
}

#[pyfunction]
fn test_read_uri(capsule: Py<PyCapsule>) -> Result<()>
{
    Python::with_gil(|py| {
        unsafe {
            let c: &ControllerState = capsule.as_ref(py).reference();
            println!("The uri of leader is {}", c.parametrization.location.leader);
            Ok(())
        }
    })
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String>
{
    Ok((a + b + b).to_string())
}

#[pyfunction]
fn call_main()
{
    dpsa4fl::main();
}

/// A Python module implemented in Rust.
#[pymodule]
fn dpsa4fl_bindings(_py: Python, m: &PyModule) -> PyResult<()>
{
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(call_main, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__new_controller_state, m)?)?;
    m.add_function(wrap_pyfunction!(test_read_uri, m)?)?;
    Ok(())
}

