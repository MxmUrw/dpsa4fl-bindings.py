
use std::ffi::CString;

use pyo3::{prelude::*, types::PyCapsule};
use dpsa4fl::{*, controller::{api__new_controller_state, ControllerState, api__create_session}, core::{CommonState_Parametrization, Locations}};
use url::Url;
use anyhow::Result;
use tokio::runtime::Runtime;

#[pyfunction]
fn controller_api__new_state() -> Result<Py<PyCapsule>>
{
    let p = CommonState_Parametrization {
        location: Locations {
            leader: Url::parse("http://127.0.0.1:9981")?, // .map_err(|e| PyErr::new(e.to_string()))?,
            helper: Url::parse("http://127.0.0.1:9982")?, // .map_err(|e| PyErr::new(e.to_string()))?,
        },
        gradient_len: 16,
    };
    let s = api__new_controller_state(p);
    let name = CString::new("foo").unwrap();
    let capsule : Py<PyCapsule> = Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, s, Some(name));
        capsule.map(|c| c.into())
    })?;

    // let s = PyCapsule::new()
    Ok(capsule)
}

// #[pyfunction]
// fn sleep_for<'p>(py: Python<'p>, secs: &'p PyAny, c: &'p PyAny) -> PyResult<&'p PyAny> {
//     let secs = secs.extract()?;
//     let c2 : Py<PyCapsule> = c.extract()?;
//     pyo3_asyncio::tokio::future_into_py_with_locals(
//         py,
//         pyo3_asyncio::tokio::get_current_locals(py)?,
//         async move {
//             tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
//             Python::with_gil(|py| Ok(py.None()))
//         }
//     )
// }

// #[pyfunction]
// fn private_create_session<'a>(py: Python<'a>, controller_state: &'a Py<PyCapsule>) -> Result<&'a PyAny>
// {
//     // Python::with_gil(|py| async move {

//     pyo3_asyncio::tokio::future_into_py_with_locals(
//         py,
//         pyo3_asyncio::tokio::get_current_locals(py)?,
//         async move
//         {
//             Python::with_gil(|py| {
//             let state : &ControllerState = unsafe {controller_state.as_ref(py).reference()};
//             api__create_session(state)
//             }).await?;
//             Ok(())
//         }
//     )
//     .map_err(|e| e.into())
//     // }).await
// }

#[pyfunction]
fn controller_api__create_session(py: Python, controller_state: Py<PyCapsule>) -> Result<()>
{
    Python::with_gil(|py| {
        let state : &ControllerState = unsafe {controller_state.as_ref(py).reference()};
        // execute async function in tokio runtime
        Runtime::new().unwrap().block_on(api__create_session(state))
    })
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
    m.add_function(wrap_pyfunction!(controller_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__create_session, m)?)?;
    // m.add_function(wrap_pyfunction!(private_create_session, m)?)?;
    m.add_function(wrap_pyfunction!(test_read_uri, m)?)?;
    Ok(())
}

