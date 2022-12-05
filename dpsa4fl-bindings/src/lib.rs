
use crate::core::PyControllerState;
use crate::core::PyControllerState_Mut;
use std::ffi::CString;

use dpsa4fl::controller::api__start_round;
use pyo3::with_embedded_python_interpreter;
use pyo3::{prelude::*, types::PyCapsule};
use dpsa4fl::{*, controller::{api__new_controller_state, ControllerState_Mut, ControllerState_Immut, api__create_session, ControllerState_Permanent}, core::{CommonState_Parametrization, Locations}};
use url::Url;
use anyhow::{anyhow, Result};
use tokio::runtime::Runtime;

pub mod core;

#[pyfunction]
fn controller_api__new_state() -> Result<PyControllerState>
{
    let p = CommonState_Parametrization {
        location: Locations {
            external_leader: Url::parse("http://127.0.0.1:9981")?, // .map_err(|e| PyErr::new(e.to_string()))?,
            external_helper: Url::parse("http://127.0.0.1:9982")?, // .map_err(|e| PyErr::new(e.to_string()))?,
            internal_leader: Url::parse("http://aggregator1:9991")?,
            internal_helper: Url::parse("http://aggregator2:9992")?,
        },
        gradient_len: 16,
    };
    let istate = api__new_controller_state(p);
    let istate : Py<PyCapsule> = Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, istate, None);
        capsule.map(|c| c.into())
    }).unwrap();

    let mstate = PyControllerState_Mut {
        training_session_id: None,
        task_id: None
    };

    let res = PyControllerState {
        mstate,
        istate,
    };

    Ok(res)
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

fn run_on_controller<A>
    (
        controller_state: Py<PyControllerState>,
        f: fn(&ControllerState_Immut, &mut ControllerState_Mut) -> Result<A>,
    )
    -> Result<A>
{
    Python::with_gil(|py| {
        let state_cell: &PyCell<PyControllerState> = controller_state.as_ref(py);
        let mut state_ref_mut = state_cell.try_borrow_mut().map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyControllerState = &mut *state_ref_mut;

        let istate : &ControllerState_Immut = unsafe {state.istate.as_ref(py).reference()};
        let mut mstate : ControllerState_Mut = state.mstate.clone().try_into()?;
        // let mut mut_state: ControllerState = state.clone();
        // execute async function in tokio runtime
        let res = f(istate, &mut mstate)?;

        // write result into state
        state.mstate = mstate.into();

        Ok(res)
    })
}

#[pyfunction]
fn controller_api__create_session(controller_state: Py<PyControllerState>) -> Result<u16>
{
    run_on_controller(
        controller_state,
        |i,m| Runtime::new().unwrap().block_on(api__create_session(i, m))
    )

    // Python::with_gil(|py| {
    //     let state_cell: &PyCell<PyControllerState> = controller_state.as_ref(py);
    //     let mut state_ref_mut = state_cell.try_borrow_mut().map_err(|_| anyhow!("could not get mut ref"))?;
    //     let state: &mut PyControllerState = &mut *state_ref_mut;

    //     let istate : &ControllerState_Immut = unsafe {state.istate.as_ref(py).reference()};
    //     let mut mstate : ControllerState_Mut = state.mstate.clone().try_into()?;
    //     // let mut mut_state: ControllerState = state.clone();
    //     // execute async function in tokio runtime
    //     let res = Runtime::new().unwrap().block_on(api__create_session(istate, &mut mstate))?;

    //     // write result into state
    //     state.mstate = mstate.into();

    //     Ok(res)
    // })
}

#[pyfunction]
fn controller_api__start_round(controller_state: Py<PyControllerState>) -> Result<String>
{
    run_on_controller(
        controller_state,
        |i,m| Runtime::new().unwrap().block_on(api__start_round(i, m))
    )
}


// #[pyfunction]
// fn test_read_uri(capsule: Py<PyCapsule>) -> Result<()>
// {
//     Python::with_gil(|py| {
//         unsafe {
//             let c: &ControllerState = capsule.as_ref(py).reference();
//             println!("The uri of leader is {}", c.parametrization.location.external_leader);
//             Ok(())
//         }
//     })
// }

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
    // add class
    m.add_class::<PyControllerState>()?;
    m.add_class::<PyControllerState_Mut>()?;

    // add functions
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(call_main, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__create_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__start_round, m)?)?;
    // m.add_function(wrap_pyfunction!(private_create_session, m)?)?;
    // m.add_function(wrap_pyfunction!(test_read_uri, m)?)?;
    Ok(())
}

