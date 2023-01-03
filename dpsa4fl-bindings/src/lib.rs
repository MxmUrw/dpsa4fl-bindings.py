
use crate::core::PyControllerState;
use crate::core::PyControllerState_Mut;
use crate::core::PyMeasurement;
use std::ffi::CString;

use dpsa4fl::client::ClientStatePU;
use dpsa4fl::client::Fx;
use dpsa4fl::client::Measurement;
use dpsa4fl::client::RoundSettings;
use dpsa4fl::client::api__new_client_state;
use dpsa4fl::client::api__submit;
use dpsa4fl::controller::api__collect;
use dpsa4fl::controller::api__start_round;
use dpsa4fl::core::Locations;
use fixed::traits::LossyInto;
use ndarray::ArrayD;
use ndarray::ArrayViewD;
use numpy::IxDyn;
use numpy::PyArray;
use numpy::PyArray1;
use numpy::PyArrayDyn;
use numpy::PyReadonlyArrayDyn;
use numpy::ToPyArray;
use pyo3::with_embedded_python_interpreter;
use pyo3::{prelude::*, types::PyCapsule};
use dpsa4fl::{*, controller::{api__new_controller_state, ControllerState_Mut, ControllerState_Immut, api__create_session, ControllerState_Permanent}, core::{CommonState_Parametrization}};
use url::Url;
use anyhow::{anyhow, Result};
use tokio::runtime::Runtime;

use fixed::types::extra::{U15, U31, U63};
use fixed::{FixedI16, FixedI32, FixedI64};
use fixed_macro::fixed;

pub mod core;

/////////////////////////////////////////////////////////////////
// Param

/// Create new parametrization object for local testing.
fn get_common_state_parametrization(gradient_len: usize) -> Result<CommonState_Parametrization>
{
    let res = CommonState_Parametrization {
        location: Locations {
            external_leader_main: Url::parse("http://127.0.0.1:9991")?,
            external_helper_main: Url::parse("http://127.0.0.1:9992")?,
            external_leader_tasks: Url::parse("http://127.0.0.1:9981")?,
            external_helper_tasks: Url::parse("http://127.0.0.1:9982")?,
            internal_leader: Url::parse("http://aggregator1:9991")?,
            internal_helper: Url::parse("http://aggregator2:9992")?,
        },
        gradient_len,
    };
    Ok(res)
}

/////////////////////////////////////////////////////////////////
// Client api

#[pyclass]
struct PyClientState
{
    mstate: ClientStatePU,
}


/// Create new client state.
#[pyfunction]
fn client_api__new_state(gradient_len: usize) -> Result<PyClientState>
{
    let p = get_common_state_parametrization(gradient_len)?;
    let res = PyClientState {
        mstate: api__new_client_state(p)
    };
    Ok(res)
}

fn array_to_vec<A>(xs: ArrayViewD<A>) -> Vec<A>
    where A : Clone
{
    let mut ys = Vec::new();
    ys.reserve_exact(xs.len());
    for x in xs
    {
        ys.push(x.clone())
    }
    ys
}

fn float_to_fixed(x: &f32) -> Fx
{
    Fx::from_num(*x)
}

/// Submit a gradient vector to a janus server.
///
/// This function takes a `task_id` to identify the janus task to which this gradient corresponds.
#[pyfunction]
fn client_api__submit(client_state: Py<PyClientState>, task_id: String, data: PyReadonlyArrayDyn<f32>) -> Result<()>
{
    //----
    // prepare data for prio
    let data: ArrayViewD<f32> = data.as_array();
    let shape = data.shape();
    assert!(shape.len() == 1, "Expected the data passed to submit to be 1-dimensional. But it was {shape:?}");

    let data = array_to_vec(data);
    let data : Vec<Fx> = data.iter().map(float_to_fixed).collect();

    //----
    let round_settings = RoundSettings::new(task_id)?;

    Python::with_gil(|py| {
        let state_cell: &PyCell<PyClientState> = client_state.as_ref(py);
        let mut state_ref_mut = state_cell.try_borrow_mut().map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyClientState = &mut *state_ref_mut;

        let actual_len = data.len();
        let expected_len = state.mstate.get_parametrization().gradient_len;
        assert!(actual_len == expected_len, "Expected data to be have length {expected_len} but it was {actual_len}");

        let res = Runtime::new().unwrap().block_on(api__submit(&mut state.mstate, round_settings, &data))?;

        Ok(res)
    })
}

/////////////////////////////////////////////////////////////////
// Controller api

/// Create new controller state.
#[pyfunction]
fn controller_api__new_state(gradient_len: usize) -> Result<PyControllerState>
{
    let p = get_common_state_parametrization(gradient_len)?;
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


/// Helper function to access the number of parameters expected by janus.
#[pyfunction]
fn controller_api__get_gradient_len(controller_state: Py<PyControllerState>) -> Result<usize>
{
    run_on_controller(
        controller_state,
        |i,m| Ok(i.parametrization.gradient_len)
    )
}

/// Run a function on controller state.
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

        // execute async function in tokio runtime
        let res = f(istate, &mut mstate)?;

        // write result into state
        state.mstate = mstate.into();

        Ok(res)
    })
}

/// Create a new training session.
#[pyfunction]
fn controller_api__create_session(controller_state: Py<PyControllerState>) -> Result<u16>
{
    run_on_controller(
        controller_state,
        |i,m| Runtime::new().unwrap().block_on(api__create_session(i, m))
    )
}

/// Start a new training round.
#[pyfunction]
fn controller_api__start_round(controller_state: Py<PyControllerState>) -> Result<String>
{
    run_on_controller(
        controller_state,
        |i,m| Runtime::new().unwrap().block_on(api__start_round(i, m))
    )
}

/// Collect resulting aggregated gradient vector from janus.
#[pyfunction]
fn controller_api__collect(py: Python, controller_state: Py<PyControllerState>) -> Result<&PyArray1<f64>>
{
    let res = run_on_controller(
        controller_state,
        |i,m| Runtime::new().unwrap().block_on(api__collect(i, m))
    )?;

    let vector = res.aggregate_result();

    Ok(vector.to_pyarray(py))
}



/// The python module definition.
#[pymodule]
fn dpsa4fl_bindings(_py: Python, m: &PyModule) -> PyResult<()>
{
    // add class
    m.add_class::<PyControllerState>()?;
    m.add_class::<PyControllerState_Mut>()?;

    // add functions
    //--- controller api ---
    m.add_function(wrap_pyfunction!(controller_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__create_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__start_round, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__collect, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__get_gradient_len, m)?)?;
    //--- client api ---
    m.add_function(wrap_pyfunction!(client_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(client_api__submit, m)?)?;

    Ok(())
}

