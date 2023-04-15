
use crate::core::PyControllerState;
use crate::core::PyControllerState_Mut;
use dpsa4fl::controller::api__end_session;
// use crate::core::fixed_to_float_ceil_any;
use dpsa4fl_janus_tasks::fixed::float_to_fixed_floor;
use dpsa4fl_janus_tasks::fixed::float_to_fixed_ceil;

use anyhow::{anyhow, Result};
use dpsa4fl::client::api__new_client_state;
use dpsa4fl::client::api__submit_with;
use dpsa4fl::client::api__update_client_round_settings;
use dpsa4fl::client::ClientStatePU;
use dpsa4fl::client::RoundSettings;
use dpsa4fl::controller::api__collect;
use dpsa4fl::controller::api__start_round;
use dpsa4fl::core::FixedAny;
use dpsa4fl::core::VecFixedAny;
use dpsa4fl_janus_tasks::core::VdafParameter;
use dpsa4fl_janus_tasks::fixed::FixedTypeTag;
// use dpsa4fl::core::FixedTypeTag;
// use dpsa4fl::core::Fx;
use dpsa4fl::core::Locations;
use dpsa4fl::{
    controller::{
        api__create_session, api__new_controller_state, ControllerState_Immut, ControllerState_Mut,
    },
    core::CommonState_Parametrization,
};
use fixed::traits::Fixed;
use fraction::GenericFraction;
use ndarray::ArrayViewD;
use numpy::PyArray1;
use numpy::PyReadonlyArrayDyn;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyCapsule};
use tokio::runtime::Runtime;
use url::Url;

pub mod core;

/////////////////////////////////////////////////////////////////
// Param

/// Create new parametrization object for local testing.
/*
fn get_common_state_parametrization(
    gradient_len: usize,
    privacy_parameter: f32,
    submission_type: FixedTypeTag,
    locations: Locations,
) -> Result<CommonState_Parametrization> {

    let vdaf_parameter = VdafParameter
    {
        gradient_len,
        privacy_parameter: todo!()
    };

    let res = CommonState_Parametrization {
        location: locations,
        // location: Locations {
        //     external_leader_main: Url::parse("http://127.0.0.1:9991")?,
        //     external_leader_tasks: Url::parse("http://127.0.0.1:9981")?,
        //     external_helper_main: Url::parse("http://127.0.0.1:9992")?,
        //     external_helper_tasks: Url::parse("http://127.0.0.1:9982")?,
        // },
        vdaf_parameter: vdaf_parameter,
        // gradient_len,
        // privacy_parameter: float_to_fixed_ceil(&privacy_parameter),
        // submission_type,
    };
    Ok(res)
}
*/

/////////////////////////////////////////////////////////////////
// Client api

#[pyclass]
struct PyClientState {
    mstate: ClientStatePU,
}

/// Create new client state.
#[pyfunction]
fn client_api__new_state(
    external_leader_main: String,
    external_leader_tasks: String,
    external_helper_main: String,
    external_helper_tasks: String,
) -> Result<PyClientState> {

    let l = Locations {
        external_leader_main: Url::parse(&external_leader_main)?,
        external_leader_tasks: Url::parse(&external_leader_tasks)?,
        external_helper_main: Url::parse(&external_helper_main)?,
        external_helper_tasks: Url::parse(&external_helper_tasks)?,
    };

    let res = PyClientState {
        mstate: api__new_client_state(l),
    };

    Ok(res)
}

fn array_to_vec<A>(xs: ArrayViewD<A>) -> Vec<A>
where
    A: Clone,
{
    let mut ys = Vec::new();
    ys.reserve_exact(xs.len());
    for x in xs {
        ys.push(x.clone())
    }
    ys
}



#[pyfunction]
fn client_api__get_privacy_parameter(
    client_state: Py<PyClientState>,
    task_id: Option<String>,
) -> Result<f32> {

    Python::with_gil(|py| {

        let state_cell: &PyCell<PyClientState> = client_state.as_ref(py);
        let mut state_ref_mut = state_cell
            .try_borrow_mut()
            .map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyClientState = &mut *state_ref_mut;


        // if we were given a task_id, we get the parameters for this task
        // from the aggregators (by writing them into the client state)
        // otherwise we assume that the client already has been registered for a round
        match task_id
        {
            Some(task_id) =>
            {
                let round_settings = RoundSettings::new(task_id)?;
                let future = api__update_client_round_settings(&mut state.mstate, round_settings);
                Runtime::new().unwrap().block_on(future)?;
            },
            None => {},
        }

        // Now try to get the privacy param
        let privacy = state.mstate.get_valid_state()
            .ok_or(anyhow!(""))
            .map(|s| s.parametrization.vdaf_parameter.privacy_parameter.clone())?;

        let privacy = (privacy.0 as f32) / (privacy.1 as f32);

        Ok(privacy)

    })
}


/// Submit a gradient vector to a janus server.
///
/// This function takes a `task_id` to identify the janus task to which this gradient corresponds.
#[pyfunction]
fn client_api__submit(
    client_state: Py<PyClientState>,
    task_id: String,
    data: PyReadonlyArrayDyn<f32>,
) -> Result<()> {
    Python::with_gil(|py| {
        //----
        // prepare data for prio
        let data: ArrayViewD<f32> = data.as_array();
        let shape = data.shape();
        assert!(
            shape.len() == 1,
            "Expected the data passed to submit to be 1-dimensional. But it was {shape:?}"
        );

        //----

        let state_cell: &PyCell<PyClientState> = client_state.as_ref(py);
        let mut state_ref_mut = state_cell
            .try_borrow_mut()
            .map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyClientState = &mut *state_ref_mut;

        let data = array_to_vec(data);
        // let data: VecFixedAny = match state.
            // data.iter().map(float_to_fixed_floor).collect();

        let round_settings = RoundSettings::new(task_id)?;
        let res = Runtime::new().unwrap().block_on(api__submit_with(
            &mut state.mstate,
            round_settings,
            |param|
            {
                println!("submitting for tag {:?}", param.vdaf_parameter.submission_type.clone());
                match param.vdaf_parameter.submission_type
                {
                    FixedTypeTag::FixedType16Bit => VecFixedAny::VecFixed16(data.into_iter().map(float_to_fixed_floor).collect()),
                    FixedTypeTag::FixedType32Bit => VecFixedAny::VecFixed32(data.into_iter().map(float_to_fixed_floor).collect()),
                    FixedTypeTag::FixedType64Bit => VecFixedAny::VecFixed64(data.into_iter().map(float_to_fixed_floor).collect()),
                }
            },
        ))?;

        Ok(res)
    })
}


/////////////////////////////////////////////////////////////////
// Controller api

/// Create new controller state.
#[pyfunction]
fn controller_api__new_state(
    gradient_len: usize,
    privacy_parameter: f32,
    fixed_bitsize: usize,
    external_leader_main: String,
    external_leader_tasks: String,
    external_helper_main: String,
    external_helper_tasks: String,
) -> Result<PyControllerState> {

    // we convert from f32 to a fraction
    let privacy_parameter_frac = GenericFraction::<u128>::from(privacy_parameter);
    let privacy_parameter = match (privacy_parameter_frac.numer(), privacy_parameter_frac.denom())
    {
        (Some(n), Some(d)) => (n.clone(),d.clone()),
        _ => Err(anyhow!("The privacy parameter {privacy_parameter_frac} is not a valid finite fraction."))?
    };

    let submission_type = match fixed_bitsize
    {
        16 => FixedTypeTag::FixedType16Bit,
        32 => FixedTypeTag::FixedType32Bit,
        64 => FixedTypeTag::FixedType64Bit,
        _  => Err(anyhow!("The bitsize {fixed_bitsize} is not supported. Only 16, 32 or 64 is."))?,
    };

    let location = Locations
    {
        external_leader_main: Url::parse(&external_leader_main)?,
        external_leader_tasks: Url::parse(&external_leader_tasks)?,
        external_helper_main: Url::parse(&external_helper_main)?,
        external_helper_tasks: Url::parse(&external_helper_tasks)?,
    };

    let vdaf_parameter = VdafParameter
    {
        gradient_len, privacy_parameter, submission_type
    };

    let p = CommonState_Parametrization
    {
        location, vdaf_parameter
    };

    let istate = api__new_controller_state(p);
    let istate: Py<PyCapsule> = Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, istate, None);
        capsule.map(|c| c.into())
    })
    .unwrap();

    let mstate = PyControllerState_Mut {
        training_session_id: None,
        task_id: None,
    };

    let res = PyControllerState { mstate, istate };

    Ok(res)
}

/// Helper function to access the number of parameters expected by janus.
#[pyfunction]
fn controller_api__get_gradient_len(controller_state: Py<PyControllerState>) -> Result<usize> {
    run_on_controller(controller_state, |i, m| Ok(i.parametrization.vdaf_parameter.gradient_len))
}

/// Run a function on controller state.
fn run_on_controller<A>(
    controller_state: Py<PyControllerState>,
    f: fn(&ControllerState_Immut, &mut ControllerState_Mut) -> Result<A>,
) -> Result<A> {
    Python::with_gil(|py| {
        let state_cell: &PyCell<PyControllerState> = controller_state.as_ref(py);
        let mut state_ref_mut = state_cell
            .try_borrow_mut()
            .map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyControllerState = &mut *state_ref_mut;

        let istate: &ControllerState_Immut = unsafe { state.istate.as_ref(py).reference() };
        let mut mstate: ControllerState_Mut = state.mstate.clone().try_into()?;

        // execute async function in tokio runtime
        let res = f(istate, &mut mstate)?;

        // write result into state
        state.mstate = mstate.into();

        Ok(res)
    })
}

/// Create a new training session.
#[pyfunction]
fn controller_api__create_session(controller_state: Py<PyControllerState>) -> Result<u16> {
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api__create_session(i, m))
    })
}

/// End the current new training session.
#[pyfunction]
fn controller_api__end_session(controller_state: Py<PyControllerState>) -> Result<()> {
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api__end_session(i, m))
    })
}

/// Start a new training round.
#[pyfunction]
fn controller_api__start_round(controller_state: Py<PyControllerState>) -> Result<String> {
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api__start_round(i, m))
    })
}

/// Collect resulting aggregated gradient vector from janus.
#[pyfunction]
fn controller_api__collect(
    py: Python,
    controller_state: Py<PyControllerState>,
) -> Result<&PyArray1<f64>> {
    let res = run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api__collect(i, m))
    })?;

    let vector = res.aggregate_result();

    Ok(vector.to_pyarray(py))
}

/// The python module definition.
#[pymodule]
fn dpsa4fl_bindings(_py: Python, m: &PyModule) -> PyResult<()> {
    // add class
    m.add_class::<PyControllerState>()?;
    m.add_class::<PyControllerState_Mut>()?;

    // add functions
    //--- controller api ---
    m.add_function(wrap_pyfunction!(controller_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__create_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__end_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__start_round, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__collect, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api__get_gradient_len, m)?)?;
    //--- client api ---
    m.add_function(wrap_pyfunction!(client_api__new_state, m)?)?;
    m.add_function(wrap_pyfunction!(client_api__submit, m)?)?;
    m.add_function(wrap_pyfunction!(client_api__get_privacy_parameter, m)?)?;

    Ok(())
}
