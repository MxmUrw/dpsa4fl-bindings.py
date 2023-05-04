use crate::core::PyClientState;
use crate::core::PyControllerState;
use crate::core::PyControllerStateMut;
// use dpsa4fl::controller::ControllerStateImmut;
// use dpsa4fl::controller::ControllerStateMut;
// use dpsa4fl::controller::api_collect;
// use dpsa4fl::controller::api_create_session;
// use dpsa4fl::controller::api_end_session;

use dpsa4fl::client::interface::embedded::api_new_client_state;
use dpsa4fl::client::interface::embedded::api_submit_with;
use dpsa4fl::client::interface::embedded::api_update_client_round_settings;
use dpsa4fl::client::interface::types::RoundSettings;
use dpsa4fl::controller::interface::embedded::api_collect;
use dpsa4fl::controller::interface::embedded::api_create_session;
use dpsa4fl::controller::interface::embedded::api_end_session;
use dpsa4fl::controller::interface::embedded::api_new_controller_state;
use dpsa4fl::controller::interface::embedded::api_start_round;
use dpsa4fl::controller::interface::types::ControllerStateImmut;
use dpsa4fl::controller::interface::types::ControllerStateMut;
// use dpsa4fl::controller::api_new_controller_state;
// use dpsa4fl::controller::api_start_round;
use dpsa4fl::core::fixed::float_to_fixed_floor;
use dpsa4fl::core::fixed::VecFixedAny;

use anyhow::{anyhow, Result};

use dpsa4fl::core::types::Locations;
use dpsa4fl::{
    // client::{
    //     // api_new_client_state, api_submit_with, api_update_client_round_settings, RoundSettings,
    // },
    core::types::CommonStateParametrization,
};
use dpsa4fl::core::types::VdafParameter;
use dpsa4fl::core::types::TasksLocations;
use dpsa4fl::core::fixed::FixedTypeTag;

use dpsa4fl::janus_manager::interface::network::consumer::get_main_locations;
use fraction::GenericFraction;
use ndarray::ArrayViewD;
use numpy::{PyArray1, PyReadonlyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyCapsule};
use tokio::runtime::Runtime;
use url::Url;

pub mod core;

use fixed_macro::fixed;

/////////////////////////////////////////////////////////////////
// Client api

/// Create new client state, containing the aggregator locations.
///
/// Parameters
/// ----------
/// external_leader_tasks: str
///     Location of the leader aggregator server in URL format including the port.
///     For example, for a server running locally: "http://127.0.0.1:9991"
/// external_helper_tasks: str
///     Location of the helper aggregator server in URL format including the port.
///     For example, for a server running locally: "http://127.0.0.1:9992"
///
/// Returns
/// -------
/// state: PyClientState
///     A client state object containing the input locations.
#[pyfunction]
fn client_api_new_state(
    external_leader_tasks: String,
    external_helper_tasks: String,
) -> Result<PyClientState>
{
    let l = TasksLocations {
        external_leader: Url::parse(&external_leader_tasks)?,
        external_helper: Url::parse(&external_helper_tasks)?,
    };

    let res = PyClientState {
        mstate: api_new_client_state(l),
    };

    Ok(res)
}

fn array_to_vec<A>(xs: ArrayViewD<A>) -> Vec<A>
where
    A: Clone,
{
    let mut ys = Vec::new();
    ys.reserve_exact(xs.len());
    for x in xs
    {
        ys.push(x.clone())
    }
    ys
}

/// Request the privacy parameter used by the aggregator to determine the amount
/// of noise added. If this function returns `eps`, the aggregation result of
/// this task will be `1/2 * eps^2` zero-concentrated differentially private.
///
/// Parameters
/// ----------
/// client_state: PyClientState
///     A client state object containing the aggregator locations.
/// task_id: str
///     ID of the task whose privacy parameter we want to know.
///
/// Returns
/// -------
/// privacy: float
///     The zCDP parameter of this aggregation task.
#[pyfunction]
fn client_api_get_privacy_parameter(
    client_state: Py<PyClientState>,
    task_id: Option<String>,
) -> Result<f32>
{
    Python::with_gil(|py| {
        let state_cell: &PyCell<PyClientState> = client_state.as_ref(py);
        let mut state_ref_mut = state_cell
            .try_borrow_mut()
            .map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyClientState = &mut state_ref_mut;

        // if we were given a task_id, we get the parameters for this task
        // from the aggregators (by writing them into the client state)
        // otherwise we assume that the client already has been registered for a round
        if let Some(task_id) = task_id
        {
            let round_settings = RoundSettings::new(task_id)?;
            let future = api_update_client_round_settings(&mut state.mstate, round_settings);
            Runtime::new().unwrap().block_on(future)?;
        }

        // Now try to get the privacy param
        let privacy = state
            .mstate
            .get_valid_state()
            .ok_or_else(|| anyhow!(""))
            .map(|s| s.parametrization.vdaf_parameter.privacy_parameter)?;

        let privacy = (privacy.0 as f32) / (privacy.1 as f32);

        Ok(privacy)
    })
}

/// Submit a gradient vector to a janus server.
///
/// Parameters
/// ----------
/// client_state: PyClientState
///     A client state object containing the aggregator locations.
/// task_id: str
///     ID to identify the janus task to which this gradient corresponds.
/// data: numpy.ndarray
///     The gradient to be submitted. Expected to be (1,)-shaped (flat).
#[pyfunction]
fn client_api_submit(
    client_state: Py<PyClientState>,
    task_id: String,
    data: PyReadonlyArrayDyn<f32>,
) -> Result<()>
{
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
        let state: &mut PyClientState = &mut state_ref_mut;

        let data = array_to_vec(data);

        let round_settings = RoundSettings::new(task_id)?;

        // send vector to janus client for secret sharing
        Runtime::new().unwrap().block_on(api_submit_with(
            &mut state.mstate,
            round_settings,
            |param| {
                println!(
                    "submitting for tag {:?}",
                    param.vdaf_parameter.submission_type.clone()
                );
                match param.vdaf_parameter.submission_type
                {
                    FixedTypeTag::FixedType16Bit => VecFixedAny::VecFixed16(
                        {
                            let v : Result<Vec<_>> = data.into_iter().map(float_to_fixed_floor).collect();
                            if let Ok(v) = v {
                                v
                            } else {
                                vec![fixed!(0.0: I1F15)]
                            }
                        }
                    ),
                    FixedTypeTag::FixedType32Bit => VecFixedAny::VecFixed32(
                        {
                            let v : Result<Vec<_>> = data.into_iter().map(float_to_fixed_floor).collect();
                            if let Ok(v) = v {
                                v
                            } else {
                                vec![fixed!(0.0 : I1F31)]
                            }
                        }
                    ),
                    FixedTypeTag::FixedType64Bit => VecFixedAny::VecFixed64(
                        {
                            let v : Result<Vec<_>> = data.into_iter().map(float_to_fixed_floor).collect();
                            if let Ok(v) = v {
                                v
                            } else {
                                vec![fixed!(0.0 : I1F63)]
                            }
                        }
                    ),
                }
            },
        ))?;

        Ok(())
    })
}

/////////////////////////////////////////////////////////////////
// Controller api

/// Create new controller state.
///
/// Parameters
/// ----------
/// gradient_len: int
///     Size of the gradients to be submitted.
/// privacy_parameter: float
///     used by the aggregators to determine the amount of noise added. Each
///     aggregation result will be `1/2 * privacy_parameter^2` zero-concentrated
///     differentially private.
/// fixed_bitsize: int
///     The resolution of the fixed-point encoding used for secure aggregation
///     A larger value will result in a less lossy representation and more
///     communication and computation overhead. Currently, 16, 32 and 64 bit are
///     supported.
/// external_leader_tasks: str
///     Location of the leader aggregator server in URL format including the port.
///     For example, for a server running locally: "http://127.0.0.1:9991"
/// external_helper_tasks: str
///     Location of the helper aggregator server in URL format including the port.
///     For example, for a server running locally: "http://127.0.0.1:9992"
///
/// Returns
/// -------
/// state: PyControllerState
///     A controller state object containing the input.
#[pyfunction]
fn controller_api_new_state(
    gradient_len: usize,
    privacy_parameter: f32,
    fixed_bitsize: usize,
    external_leader_tasks: String,
    external_helper_tasks: String,
) -> Result<PyControllerState>
{
    // we convert from f32 to a fraction
    let privacy_parameter_frac = GenericFraction::<u128>::from(privacy_parameter);
    let privacy_parameter = match (
        privacy_parameter_frac.numer(),
        privacy_parameter_frac.denom(),
    )
    {
        (Some(n), Some(d)) => (*n, *d),
        _ => Err(anyhow!(
            "The privacy parameter {privacy_parameter_frac} is not a valid finite fraction."
        ))?,
    };

    let submission_type = match fixed_bitsize
    {
        16 => FixedTypeTag::FixedType16Bit,
        32 => FixedTypeTag::FixedType32Bit,
        64 => FixedTypeTag::FixedType64Bit,
        _ => Err(anyhow!(
            "The bitsize {fixed_bitsize} is not supported. Only 16, 32 or 64 is."
        ))?,
    };

    let tasks_locations = TasksLocations {
        external_leader: Url::parse(&external_leader_tasks)?,
        external_helper: Url::parse(&external_helper_tasks)?,
    };

    let main_locations = get_main_locations(tasks_locations.clone());

    let main_locations = Runtime::new().unwrap().block_on(main_locations)?;

    let location = Locations {
        main: main_locations,
        tasks: tasks_locations,
    };

    let vdaf_parameter = VdafParameter {
        gradient_len,
        privacy_parameter,
        submission_type,
    };

    let p = CommonStateParametrization {
        location,
        vdaf_parameter,
    };

    let istate = api_new_controller_state(p);
    let istate: Py<PyCapsule> = Python::with_gil(|py| {
        let capsule = PyCapsule::new(py, istate, None);
        capsule.map(|c| c.into())
    })
    .unwrap();

    let mstate = PyControllerStateMut {
        training_session_id: None,
        task_id: None,
    };

    let res = PyControllerState { mstate, istate };

    Ok(res)
}

/// Run a function on controller state.
fn run_on_controller<A>(
    controller_state: Py<PyControllerState>,
    f: fn(&ControllerStateImmut, &mut ControllerStateMut) -> Result<A>,
) -> Result<A>
{
    Python::with_gil(|py| {
        let state_cell: &PyCell<PyControllerState> = controller_state.as_ref(py);
        let mut state_ref_mut = state_cell
            .try_borrow_mut()
            .map_err(|_| anyhow!("could not get mut ref"))?;
        let state: &mut PyControllerState = &mut state_ref_mut;

        let istate: &ControllerStateImmut = unsafe { state.istate.as_ref(py).reference() };
        let mut mstate: ControllerStateMut = state.mstate.clone().try_into()?;

        // execute async function in tokio runtime
        let res = f(istate, &mut mstate)?;

        // write result into state
        state.mstate = mstate.into();

        Ok(res)
    })
}

/// Create a new training session.
///
/// Parameters
/// ----------
/// controller_state: PyControllerState
///     A controller state object identifying the aggregator servers on which
///     the session is supposed to run.
///
/// Returns
/// -------
/// session_id: int
///     The ID of the newly created training session.
#[pyfunction]
fn controller_api_create_session(controller_state: Py<PyControllerState>) -> Result<u16>
{
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api_create_session(i, m))
    })
}

/// End the current training session.
///
/// Parameters
/// ----------
/// controller_state: PyControllerState
///     A controller state object identifying the aggregator servers on which
///     the session is running.
#[pyfunction]
fn controller_api_end_session(controller_state: Py<PyControllerState>) -> Result<()>
{
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api_end_session(i, m))
    })
}

/// Start a new training round.
///
/// Parameters
/// ----------
/// controller_state: PyControllerState
///     A controller state object identifying the aggregator servers on which
///     the training round is supposed to run.
///
/// Returns
/// -------
/// task_id: str
///     The ID of the newly started task, as a string.
#[pyfunction]
fn controller_api_start_round(controller_state: Py<PyControllerState>) -> Result<String>
{
    run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api_start_round(i, m))
    })
}

/// Collect resulting aggregated gradient vector from janus.
///
/// Parameters
/// ----------
/// controller_state: PyControllerState
///     A controller state object identifying the aggregator servers from which
///     to collect.
///
/// Returns
/// -------
/// aggregate: numpy.ndarray
///     The aggregated and noised gradient vector. (1,)-shaped (flat).
#[pyfunction]
fn controller_api_collect(
    py: Python,
    controller_state: Py<PyControllerState>,
) -> Result<&PyArray1<f64>>
{
    let res = run_on_controller(controller_state, |i, m| {
        Runtime::new().unwrap().block_on(api_collect(i, m))
    })?;

    let vector = res.aggregate_result();

    Ok(vector.to_pyarray(py))
}

/// The python module definition.
#[pymodule]
fn dpsa4fl_bindings(_py: Python, m: &PyModule) -> PyResult<()>
{
    // add class
    m.add_class::<PyClientState>()?;
    m.add_class::<PyControllerState>()?;
    m.add_class::<PyControllerStateMut>()?;

    // add functions
    //--- controller api ---
    m.add_function(wrap_pyfunction!(controller_api_new_state, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api_create_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api_end_session, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api_start_round, m)?)?;
    m.add_function(wrap_pyfunction!(controller_api_collect, m)?)?;
    //--- client api ---
    m.add_function(wrap_pyfunction!(client_api_new_state, m)?)?;
    m.add_function(wrap_pyfunction!(client_api_submit, m)?)?;
    m.add_function(wrap_pyfunction!(client_api_get_privacy_parameter, m)?)?;

    Ok(())
}
