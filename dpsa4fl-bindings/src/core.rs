

use std::ffi::CString;

use prio::codec::{Encode, Decode, CodecError};
use pyo3::{prelude::*, types::PyCapsule};
use dpsa4fl::{*, controller::{api__new_controller_state, ControllerState_Immut, ControllerState_Mut, api__create_session, ControllerState_Permanent, ControllerState_Round}, core::{CommonState_Parametrization, Locations}};
use url::Url;
use anyhow::Result;
use tokio::runtime::Runtime;

// pub struct PyControllerState_Immut
// {
//     pub parametrization: CommonState_Parametrization,
//     pub permanent: ControllerState_Permanent,
// }

pub type PyMeasurement = f64;

#[derive(Clone)]
#[pyclass]
pub struct PyControllerState_Mut
{
    #[pyo3(get,set)]
    pub training_session_id: Option<u16>,

    #[pyo3(get,set)]
    pub task_id: Option<String>,
}

#[pyclass]
pub struct PyControllerState
{
    #[pyo3(get, set)]
    pub mstate: PyControllerState_Mut,

    pub istate: Py<PyCapsule>
}


impl From<ControllerState_Mut> for PyControllerState_Mut
{
    fn from(s: ControllerState_Mut) -> Self {
        // let immutstate : Py<PyCapsule> = Python::with_gil(|py| {
        //     let content = PyControllerState_Immut {
        //         parametrization: s.parametrization,
        //         permanent: s.permanent,
        //     };

        //     let capsule = PyCapsule::new(py, content, None);
        //     capsule.map(|c| c.into())
        // }).unwrap();

        PyControllerState_Mut {
            training_session_id: s.round.training_session_id.map(|x| x.into()),
            task_id: s.round.task_id.map(dpsa4fl::helpers::task_id_to_string),
        }

        // PyControllerState {
        //     mutstate,
        //     immutstate,
        // }
    }
}

impl TryInto<ControllerState_Mut> for PyControllerState_Mut
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ControllerState_Mut>
    {
        let task_id = if let Some(task_id) = self.task_id
        {
            Some(dpsa4fl::helpers::task_id_from_string(task_id)?)
        }
        else
        {
            None
        };

        let round = ControllerState_Round {
            training_session_id: self.training_session_id.map(|x| x.into()),
            task_id,
        };

        let res = ControllerState_Mut {
            round
        };

        Ok(res)
    }
}

// impl Into<Py<PyCapsule>> for ControllerState_Immut
// {
//     fn into(self) -> Py<PyCapsule> {
//         let immutstate : Py<PyCapsule> = Python::with_gil(|py| {
//             let capsule = PyCapsule::new(py, self, None);
//             capsule.map(|c| c.into())
//         }).unwrap();
//     }
// }

