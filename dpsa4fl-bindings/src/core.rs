use anyhow::Result;
use dpsa4fl::{controller::{ControllerState_Mut, ControllerState_Round}, core::FixedAny};
use dpsa4fl_janus_tasks::fixed::{Fixed16, Fixed32, Fixed64};
use fixed::traits::Fixed;
use pyo3::{prelude::*, types::PyCapsule};

pub type PyMeasurement = f64;

#[derive(Clone)]
#[pyclass]
pub struct PyControllerState_Mut {
    #[pyo3(get, set)]
    pub training_session_id: Option<u16>,

    #[pyo3(get, set)]
    pub task_id: Option<String>,
}

#[pyclass]
pub struct PyControllerState {
    #[pyo3(get, set)]
    pub mstate: PyControllerState_Mut,

    pub istate: Py<PyCapsule>,
}

impl From<ControllerState_Mut> for PyControllerState_Mut {
    fn from(s: ControllerState_Mut) -> Self {
        PyControllerState_Mut {
            training_session_id: s.round.training_session_id.map(|x| x.into()),
            task_id: s.round.task_id.map(dpsa4fl::helpers::task_id_to_string),
        }
    }
}

impl TryInto<ControllerState_Mut> for PyControllerState_Mut {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ControllerState_Mut> {
        let task_id = if let Some(task_id) = self.task_id {
            Some(dpsa4fl::helpers::task_id_from_string(task_id)?)
        } else {
            None
        };

        let round = ControllerState_Round {
            training_session_id: self.training_session_id.map(|x| x.into()),
            task_id,
        };

        let res = ControllerState_Mut { round };

        Ok(res)
    }
}





