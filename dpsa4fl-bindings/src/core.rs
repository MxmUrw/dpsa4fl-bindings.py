use anyhow::Result;
use dpsa4fl::{
    controller::{ControllerStateMut, ControllerStateRound},
    core::FixedAny,
};
use dpsa4fl_janus_tasks::fixed::{Fixed16, Fixed32, Fixed64};
use fixed::traits::Fixed;
use pyo3::{prelude::*, types::PyCapsule};

pub type PyMeasurement = f64;

#[derive(Clone)]
#[pyclass]
pub struct PyControllerStateMut
{
    #[pyo3(get, set)]
    pub training_session_id: Option<u16>,

    #[pyo3(get, set)]
    pub task_id: Option<String>,
}

#[pyclass]
pub struct PyControllerState
{
    #[pyo3(get, set)]
    pub mstate: PyControllerStateMut,

    pub istate: Py<PyCapsule>,
}

impl From<ControllerStateMut> for PyControllerStateMut
{
    fn from(s: ControllerStateMut) -> Self
    {
        PyControllerStateMut {
            training_session_id: s.round.training_session_id.map(|x| x.into()),
            task_id: s.round.task_id.map(dpsa4fl::helpers::task_id_to_string),
        }
    }
}

impl TryInto<ControllerStateMut> for PyControllerStateMut
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ControllerStateMut>
    {
        let task_id = if let Some(task_id) = self.task_id
        {
            Some(dpsa4fl::helpers::task_id_from_string(task_id)?)
        }
        else
        {
            None
        };

        let round = ControllerStateRound {
            training_session_id: self.training_session_id.map(|x| x.into()),
            task_id,
        };

        let res = ControllerStateMut { round };

        Ok(res)
    }
}
