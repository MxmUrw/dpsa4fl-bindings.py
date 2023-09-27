
# Python bindings for dpsa4fl

**Warning: This project is work in progress and should not be used in production. The current implementation is a prototype.**

The dpsa4fl project aims at providing a mechanism for secure and differentially private aggregation
of gradients in federated machine learning. For more information see the [project overview](https://github.com/dpsa-project/overview).

## About this package
This package provides python bindings for the [dpsa4fl](https://github.com/dpsa-project/dpsa4fl) library.
The following functionality is provided:
 - **Controller api**: start a training session on the janus server, and collect aggregated gradients.
 - **Client api**: securely submit gradients to the janus server.
 
A modified [janus server setup](https://github.com/dpsa-project/dpsa4fl-testing-infrastructure) is required,
see the [example project](https://github.com/dpsa-project/dpsa4fl-example-project) for step-by-step instructions.

## Controller api

### `controller_api_new_state(gradient_len, privacy_parameter, fixed_bitsize, external_leader_tasks, external_helper_tasks)`
Create new controller state.

#### Parameters
- `gradient_len: int` Size of the gradients to be submitted.
- `privacy_parameter: float` used by the aggregators to determine the amount of noise added. Each aggregation result will be `1/2 * privacy_parameter^2` zero-concentrated differentially private.
- `fixed_bitsize: int` The resolution of the fixed-point encoding used for secure aggregation A larger value will result in a less lossy representation and more communication and computation overhead. Currently, 16, 32 and 64 bit are supported.
- `external_leader_tasks: str` Location of the leader aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9991"
- `external_helper_tasks: str` Location of the helper aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9992"

#### Returns
- `state: PyControllerState` A controller state object containing the input.
---

### `controller_api_create_session(controller_state)`
Create a new training session.

#### Parameters
- `controller_state: PyControllerState` A controller state object identifying the aggregator servers on which the session is supposed to run.

#### Returns
- `session_id: int` The ID of the newly created training session.

---

### `controller_api_end_session(controller_state)`
End the current training session.

#### Parameters
- `controller_state: PyControllerState` A controller state object identifying the aggregator servers on which the session is running.
---

### `controller_api_start_round(controller_state)`
Start a new training round.

#### Parameters
- `controller_state: PyControllerState` A controller state object identifying the aggregator servers on which the training round is supposed to run.

#### Returns
- `task_id: str` The ID of the newly started task, as a string.

---

### `controller_api_collect(controller_state)`
Collect resulting aggregated gradient vector from janus.

#### Parameters
- `controller_state: PyControllerState` A controller state object identifying the aggregator servers from which to collect.

#### Returns
- `aggregate: numpy.ndarray` The aggregated and noised gradient vector. (1,)-shaped (flat).

---

### `class PyControllerState`
State object for dpsa4fl servers. Stores gradient length, privacy parameter, fixedpoint resolution and aggregator locations. Also contains current task and training session IDs, accessible with the `mstate: PyControllerStateMut` attribute.

### `class PyControllerStateMut`
State object for dpsa4fl servers containing the current training session and taks IDs.


## Client api

### `client_api_new_state( external_leader_tasks, external_helper_tasks)`
Create new client state, containing the aggregator locations.

#### Parameters
- `external_leader_tasks: str` Location of the leader aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9991"
- `external_helper_tasks: str` Location of the helper aggregator server in URL format including the port. For example, for a server running locally: "http://127.0.0.1:9992"

#### Returns
- `state: PyClientState` A client state object containing the input locations.

---

### `client_api_get_privacy_parameter(client_state, task_id)`
Request the privacy parameter used by the aggregator to determine the amount
of noise added. If this function returns `eps`, the aggregation result of
this task will be `1/2 * eps^2` zero-concentrated differentially private.

#### Parameters
- `client_state: PyClientState` A client state object containing the aggregator locations.
- `task_id: str` ID of the task whose privacy parameter we want to know.

#### Returns
- `privacy: float` The zCDP parameter of this aggregation task.i



---

### `client_api_submit(client_state, task_id, data)`
Submit a gradient vector to a janus server.

#### Parameters
- `client_state: PyClientState` A client state object containing the aggregator locations.
- `task_id: str` ID to identify the janus task to which this gradient corresponds.
- `data: numpy.ndarray` The gradient to be submitted. Expected to be (1,)-shaped (flat).

---

### `class PyClientState`
State object for dpsa4fl clients. Stores aggregator locations.




## Development notes
To release a new version of this package, you have to:
 1. Increment the version number in `Cargo.toml`.
 2. Push the current state to the `release` branch. Then github actions will do the rest.
    Alternatively, you can use [act](https://github.com/nektos/act) to run github actions locally.


Use [`maturin`](https://github.com/PyO3/maturin) to build the python package. To do so using the [`poetry`](https://python-poetry.org/) python package manager, do the following in the `dpsa4fl-bindings` folder:
```
poetry shell
maturin develop
```
