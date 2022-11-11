
# Python bindings for dpsa4fl
This python package is a wrapper around the [dpsa4fl](https://github.com/dpsa-project/dpsa4fl) crate.
See [here](https://github.com/dpsa-project/overview) for more information.

## Development notes
To release a new version of this package, you have to:
 1. Increment the version number in `Cargo.toml`.
 2. Push the current state to the `release` branch. Then github actions will do the rest.
    Alternatively, you can use [act](https://github.com/nektos/act) to run github actions locally.

