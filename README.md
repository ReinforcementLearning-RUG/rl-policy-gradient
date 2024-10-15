<br />
<p align="center">
  <h1 align="center">Reinforcement Learning</h1>

  <p align="center">
  </p>
</p>

## About
This project contains skeleton code and a virtual environment to help you get started on the programming part of the final assignment. 

## Getting started

### Prerequisites

- [Poetry](https://python-poetry.org/).

## Running
<!--
-->

#### Setting up a virtual environment

You can also setup a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment (after `Add new interpreter`). The interpreter that Pycharm should use is `./.venv/bin/python3.10`.

If you want to add dependencies to the project then you can simply do
```
poetry add <package_name>
```

#### Running the docker container

Instead of running locally you can also run the program inside a container using docker. A `docker-compose.yaml` file is provided which you can use to run the container using `docker compose up --build`.

## Usage
You can add here some description on how to run the project (which file to run for example).
