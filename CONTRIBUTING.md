# Contributing to GeneralizedNotationNotation

First off, thank you for considering contributing to GeneralizedNotationNotation (GNN)! It's people like you that make GNN such a great tool.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective contribution process.

## How Can I Contribute?

There are many ways you can contribute to the GNN project:

- **Reporting Bugs**: If you find a bug, please report it by opening an issue on GitHub.
- **Suggesting Enhancements**: If you have ideas for new features or improvements, feel free to open an issue to discuss them.
- **Writing Documentation**: Clear and comprehensive documentation is crucial. You can help by improving existing documentation or adding new content.
- **Submitting Code**: If you want to contribute code, please follow the guidelines below.
- **Improving GNN Examples**: Add new examples or refine existing ones in `src/gnn/examples/`.
- **Expanding the Ontology**: Contribute to the Active Inference Ontology and its mapping within GNN.
- **Developing Tools**: Create or enhance tools that work with GNN files.

## Getting Started

1.  **Fork the Repository**: Start by forking the main [GeneralizedNotationNotation repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation) on GitHub.
2.  **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YourUsername/GeneralizedNotationNotation.git
    cd GeneralizedNotationNotation
    ```
3.  **Set Up Your Environment**: Ensure you have Python installed. It's recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    pip install -r src/requirements.txt
    ```
    Refer to `src/setup/setup.py` for more details on the setup process, although manual execution of `pip install` as above is often sufficient for development contributions.

4.  **Create a Branch**: Create a new branch for your changes.
    ```bash
    git checkout -b your-feature-or-bugfix-branch
    ```

## Making Changes

- **Coding Style**: Please adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. We use `flake8` and `black` for linting and formatting. Consider configuring your editor to use these tools.
- **Commit Messages**: Write clear and concise commit messages. Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification if possible.
- **Tests**: If you add new features, please include corresponding tests in the `src/tests/` directory. Ensure all tests pass before submitting your changes.
    ```bash
    python src/main.py --only-steps 3_tests
    ```
- **Documentation**: If your changes affect the behavior of the software or add new features, please update the documentation accordingly (files in `doc/` and potentially `README.md`).

## Submitting Changes

1.  **Push Your Changes**: Push your changes to your forked repository.
    ```bash
    git push origin your-feature-or-bugfix-branch
    ```
2.  **Open a Pull Request**: Open a pull request (PR) from your branch to the `main` branch of the official GNN repository.
    - Provide a clear title and description for your PR.
    - Link to any relevant issues.
    - Ensure your PR passes all CI checks.

## Code Review

- A project maintainer will review your PR.
- Be prepared to make changes based on the feedback received.
- Once your PR is approved, it will be merged into the main codebase.

## Issue and PR Management

- **Issues**: Use GitHub Issues to report bugs, request features, or discuss potential changes.
- **Pull Requests**: Ensure PRs are focused on a single issue or feature. Avoid mixing multiple unrelated changes in one PR.

## Community

- Join the discussion on the [Active Inference Institute community platforms](https://activeinference.institute/community) (replace with specific links if available).

Thank you for your contribution! 