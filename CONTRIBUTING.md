# Contributing to GeneralizedNotationNotation

> **📋 Document Metadata**  
> **Type**: Contribution Guide | **Audience**: Contributors, Developers | **Complexity**: Intermediate  
> **Last Updated**: July 2025 | **Status**: Production-Ready  
> **Cross-References**: [Code of Conduct](CODE_OF_CONDUCT.md) | [Security Policy](SECURITY.md) | [Documentation Guide](doc/README.md)

First off, thank you for considering contributing to GeneralizedNotationNotation (GNN)! It's agents like you that make GNN such a great tool.

Contact: [blanket@activeinference.institute](mailto:blanket@activeinference.institute)

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

## 📚 Before You Start: Documentation Resources

GNN has comprehensive documentation to help you understand the project:

- **[Documentation Overview](doc/README.md)**: Complete guide to all documentation
- **[Development Guide](doc/development/README.md)**: Detailed developer documentation
- **[API Reference](doc/api/README.md)**: Complete API documentation
- **[Testing Guide](doc/testing/README.md)**: Testing strategies and framework
- **[Template System](doc/templates/README.md)**: Production-ready templates for common patterns
- **[Pipeline Architecture](doc/pipeline/PIPELINE_ARCHITECTURE.md)**: Technical architecture overview

## Getting Started

1.  **Fork the Repository**: Start by forking the main [GeneralizedNotationNotation repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation) on GitHub.
2.  **Clone Your Fork**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YourUsername/GeneralizedNotationNotation.git
    cd GeneralizedNotationNotation
    ```
3.  **Set Up Your Environment**: Ensure you have Python installed. It's recommended to use a virtual environment.
    ```bash
    # Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize UV environment
uv init
uv sync
    pip install -r requirements.txt
    ```
    Refer to **[SETUP.md](doc/SETUP.md)** for comprehensive setup instructions and the **[Configuration Guide](doc/configuration/README.md)** for detailed configuration options.

4.  **Create a Branch**: Create a new branch for your changes.
    ```bash
    git checkout -b your-feature-or-bugfix-branch
    ```

## Making Changes

### 🔧 Development Standards

- **Coding Style**: Please adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code. We use `flake8` and `black` for linting and formatting. Consider configuring your editor to use these tools.
- **Commit Messages**: Write clear and concise commit messages. Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification if possible.
- **GNN Standards**: Follow the [GNN Coding Standards](.cursorrules) for GNN-specific conventions, including variable naming (`s_f0`, `o_m0`, etc.) and function prefixes (`gnn_`).

### 🧪 Testing Requirements

- **Comprehensive Testing**: If you add new features, please include corresponding tests in the `src/tests/` directory. 
- **14-Step Pipeline Testing**: Ensure all tests pass before submitting your changes:
    ```bash
    python src/main.py --only-steps 3
    ```
- **Template Testing**: If you modify templates, test them with the validation pipeline:
    ```bash
    python src/main.py --only-steps 4
    ```
- **Framework Integration**: Test integration with PyMDP, RxInfer.jl, ActiveInference.jl, or other frameworks if your changes affect framework compatibility.

See the **[Testing Guide](doc/testing/README.md)** for comprehensive testing strategies.

### 📝 Documentation Requirements

- **Documentation Updates**: If your changes affect the behavior of the software or add new features, please update the documentation accordingly.
- **Template System**: Use the **[Template System](doc/templates/README.md)** for creating new GNN model examples.
- **API Documentation**: Update **[API documentation](doc/api/README.md)** for any new or modified functions.
- **Pipeline Documentation**: Update relevant pipeline step documentation if you modify pipeline behavior.

### 🎯 Contribution Types

#### **Code Contributions**
- Follow the **[Development Guide](doc/development/README.md)** for code standards
- Reference the **[Pipeline Architecture](doc/pipeline/PIPELINE_ARCHITECTURE.md)** for understanding system design
- Use the **[API Reference](doc/api/README.md)** for consistent function signatures

#### **Documentation Contributions**
- Follow the **[Documentation Style Guide](doc/style_guide.md)**
- Use existing templates and maintain consistency with current documentation structure
- Ensure cross-references between related documents

#### **Example/Template Contributions**
- Use the **[Template System](doc/templates/)** as a foundation
- Test examples with the **[Type Checker](doc/pipeline/README.md#step-4-gnn-type-checker)**
- Include both basic and advanced usage patterns

## Submitting Changes

1.  **Push Your Changes**: Push your changes to your forked repository.
    ```bash
    git push origin your-feature-or-bugfix-branch
    ```
2.  **Open a Pull Request**: Open a pull request (PR) from your branch to the `main` branch of the official GNN repository.
    - Provide a clear title and description for your PR.
    - Link to any relevant issues.
    - Ensure your PR passes all CI checks.
    - Reference relevant documentation sections that support your changes.

## Code Review

- A project maintainer will review your PR using the **[Quality Assurance Framework](doc/style_guide.md#quality-assurance)**.
- Be prepared to make changes based on the feedback received.
- Ensure your contribution aligns with **[GNN Standards](.cursorrules)** and **[Testing Requirements](doc/testing/README.md)**.
- Once your PR is approved, it will be merged into the main codebase.

## Issue and PR Management

- **Issues**: Use GitHub Issues to report bugs, request features, or discuss potential changes. Reference the **[Troubleshooting Guide](doc/troubleshooting/README.md)** for common issues.
- **Pull Requests**: Ensure PRs are focused on a single issue or feature. Avoid mixing multiple unrelated changes in one PR.
- **Documentation Issues**: Use documentation-specific labels and reference the **[Documentation Style Guide](doc/style_guide.md)** for standards.

## 🎯 Specialized Contribution Areas

### **Framework Integration**
- **PyMDP**: See **[PyMDP Integration Guide](doc/pymdp/gnn_pymdp.md)** (comprehensive examples)
- **RxInfer.jl**: See **[RxInfer Integration Guide](doc/rxinfer/gnn_rxinfer.md)** (extensive documentation)
- **ActiveInference.jl**: See **[ActiveInference.jl Integration Guide](doc/activeinference_jl/activeinference-jl.md)** (Julia-based implementation)
- **DisCoPy**: Reference **[DisCoPy documentation](doc/discopy/)** for categorical diagram contributions

### **Pipeline Development**
- **14-Step Pipeline**: Understand the **[Pipeline Architecture](doc/pipeline/PIPELINE_ARCHITECTURE.md)**
- **New Pipeline Steps**: Follow the **[Development Guide](doc/development/README.md)** for adding new numbered scripts
- **Configuration**: Use the **[Configuration Guide](doc/configuration/README.md)** for proper parameter handling

### **Documentation Contributions**
- **Maintenance**: Follow the **[Documentation Style Guide](doc/style_guide.md)**
- **Quality Standards**: Maintain professional documentation quality per the style guide
- **Template Creation**: Use the **[Template System](doc/templates/README.md)** for consistent formatting

## Community

- Join the discussion on the [Active Inference Institute community platforms](https://activeinference.institute/community).
- Reference the **[Support Guide](SUPPORT.md)** for getting help with contributions.
- Follow the **[Code of Conduct](CODE_OF_CONDUCT.md)** in all community interactions.

## 🏆 Recognition

We value all contributions to the GNN project. Contributors will be:
- Acknowledged in release notes for significant contributions
- Listed in the project's contributor recognition system
- Invited to join the GNN development community for ongoing contributors

Thank you for your contribution! Your work helps make GNN a better tool for the Active Inference research community. 