# Support for GeneralizedNotationNotation

> **📋 Document Metadata**  
> **Type**: Support Guide | **Audience**: All Users | **Complexity**: Beginner-Friendly  
> **Last Updated**: July 2025 | **Status**: Production-Ready  
> **Cross-References**: [README](README.md) | [Troubleshooting](doc/troubleshooting/README.md) | [Contributing](CONTRIBUTING.md)

If you need help with GeneralizedNotationNotation (GNN), have questions, or want to discuss the project, here are several ways to get support:

## Documentation

Before seeking direct support, please check the extensive documentation provided with the project:

-   **README.md**: The [main README file](./README.md) provides an overview of the project, setup instructions, and how to run the processing pipeline.
-   **`doc/` Directory**: This directory contains detailed information on various aspects of GNN:
    -   [`doc/gnn/about_gnn.md`](./doc/gnn/about_gnn.md): General information about GNN.
    -   [`doc/gnn/gnn_overview.md`](./doc/gnn/gnn_overview.md): A high-level overview.
    -   [`doc/gnn/gnn_syntax.md`](./doc/gnn/gnn_syntax.md): Detailed specification of GNN syntax.
    -   [`doc/gnn/gnn_file_structure_doc.md`](./doc/gnn/gnn_file_structure_doc.md): Description of GNN file organization.
    -   [`doc/gnn/gnn_examples_doc.md`](./doc/gnn/gnn_examples_doc.md): Examples and use cases.
    -   And many other useful documents covering implementation, tools, and the GNN paper.
-   **Example GNN Files**: Explore the GNN files in `src/gnn/examples/` to understand practical applications.
-   **Pipeline Documentation**: Check [`src/README.md`](./src/README.md) for detailed pipeline information.

## Quick Troubleshooting

### Common Issues and Solutions

**🐍 Python Version Issues**
```bash
# Check Python version
python --version
# If < 3.8, install Python 3.8+ from python.org
```

**📦 Dependency Issues**
```bash
# Force reinstall dependencies
python src/main.py --only-steps 2 --recreate-venv --dev
```

**🔧 Pipeline Failures**
```bash
# Run with verbose logging
python src/main.py --verbose
# Check specific step
python src/main.py --only-steps 4 --verbose
```

**💾 Disk Space Issues**
```bash
# Check available space
df -h
# Clean output directory
rm -rf output/*
```

### Getting Started Support

**🚀 First Time Setup**
```bash
# Clone and setup
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation
python src/main.py --only-steps 2 --dev
```

**🎯 Quick Test**
```bash
# Test with example files
python src/main.py --target-dir src/gnn/examples --verbose
```

## GitHub Issues

For bug reports, feature requests, or specific questions that are not covered by the documentation, the primary place to seek support is through [GitHub Issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues).

-   **Search Existing Issues**: Before creating a new issue, please search existing open and closed issues to see if your question has already been addressed.
-   **Bug Reports**: If you are reporting a bug, please provide:
    -   A clear and descriptive title.
    -   Steps to reproduce the bug.
    -   The version of GNN you are using (if applicable).
    -   Any relevant error messages or logs.
    -   Your operating system and Python version.
-   **Feature Requests**: For feature requests, describe the feature, its potential benefits, and any proposed implementation ideas.

## Community Channels

Join the wider Active Inference community for discussions, questions, and collaboration related to GNN and Active Inference in general:

-   **Active Inference Institute Community**: Visit the [Active Inference Institute website](https://activeinference.institute/) for links to community platforms such as Discord, forums, or mailing lists where GNN might be discussed.
-   **GitHub Discussions**: Use [GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions) for community discussions and questions.
-   **Discord Community**: Join the [Active Inference Discord](https://discord.activeinference.institute/) for real-time discussions.

## Direct Contact (for specific inquiries)

For matters not suitable for public forums or GitHub issues (e.g., sensitive inquiries), you may try to reach out to the project maintainers. However, for general support and technical questions, public channels are preferred as they benefit the entire community.

-   Refer to the project maintainers listed on the [GitHub repository page](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation).
-   For security-related issues, see [SECURITY.md](./SECURITY.md) for proper reporting procedures.

## Commercial Support

Currently, there is no official commercial support offered for GeneralizedNotationNotation. The project is community-driven.

## Support Resources by Topic

### 🧠 Active Inference Concepts
- [About GNN](./doc/gnn/about_gnn.md) - Introduction to GNN and Active Inference
- [GNN Overview](./doc/gnn/gnn_overview.md) - High-level concepts
- [GNN Paper](./doc/gnn/gnn_paper.md) - Academic paper details

### 🛠️ Technical Implementation
- [GNN Syntax](./doc/gnn/gnn_syntax.md) - Detailed syntax specification
- [File Structure](./doc/gnn/gnn_file_structure_doc.md) - GNN file organization
- [Implementation Guide](./doc/gnn/gnn_implementation.md) - Implementation details

### 🎯 Framework Integration
- [PyMDP Integration](./doc/pymdp/) - PyMDP framework integration
- [RxInfer.jl Integration](./doc/rxinfer/) - RxInfer.jl framework integration
- [ActiveInference.jl Integration](./doc/activeinference_jl/) - ActiveInference.jl framework integration
- [MCP Integration](./doc/mcp/) - Model Context Protocol

### 🔧 Pipeline and Tools
- [Pipeline Architecture](./doc/pipeline/PIPELINE_ARCHITECTURE.md) - 14-step pipeline overview
- [API Reference](./doc/api/README.md) - Complete API documentation
- [Testing Guide](./doc/testing/README.md) - Testing strategies

### 🚨 Troubleshooting
- [Troubleshooting Guide](./doc/troubleshooting/) - Common issues and solutions
- [Security Guide](./doc/security/README.md) - Security considerations
- [Deployment Guide](./doc/deployment/README.md) - Production deployment

---

We strive to support our users and contributors. Using the channels above helps us manage requests efficiently and build a helpful knowledge base for everyone. 