# Changelog

> **📋 Document Metadata**  
> **Type**: Change Log | **Audience**: All Users, Maintainers | **Complexity**: Reference  
> **Last Updated**: July 2025 | **Status**: Production-Ready  
> **Cross-References**: [Release Management](doc/releases/README.md) | [Security Policy](SECURITY.md) | [Contributing](CONTRIBUTING.md)

All notable changes to the GeneralizedNotationNotation (GNN) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-07-16

### Added
- Comprehensive 14-step processing pipeline
- Enhanced MCP (Model Context Protocol) integration
- SAPF (Sound As Pure Form) audio generation capabilities
- Advanced visualization system with matrix heatmaps and network graphs
- Multi-format export support (JSON, XML, GraphML, GEXF, Pickle)
- ActiveInference.jl integration for Julia-based simulations
- Enhanced LLM analysis and model interpretation
- Comprehensive test suite with pytest integration
- Static HTML website generation from pipeline artifacts
- Advanced type checking and validation system
- Complete documentation system with cross-references
- Security framework and vulnerability reporting system

### Changed
- Restructured pipeline architecture for better modularity
- Improved error handling and logging throughout the system
- Enhanced argument parsing with centralized utilities
- Standardized output directory structure
- Updated documentation with comprehensive guides and examples

### Fixed
- Version consistency across all metadata files
- Email case inconsistencies in documentation
- Cross-reference links in documentation
- Pipeline step counting accuracy

### Security
- Implemented comprehensive security policy
- Added vulnerability reporting procedures
- Enhanced input validation and sanitization
- Secure API key management guidelines

## [1.0.0] - 2025-06-01

### Added
- Core GNN parsing and validation functionality
- Basic pipeline structure with numbered steps
- PyMDP and RxInfer.jl integration
- Initial documentation framework
- Basic visualization capabilities
- Export system for multiple formats
- Initial MCP integration
- Test framework setup

### Changed
- Migrated from prototype to production-ready architecture
- Standardized coding conventions and practices
- Improved module organization and structure

### Fixed
- Core parsing bugs and edge cases
- Cross-platform compatibility issues
- Documentation consistency problems

## [0.1.0] - 2025-05-15

### Added
- Initial prototype release
- Basic GNN syntax specification
- Simple parsing capabilities
- Foundation documentation
- Initial project structure
- MIT license
- Basic README and setup instructions

### Notes
- This was the initial development release
- Many features were experimental and subject to change
- Limited framework integration
- Basic documentation only

---

## Version Support Policy

| Version | Supported | Security Coverage |
|---------|-----------|-------------------|
| 1.1.x   | ✅ Full support | Complete security framework |
| 1.0.x   | ✅ LTS support | Backported security fixes |
| 0.1.x   | ⚠️ Legacy support | Critical fixes only |
| < 0.1.0 | ❌ Unsupported | No security support |

## Links

- [Repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)
- [Documentation](doc/README.md)
- [Security Policy](SECURITY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Release Management](doc/releases/README.md)

[Unreleased]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/releases/tag/v0.1.0 