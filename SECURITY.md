# Security Policy

> **📋 Document Metadata**  
> **Type**: Security Policy | **Audience**: All Users | **Complexity**: Intermediate  
> **Last Updated**: January 2026 | **Status**: Production-Ready  
> **Cross-References**: [Comprehensive Security Guide](doc/security/README.md) | [Deployment Security](doc/deployment/README.md) | [MCP Security](doc/mcp/README.md)

## 🔒 Comprehensive Security Framework

The GNN (GeneralizedNotationNotation) project maintains a comprehensive multi-layered security approach covering development, deployment, and production environments.

> **📖 Complete Security Documentation**: For comprehensive security information, see [Security Guide](doc/security/README.md)

## Supported Versions

We are committed to ensuring the security of the GeneralizedNotationNotation (GNN) project.

| Version | Supported | Security Coverage |
| ------- | ------------------ | ----------------- |
| 1.1.x   | ✅ Full support | Complete security framework |
| 1.0.x   | ✅ LTS support | Backported security fixes |
| 0.1.x   | ⚠️ Legacy support | Critical fixes only |
| < 0.1.0 | ❌ Unsupported | No security support |

> **🔄 Version Updates**: This table is updated with each release. See [Changelog](CHANGELOG.md) for version history.

## 📋 Recent CVE Remediation History

| Date | CVE ID | Package | Action |
|------|--------|---------|--------|
| 2026-01-27 | CVE-2026-24486 | python-multipart | Upgraded 0.0.21 → 0.0.22 |
| 2026-01-21 | CVE-2026-0994 | protobuf | Documented mitigation (pinned to 6.33.4) |

> **ℹ️ Known Accepted Risks**: The following vulnerabilities are documented and accepted:
>
> - **CVE-2024-39236** (gradio): Disputed - self-attack scenario only
> - **CVE-2022-42969** (py): Disputed ReDoS - transitive via deprecated `nose` (upstream pymdp dependency)

## 🚨 Reporting Security Vulnerabilities

The GNN team and community take all security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### Reporting Channels

**Primary Contact:**

- **Email**: Send an email to `blanket@activeinference.institute`
- **Subject Line**: Use "Security Vulnerability in GNN Project"

**GitHub Security:**

- **Platform**: [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
- **Repository**: [GeneralizedNotationNotation](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/security/advisories)
- **Benefits**: Automated coordination with dependency maintainers

> **⚠️ Important**: Please do not report security vulnerabilities through public GitHub issues.

### What to Include

When reporting a vulnerability, please provide:

- **Clear description** of the vulnerability and its impact
- **Component identification**: Affected files, modules, or pipeline steps
- **Reproduction steps**: Detailed steps to reproduce the issue  
- **Version information**: Affected GNN versions and dependencies
- **Environment details**: Operating system, Python version, framework versions
- **Proof of concept**: If applicable, demonstration code (safely)
- **Suggested mitigations**: If you have ideas for fixes

### Security-Specific Concerns

**LLM Integration Security** (Pipeline Step 13):

- API key exposure in configuration files
- Prompt injection attacks through GNN files  
- Unsafe code generation from LLM outputs

**MCP Security** (Pipeline Step 21):

- Model Context Protocol authentication issues
- Unsafe resource access patterns
- Data leakage through model context

**Pipeline Security** (All 25 Steps):

- Code injection through GNN file parsing
- Unsafe file operations in output generation
- Privilege escalation in execution steps

## 🛡️ Our Security Commitment

### Response Timeline

Once a security vulnerability is reported, we commit to:

**Immediate Response (24-48 hours):**

- Acknowledge receipt of the vulnerability report
- Assign a security team member as primary contact
- Begin initial assessment and triage

**Investigation Phase (1-7 days):**

- Validate and reproduce the vulnerability
- Assess severity using CVSS scoring
- Determine affected versions and components
- Develop initial mitigation strategies

**Resolution Phase (Variable, based on severity):**

- **Critical**: 24-72 hours for emergency patch
- **High**: 1-2 weeks for comprehensive fix
- **Medium**: 2-4 weeks for scheduled release
- **Low**: Next planned release cycle

**Disclosure Phase:**

- Coordinate responsible disclosure timeline
- Prepare security advisory and documentation
- Release patched versions across supported branches
- Publicly acknowledge contributor (unless requested otherwise)

### Security Integration

**Development Security:**

- All code changes reviewed for security implications
- Automated security scanning in CI/CD pipeline  
- Dependency vulnerability monitoring
- Regular security audits of critical components

**Documentation Security:**

- Security considerations in all operational guides
- Threat model documentation for each pipeline step
- Security configuration examples and best practices
- Incident response procedures and playbooks

## 🔧 Security Best Practices for Users

### Development Environment Security

**Environment Setup:**

- Use isolated Python virtual environments
- Keep dependencies updated: `uv sync --refresh`
- Validate GNN file sources before processing
- Use secure API key storage (environment variables, not files)

**Code Security:**

- Review generated code before execution
- Validate all inputs to GNN parsers
- Use sandbox environments for testing unknown models
- Follow secure coding practices for extensions

### Production Deployment Security

**Infrastructure Security:**

- Deploy with minimal required privileges
- Use encrypted connections for all API calls
- Implement proper logging and monitoring
- Regular security updates and patches

**Configuration Security:**

- Secure API key management (Azure Key Vault, AWS Secrets Manager)
- Network segmentation for GNN processing
- Input validation for all user-provided GNN files
- Output sanitization for generated code

### Framework-Specific Security

**PyMDP Security:**

- Validate matrix dimensions before processing
- Sanitize numerical inputs for stability
- Monitor memory usage for large state spaces

**RxInfer.jl Security:**  

- Validate Julia code generation outputs
- Secure inter-process communication with Julia
- Monitor computational resource usage

**ActiveInference.jl Security:**

- Validate Julia ActiveInference.jl code generation outputs
- Secure inter-process communication with Julia
- Monitor computational resource usage for ActiveInference.jl simulations

**LLM Integration Security:**

- Never include sensitive data in prompts
- Validate all LLM-generated outputs
- Use prompt injection prevention techniques
- Implement rate limiting for API calls

## 📚 Security Resources

### Comprehensive Guides

- **[Complete Security Framework](doc/security/README.md)** - Comprehensive security guide
- **[Deployment Security](doc/deployment/README.md)** - Production security configurations
- **[MCP Security](doc/mcp/README.md)** - Model Context Protocol security measures

### Framework Security

- **[PyMDP Security](doc/pymdp/gnn_pymdp.md#security-considerations)** - PyMDP-specific security
- **[RxInfer.jl Security](doc/rxinfer/gnn_rxinfer.md#security-considerations)** - Julia integration security
- **[ActiveInference.jl Security](doc/activeinference_jl/activeinference-jl.md#security-considerations)** - ActiveInference.jl integration security
- **[LLM Security](doc/llm/security_guidelines.md)** - AI integration security practices

### Incident Response

- **[Security Incident Response](doc/security/incident_response.md)** - Response procedures
- **[Vulnerability Assessment](doc/security/vulnerability_assessment.md)** - Assessment frameworks
- **[Security Monitoring](doc/security/monitoring.md)** - Monitoring and alerting

## 🤝 Security Community

### Contributing to Security

- **Security Review**: Participate in security-focused code reviews
- **Vulnerability Research**: Help identify potential security issues
- **Documentation**: Improve security documentation and guides
- **Tool Development**: Create security-focused tools and utilities

### Security Updates

- **Security Announcements**: Subscribe to repository notifications
- **Release Notes**: Check [Changelog](CHANGELOG.md) for security fixes
- **Community Forum**: Engage in security discussions
- **Best Practices**: Share security configurations and patterns

---

**We appreciate your help in keeping GeneralizedNotationNotation secure across all dimensions: physical, digital, and cognitive.**

> **🔗 Related Documentation**: [Security Guide](doc/security/README.md) | [Deployment Security](doc/deployment/README.md) | [Contributing Security](CONTRIBUTING.md#security-considerations)
