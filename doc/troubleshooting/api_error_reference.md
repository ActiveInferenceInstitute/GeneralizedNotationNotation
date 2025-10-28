# GNN API Error Reference

> **📋 Document Metadata**  
> **Type**: API Reference | **Audience**: Developers | **Complexity**: Advanced  
> **Last Updated: October 2025 | **Status**: Production-Ready  
> **Cross-References**: [Error Taxonomy](error_taxonomy.md) | [Debugging Workflows](debugging_workflows.md) | [API Documentation](../api/README.md)

## Overview

This document provides comprehensive reference for programmatic error handling in the GNN (Generalized Notation Notation) system. It covers exception hierarchies, error codes, and best practices for robust error handling in applications using GNN.

## Exception Hierarchy

```python
GNNException
├── GNNSyntaxError
│   ├── MissingSectionError
│   ├── InvalidHeaderError
│   ├── MalformedVariableError
│   └── InvalidConnectionError
├── GNNValidationError
│   ├── DimensionMismatchError
│   ├── MatrixValidationError
│   ├── ConnectionValidationError
│   └── ParameterValidationError
├── GNNRuntimeError
│   ├── PipelineExecutionError
│   ├── ResourceExhaustionError
│   ├── DependencyError
│   └── PermissionError
├── GNNIntegrationError
│   ├── PyMDPIntegrationError
│   ├── RxInferIntegrationError
│   └── DisCoPyIntegrationError
└── GNNPerformanceWarning
    ├── MemoryWarning
    ├── ComputationWarning
    └── NetworkWarning
```

## Exception Classes

### Base Exception

```python
class GNNException(Exception):
    """Base exception for all GNN-related errors"""
    
    def __init__(self, message: str, error_code: str = None, 
                 file_path: str = None, line_number: int = None,
                 context: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.file_path = file_path
        self.line_number = line_number
        self.context = context or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'error_code': self.error_code,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()
```

### Syntax Errors

```python
class GNNSyntaxError(GNNException):
    """Base class for syntax-related errors"""
    pass

class MissingSectionError(GNNSyntaxError):
    """Raised when required GNN section is missing"""
    
    def __init__(self, section_name: str, file_path: str = None):
        message = f"Missing required section: {section_name}"
        super().__init__(
            message=message,
            error_code="SYN-101",
            file_path=file_path,
            context={'missing_section': section_name}
        )

class InvalidHeaderError(GNNSyntaxError):
    """Raised when section header format is incorrect"""
    
    def __init__(self, header: str, line_number: int = None, 
                 file_path: str = None):
        message = f"Invalid section header: {header}"
        super().__init__(
            message=message,
            error_code="SYN-102",
            file_path=file_path,
            line_number=line_number,
            context={'invalid_header': header}
        )

class MalformedVariableError(GNNSyntaxError):
    """Raised when variable declaration syntax is incorrect"""
    
    def __init__(self, variable_declaration: str, issue: str,
                 line_number: int = None, file_path: str = None):
        message = f"Malformed variable declaration: {variable_declaration} ({issue})"
        super().__init__(
            message=message,
            error_code="SYN-201",
            file_path=file_path,
            line_number=line_number,
            context={
                'variable_declaration': variable_declaration,
                'issue': issue
            }
        )

class InvalidConnectionError(GNNSyntaxError):
    """Raised when connection syntax is malformed"""
    
    def __init__(self, connection: str, issue: str,
                 line_number: int = None, file_path: str = None):
        message = f"Invalid connection: {connection} ({issue})"
        super().__init__(
            message=message,
            error_code="SYN-301",
            file_path=file_path,
            line_number=line_number,
            context={
                'connection': connection,
                'issue': issue
            }
        )
```

### Validation Errors

```python
class GNNValidationError(GNNException):
    """Base class for validation-related errors"""
    pass

class DimensionMismatchError(GNNValidationError):
    """Raised when matrix dimensions are incompatible"""
    
    def __init__(self, matrix_name: str, expected_shape: tuple, 
                 actual_shape: tuple, file_path: str = None):
        message = (f"Dimension mismatch in {matrix_name}: "
                  f"expected {expected_shape}, got {actual_shape}")
        super().__init__(
            message=message,
            error_code="VAL-101",
            file_path=file_path,
            context={
                'matrix_name': matrix_name,
                'expected_shape': expected_shape,
                'actual_shape': actual_shape
            }
        )

class MatrixValidationError(GNNValidationError):
    """Raised when matrix fails mathematical validation"""
    
    def __init__(self, matrix_name: str, validation_type: str,
                 details: str, file_path: str = None):
        message = f"Matrix validation failed for {matrix_name}: {details}"
        super().__init__(
            message=message,
            error_code="VAL-201",
            file_path=file_path,
            context={
                'matrix_name': matrix_name,
                'validation_type': validation_type,
                'details': details
            }
        )

class ConnectionValidationError(GNNValidationError):
    """Raised when connections reference undefined variables"""
    
    def __init__(self, connection: str, undefined_variables: list,
                 file_path: str = None):
        message = f"Connection references undefined variables: {undefined_variables}"
        super().__init__(
            message=message,
            error_code="SYN-302",
            file_path=file_path,
            context={
                'connection': connection,
                'undefined_variables': undefined_variables
            }
        )
```

### Runtime Errors

```python
class GNNRuntimeError(GNNException):
    """Base class for runtime-related errors"""
    pass

class PipelineExecutionError(GNNRuntimeError):
    """Raised when pipeline step fails to execute"""
    
    def __init__(self, step_name: str, step_number: int, 
                 exit_code: int, stderr: str = None):
        message = f"Pipeline step {step_number} ({step_name}) failed with exit code {exit_code}"
        super().__init__(
            message=message,
            error_code="RUN-101",
            context={
                'step_name': step_name,
                'step_number': step_number,
                'exit_code': exit_code,
                'stderr': stderr
            }
        )

class ResourceExhaustionError(GNNRuntimeError):
    """Raised when system resources are exhausted"""
    
    def __init__(self, resource_type: str, current_usage: str,
                 limit: str = None):
        message = f"Resource exhaustion: {resource_type} usage {current_usage}"
        if limit:
            message += f" exceeds limit {limit}"
        
        super().__init__(
            message=message,
            error_code="RUN-102",
            context={
                'resource_type': resource_type,
                'current_usage': current_usage,
                'limit': limit
            }
        )

class DependencyError(GNNRuntimeError):
    """Raised when required dependencies are missing or incompatible"""
    
    def __init__(self, dependency_name: str, required_version: str = None,
                 installed_version: str = None, issue: str = None):
        if issue:
            message = f"Dependency issue with {dependency_name}: {issue}"
        elif required_version and installed_version:
            message = (f"Version mismatch for {dependency_name}: "
                      f"required {required_version}, installed {installed_version}")
        else:
            message = f"Missing dependency: {dependency_name}"
        
        super().__init__(
            message=message,
            error_code="RUN-103",
            context={
                'dependency_name': dependency_name,
                'required_version': required_version,
                'installed_version': installed_version,
                'issue': issue
            }
        )
```

### Integration Errors

```python
class GNNIntegrationError(GNNException):
    """Base class for framework integration errors"""
    pass

class PyMDPIntegrationError(GNNIntegrationError):
    """Raised when PyMDP integration fails"""
    
    def __init__(self, operation: str, details: str):
        message = f"PyMDP integration failed during {operation}: {details}"
        super().__init__(
            message=message,
            error_code="INT-101",
            context={
                'operation': operation,
                'details': details,
                'framework': 'PyMDP'
            }
        )

class RxInferIntegrationError(GNNIntegrationError):
    """Raised when RxInfer integration fails"""
    
    def __init__(self, operation: str, details: str):
        message = f"RxInfer integration failed during {operation}: {details}"
        super().__init__(
            message=message,
            error_code="INT-201",
            context={
                'operation': operation,
                'details': details,
                'framework': 'RxInfer'
            }
        )

class DisCoPyIntegrationError(GNNIntegrationError):
    """Raised when DisCoPy integration fails"""
    
    def __init__(self, operation: str, details: str):
        message = f"DisCoPy integration failed during {operation}: {details}"
        super().__init__(
            message=message,
            error_code="INT-301",
            context={
                'operation': operation,
                'details': details,
                'framework': 'DisCoPy'
            }
        )
```

### Performance Warnings

```python
class GNNPerformanceWarning(UserWarning):
    """Base class for performance-related warnings"""
    
    def __init__(self, message: str, metric: str, threshold: str,
                 current_value: str, recommendation: str = None):
        super().__init__(message)
        self.metric = metric
        self.threshold = threshold
        self.current_value = current_value
        self.recommendation = recommendation

class MemoryWarning(GNNPerformanceWarning):
    """Warning for high memory usage"""
    
    def __init__(self, current_mb: float, threshold_mb: float):
        message = f"High memory usage: {current_mb:.1f}MB (threshold: {threshold_mb:.1f}MB)"
        super().__init__(
            message=message,
            metric="memory_usage_mb",
            threshold=str(threshold_mb),
            current_value=str(current_mb),
            recommendation="Consider reducing model complexity or using chunked processing"
        )

class ComputationWarning(GNNPerformanceWarning):
    """Warning for long computation times"""
    
    def __init__(self, current_time: float, threshold_time: float):
        message = f"Long computation time: {current_time:.1f}s (threshold: {threshold_time:.1f}s)"
        super().__init__(
            message=message,
            metric="computation_time_s",
            threshold=str(threshold_time),
            current_value=str(current_time),
            recommendation="Consider model simplification or parallel processing"
        )
```

## Error Handling Patterns

### Basic Error Handling

```python
from src.gnn import GNNModel
from src.gnn.exceptions import GNNException, GNNSyntaxError, GNNValidationError

def load_and_validate_model(file_path: str) -> GNNModel:
    """Load and validate GNN model with comprehensive error handling"""
    
    try:
        # Load model
        model = GNNModel.from_file(file_path)
        
        # Validate model
        from src.gnn_type_checker import TypeChecker
        checker = TypeChecker(strict_mode=True)
        result = checker.check_model(model)
        
        if result.errors:
            raise GNNValidationError(
                message=f"Model validation failed: {len(result.errors)} errors",
                file_path=file_path,
                context={'errors': result.errors}
            )
        
        return model
        
    except GNNSyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        print(f"Error code: {e.error_code}")
        if e.line_number:
            print(f"Line {e.line_number}: Check syntax near this location")
        raise
        
    except GNNValidationError as e:
        print(f"Validation error in {file_path}: {e}")
        print(f"Context: {e.context}")
        raise
        
    except GNNException as e:
        print(f"General GNN error: {e}")
        print(f"Error details: {e.to_dict()}")
        raise
        
    except Exception as e:
        # Wrap unexpected errors
        raise GNNRuntimeError(
            message=f"Unexpected error loading model: {str(e)}",
            error_code="RUN-999",
            file_path=file_path,
            context={'original_error': str(e)}
        )
```

### Pipeline Error Handling

```python
from src.pipeline import PipelineExecutor
from src.gnn.exceptions import PipelineExecutionError

class RobustPipelineExecutor(PipelineExecutor):
    """Pipeline executor with enhanced error handling and recovery"""
    
    def __init__(self, retry_attempts: int = 3, 
                 continue_on_warnings: bool = True):
        super().__init__()
        self.retry_attempts = retry_attempts
        self.continue_on_warnings = continue_on_warnings
        self.execution_log = []
    
    def execute_step(self, step_number: int, **kwargs) -> bool:
        """Execute pipeline step with retry logic and error recovery"""
        
        for attempt in range(self.retry_attempts):
            try:
                result = super().execute_step(step_number, **kwargs)
                
                # Log successful execution
                self.execution_log.append({
                    'step': step_number,
                    'attempt': attempt + 1,
                    'status': 'success',
                    'timestamp': datetime.utcnow()
                })
                
                return result
                
            except PipelineExecutionError as e:
                # Log failed attempt
                self.execution_log.append({
                    'step': step_number,
                    'attempt': attempt + 1,
                    'status': 'failed',
                    'error': e.to_dict(),
                    'timestamp': datetime.utcnow()
                })
                
                if attempt < self.retry_attempts - 1:
                    print(f"Step {step_number} failed (attempt {attempt + 1}), retrying...")
                    self._apply_recovery_strategy(step_number, e)
                else:
                    print(f"Step {step_number} failed after {self.retry_attempts} attempts")
                    raise
                    
            except GNNPerformanceWarning as w:
                if self.continue_on_warnings:
                    print(f"Performance warning in step {step_number}: {w}")
                    return True
                else:
                    raise
    
    def _apply_recovery_strategy(self, step_number: int, error: PipelineExecutionError):
        """Apply recovery strategies based on error type"""
        
        if error.error_code == "RUN-102":  # Memory exhaustion
            print("Applying memory recovery: clearing caches...")
            import gc
            gc.collect()
            
        elif error.error_code == "RUN-103":  # Dependency error
            print("Applying dependency recovery: checking installations...")
            self._validate_dependencies()
            
        elif error.error_code == "RUN-104":  # Permission error
            print("Applying permission recovery: checking file permissions...")
            self._fix_permissions()
```

### Context Manager for Error Handling

```python
from contextlib import contextmanager
import logging

@contextmanager
def gnn_error_context(operation_name: str, file_path: str = None):
    """Context manager for standardized GNN error handling"""
    
    logger = logging.getLogger('gnn.errors')
    
    try:
        logger.info(f"Starting operation: {operation_name}")
        yield
        logger.info(f"Completed operation: {operation_name}")
        
    except GNNSyntaxError as e:
        logger.error(f"Syntax error in {operation_name}: {e}")
        logger.error(f"File: {e.file_path or file_path}")
        logger.error(f"Line: {e.line_number}")
        raise
        
    except GNNValidationError as e:
        logger.error(f"Validation error in {operation_name}: {e}")
        logger.error(f"Context: {e.context}")
        raise
        
    except GNNRuntimeError as e:
        logger.error(f"Runtime error in {operation_name}: {e}")
        logger.error(f"Error context: {e.context}")
        raise
        
    except Exception as e:
        logger.exception(f"Unexpected error in {operation_name}")
        raise GNNRuntimeError(
            message=f"Unexpected error in {operation_name}: {str(e)}",
            error_code="RUN-999",
            file_path=file_path,
            context={'operation': operation_name, 'original_error': str(e)}
        )

# Usage
with gnn_error_context("model_loading", "my_model.md"):
    model = GNNModel.from_file("my_model.md")
    
with gnn_error_context("type_checking"):
    checker = TypeChecker()
    result = checker.check_model(model)
```

## Error Recovery Strategies

### Automatic Error Recovery

```python
class ErrorRecoveryManager:
    """Manages automatic error recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies = {
            'SYN-101': self._recover_missing_section,
            'SYN-201': self._recover_variable_syntax,
            'VAL-101': self._recover_dimension_mismatch,
            'RUN-102': self._recover_memory_exhaustion,
            'RUN-103': self._recover_dependency_error,
        }
    
    def attempt_recovery(self, error: GNNException) -> bool:
        """Attempt automatic recovery for the given error"""
        
        if error.error_code in self.recovery_strategies:
            try:
                return self.recovery_strategies[error.error_code](error)
            except Exception as recovery_error:
                print(f"Recovery failed: {recovery_error}")
                return False
        
        return False
    
    def _recover_missing_section(self, error: MissingSectionError) -> bool:
        """Attempt to recover from missing section error"""
        
        section_templates = {
            'ModelName': '## ModelName\nGenerated_Model',
            'StateSpaceBlock': '## StateSpaceBlock\n# Auto-generated placeholder',
            'Connections': '## Connections\n# Auto-generated placeholder'
        }
        
        missing_section = error.context.get('missing_section')
        if missing_section in section_templates:
            print(f"Auto-generating missing section: {missing_section}")
            # Implementation would modify the file
            return True
        
        return False
    
    def _recover_memory_exhaustion(self, error: ResourceExhaustionError) -> bool:
        """Attempt to recover from memory exhaustion"""
        
        import gc
        print("Attempting memory recovery...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if hasattr(self, '_clear_caches'):
            self._clear_caches()
        
        return True
    
    def _recover_dependency_error(self, error: DependencyError) -> bool:
        """Attempt to recover from dependency errors"""
        
        dependency = error.context.get('dependency_name')
        if dependency:
            print(f"Attempting to install missing dependency: {dependency}")
            # Implementation would use pip/conda to install
            return True
        
        return False
```

## Error Reporting and Analytics

### Error Aggregation

```python
class ErrorAnalytics:
    """Collect and analyze error patterns for system improvement"""
    
    def __init__(self):
        self.error_history = []
        self.error_patterns = {}
    
    def record_error(self, error: GNNException):
        """Record error for analytics"""
        
        error_record = {
            'timestamp': datetime.utcnow(),
            'error_type': error.__class__.__name__,
            'error_code': error.error_code,
            'file_path': error.file_path,
            'line_number': error.line_number,
            'context': error.context
        }
        
        self.error_history.append(error_record)
        self._update_patterns(error_record)
    
    def _update_patterns(self, error_record: dict):
        """Update error pattern statistics"""
        
        error_code = error_record['error_code']
        if error_code not in self.error_patterns:
            self.error_patterns[error_code] = {
                'count': 0,
                'files': set(),
                'first_seen': error_record['timestamp'],
                'last_seen': error_record['timestamp']
            }
        
        pattern = self.error_patterns[error_code]
        pattern['count'] += 1
        pattern['files'].add(error_record['file_path'])
        pattern['last_seen'] = error_record['timestamp']
    
    def generate_report(self) -> dict:
        """Generate error analytics report"""
        
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {'total_errors': 0, 'patterns': {}}
        
        # Convert sets to lists for JSON serialization
        patterns = {}
        for code, data in self.error_patterns.items():
            patterns[code] = {
                'count': data['count'],
                'frequency': data['count'] / total_errors,
                'affected_files': list(data['files']),
                'first_seen': data['first_seen'].isoformat(),
                'last_seen': data['last_seen'].isoformat()
            }
        
        return {
            'total_errors': total_errors,
            'patterns': patterns,
            'most_common': sorted(patterns.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)[:5]
        }
```

## Integration with Logging

```python
import logging
from typing import Optional

class GNNErrorLogger:
    """Specialized logger for GNN errors with structured output"""
    
    def __init__(self, logger_name: str = 'gnn.errors'):
        self.logger = logging.getLogger(logger_name)
        self._setup_formatter()
    
    def _setup_formatter(self):
        """Setup structured logging formatter"""
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | '
            '%(error_code)s | %(file_path)s:%(line_number)s | %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_error(self, error: GNNException, level: int = logging.ERROR):
        """Log GNN error with structured information"""
        
        extra = {
            'error_code': error.error_code or 'UNKNOWN',
            'file_path': error.file_path or 'N/A',
            'line_number': error.line_number or 'N/A',
            'error_context': error.context
        }
        
        self.logger.log(level, str(error), extra=extra)
    
    def log_recovery_attempt(self, error: GNNException, success: bool):
        """Log error recovery attempts"""
        
        status = "SUCCEEDED" if success else "FAILED"
        message = f"Error recovery {status} for {error.error_code}"
        
        extra = {
            'error_code': error.error_code or 'UNKNOWN',
            'file_path': error.file_path or 'N/A',
            'line_number': error.line_number or 'N/A',
            'recovery_status': status
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, message, extra=extra)
```

---

## Best Practices

### Error Handling Guidelines

1. **Always catch specific exceptions first**: Handle `GNNSyntaxError` before `GNNException`
2. **Provide context**: Include file paths, line numbers, and relevant context
3. **Use appropriate error codes**: Follow the error taxonomy for consistent coding
4. **Log errors systematically**: Use structured logging for analytics
5. **Implement recovery where possible**: Attempt automatic recovery for common issues
6. **Preserve original errors**: When wrapping exceptions, preserve the original error information

### Performance Considerations

1. **Lazy error context**: Only compute expensive context when needed
2. **Batch error reporting**: Collect multiple errors before reporting
3. **Cache error patterns**: Avoid recomputing known error patterns
4. **Limit error history**: Implement rotation for long-running applications

---

## Related Documentation

- **[Error Taxonomy](error_taxonomy.md)**: Complete error classification system
- **[Debugging Workflows](debugging_workflows.md)**: Step-by-step debugging procedures
- **[Common Errors](common_errors.md)**: Frequent issues and solutions
- **[Performance Guide](../performance/README.md)**: Performance optimization strategies

---

**Last Updated: October 2025  
**API Error Reference Version**: 1.0  
**Exception Classes**: 25+ specialized exception types  
**Status**: Production-Ready 