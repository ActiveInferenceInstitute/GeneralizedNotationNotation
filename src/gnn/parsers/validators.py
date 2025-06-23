"""
GNN Validators - Model Validation and Constraint Checking

This module provides comprehensive validation capabilities for GNN models
including structure validation, semantic checking, and constraint verification.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .common import (
    GNNInternalRepresentation, Variable, Connection, Parameter,
    VariableType, DataType, ConnectionType, ValidationError, ValidationWarning
)

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """A validation issue found in a GNN model."""
    severity: ValidationSeverity
    message: str
    component: str  # variable name, connection, etc.
    component_type: str  # 'variable', 'connection', 'parameter', etc.
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.severity.value.upper()}: {self.component_type} '{self.component}': {self.message}"

@dataclass
class ValidationResult:
    """Result of model validation."""
    success: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.success = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        elif issue.severity == ValidationSeverity.INFO:
            self.info.append(issue)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.success and not self.warnings:
            return "Validation passed with no issues"
        elif self.success:
            return f"Validation passed with {len(self.warnings)} warnings"
        else:
            return f"Validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings"

class GNNValidator:
    """Comprehensive validator for GNN models."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize the GNN validator.
        
        Args:
            strict: Whether to perform strict validation (warnings become errors)
        """
        self.strict = strict
        
    def validate(self, model: GNNInternalRepresentation) -> ValidationResult:
        """
        Validate a GNN model comprehensively.
        
        Args:
            model: GNN model to validate
            
        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(success=True)
        
        # Basic structure validation
        self._validate_basic_structure(model, result)
        
        # Variable validation
        self._validate_variables(model, result)
        
        # Connection validation
        self._validate_connections(model, result)
        
        # Parameter validation
        self._validate_parameters(model, result)
        
        # Semantic validation
        self._validate_semantics(model, result)
        
        # Active Inference specific validation
        self._validate_active_inference(model, result)
        
        return result
    
    def _validate_basic_structure(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate basic model structure."""
        # Check model name
        if not model.model_name or not model.model_name.strip():
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Model name is required",
                component="model",
                component_type="structure"
            ))
        
        # Check version
        if not model.version:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Model version not specified",
                component="model",
                component_type="structure"
            ))
        
        # Check minimum components
        if not model.variables:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Model has no variables defined",
                component="model",
                component_type="structure"
            ))
    
    def _validate_variables(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate variables in the model."""
        variable_names = set()
        
        for variable in model.variables:
            # Check for duplicate names
            if variable.name in variable_names:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Duplicate variable name",
                    component=variable.name,
                    component_type="variable"
                ))
            else:
                variable_names.add(variable.name)
            
            # Validate individual variable
            self._validate_variable(variable, result)
    
    def _validate_variable(self, variable: Variable, result: ValidationResult):
        """Validate a single variable."""
        # Check name
        if not variable.name or not variable.name.strip():
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Variable name is required",
                component=variable.name or "unnamed",
                component_type="variable"
            ))
        
        # Check name format
        if variable.name and not variable.name.replace('_', '').replace('-', '').isalnum():
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Variable name contains special characters",
                component=variable.name,
                component_type="variable",
                details={"name": variable.name}
            ))
        
        # Check dimensions
        if variable.dimensions:
            for dim in variable.dimensions:
                if not isinstance(dim, int) or dim <= 0:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Variable dimensions must be positive integers",
                        component=variable.name,
                        component_type="variable",
                        details={"dimensions": variable.dimensions}
                    ))
                    break
        
        # Validate Active Inference specific constraints
        self._validate_variable_active_inference(variable, result)
    
    def _validate_variable_active_inference(self, variable: Variable, result: ValidationResult):
        """Validate Active Inference specific variable constraints."""
        name_lower = variable.name.lower()
        
        # Check matrix variables have appropriate dimensions
        if variable.var_type in [VariableType.LIKELIHOOD_MATRIX, VariableType.TRANSITION_MATRIX]:
            if len(variable.dimensions) != 2:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Matrix variables should have 2 dimensions",
                    component=variable.name,
                    component_type="variable",
                    details={"expected_dims": 2, "actual_dims": len(variable.dimensions)}
                ))
        
        # Check vector variables have appropriate dimensions
        elif variable.var_type in [VariableType.PREFERENCE_VECTOR, VariableType.PRIOR_VECTOR]:
            if len(variable.dimensions) != 1:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Vector variables should have 1 dimension",
                    component=variable.name,
                    component_type="variable",
                    details={"expected_dims": 1, "actual_dims": len(variable.dimensions)}
                ))
        
        # Check naming conventions
        if variable.var_type == VariableType.HIDDEN_STATE and not any(keyword in name_lower for keyword in ['state', 'hidden', 's_']):
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Hidden state variable name doesn't follow common conventions",
                component=variable.name,
                component_type="variable",
                details={"suggestions": ["state_", "hidden_", "s_"]}
            ))
        
        # Check data type compatibility
        if variable.var_type == VariableType.ACTION and variable.data_type not in [DataType.BINARY, DataType.CATEGORICAL, DataType.INTEGER]:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Action variables typically use discrete data types",
                component=variable.name,
                component_type="variable",
                details={"current_type": variable.data_type.value, "suggested_types": ["binary", "categorical", "integer"]}
            ))
    
    def _validate_connections(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate connections in the model."""
        variable_names = {var.name for var in model.variables}
        
        for i, connection in enumerate(model.connections):
            # Check source variables exist
            for source_var in connection.source_variables:
                if source_var not in variable_names:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Connection references non-existent source variable",
                        component=f"connection_{i}",
                        component_type="connection",
                        details={"source_variable": source_var, "available_variables": list(variable_names)}
                    ))
            
            # Check target variables exist
            for target_var in connection.target_variables:
                if target_var not in variable_names:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Connection references non-existent target variable",
                        component=f"connection_{i}",
                        component_type="connection",
                        details={"target_variable": target_var, "available_variables": list(variable_names)}
                    ))
            
            # Check for self-loops in directed connections
            if connection.connection_type == ConnectionType.DIRECTED:
                common_vars = set(connection.source_variables) & set(connection.target_variables)
                if common_vars:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Directed connection contains self-loops",
                        component=f"connection_{i}",
                        component_type="connection",
                        details={"self_loop_variables": list(common_vars)}
                    ))
            
            # Validate Active Inference specific connections
            self._validate_connection_active_inference(connection, model, result, i)
    
    def _validate_connection_active_inference(self, connection: Connection, model: GNNInternalRepresentation, 
                                             result: ValidationResult, index: int):
        """Validate Active Inference specific connection constraints."""
        var_lookup = {var.name: var for var in model.variables}
        
        # Check observation model connections (A matrix)
        for source_var_name in connection.source_variables:
            source_var = var_lookup.get(source_var_name)
            if source_var and source_var.var_type == VariableType.HIDDEN_STATE:
                for target_var_name in connection.target_variables:
                    target_var = var_lookup.get(target_var_name)
                    if target_var and target_var.var_type == VariableType.OBSERVATION:
                        # This should be mediated by A matrix
                        a_matrices = [var for var in model.variables if var.var_type == VariableType.LIKELIHOOD_MATRIX]
                        if not a_matrices:
                            result.add_issue(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message="Hidden state to observation connection without A matrix",
                                component=f"connection_{index}",
                                component_type="connection",
                                details={"source": source_var_name, "target": target_var_name}
                            ))
        
        # Check transition model connections (B matrix)
        for source_var_name in connection.source_variables:
            source_var = var_lookup.get(source_var_name)
            if source_var and source_var.var_type == VariableType.HIDDEN_STATE:
                for target_var_name in connection.target_variables:
                    target_var = var_lookup.get(target_var_name)
                    if target_var and target_var.var_type == VariableType.HIDDEN_STATE and source_var_name != target_var_name:
                        # This might need B matrix
                        b_matrices = [var for var in model.variables if var.var_type == VariableType.TRANSITION_MATRIX]
                        if not b_matrices:
                            result.add_issue(ValidationIssue(
                                severity=ValidationSeverity.INFO,
                                message="State-to-state connection might need B matrix",
                                component=f"connection_{index}",
                                component_type="connection",
                                details={"source": source_var_name, "target": target_var_name}
                            ))
    
    def _validate_parameters(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate parameters in the model."""
        parameter_names = set()
        
        for parameter in model.parameters:
            # Check for duplicate names
            if parameter.name in parameter_names:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Duplicate parameter name",
                    component=parameter.name,
                    component_type="parameter"
                ))
            else:
                parameter_names.add(parameter.name)
            
            # Check parameter name
            if not parameter.name or not parameter.name.strip():
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Parameter name is required",
                    component=parameter.name or "unnamed",
                    component_type="parameter"
                ))
    
    def _validate_semantics(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate semantic consistency of the model."""
        # Check for unreferenced variables
        referenced_vars = set()
        for connection in model.connections:
            referenced_vars.update(connection.source_variables)
            referenced_vars.update(connection.target_variables)
        
        for ontology_mapping in model.ontology_mappings:
            referenced_vars.add(ontology_mapping.variable_name)
        
        all_vars = {var.name for var in model.variables}
        unreferenced = all_vars - referenced_vars
        
        for var_name in unreferenced:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Variable is not referenced in connections or ontology mappings",
                component=var_name,
                component_type="variable",
                details={"suggestion": "Consider adding connections or removing if unused"}
            ))
        
        # Check for disconnected components
        self._check_connectivity(model, result)
    
    def _check_connectivity(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Check if the model graph is connected."""
        if not model.variables or not model.connections:
            return
        
        # Build adjacency list
        graph = {}
        for var in model.variables:
            graph[var.name] = set()
        
        for connection in model.connections:
            for source in connection.source_variables:
                for target in connection.target_variables:
                    if source in graph and target in graph:
                        graph[source].add(target)
                        if connection.connection_type != ConnectionType.DIRECTED:
                            graph[target].add(source)
        
        # Find connected components
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                dfs(neighbor, component)
        
        for var_name in graph:
            if var_name not in visited:
                component = set()
                dfs(var_name, component)
                components.append(component)
        
        if len(components) > 1:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Model has disconnected components",
                component="model",
                component_type="structure",
                details={"num_components": len(components), "components": [list(comp) for comp in components]}
            ))
    
    def _validate_active_inference(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate Active Inference specific requirements."""
        # Check for required Active Inference components
        var_types = {var.var_type for var in model.variables}
        
        # Should have hidden states
        if VariableType.HIDDEN_STATE not in var_types:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Active Inference models typically require hidden states",
                component="model",
                component_type="active_inference",
                details={"missing_component": "hidden_state"}
            ))
        
        # Should have observations
        if VariableType.OBSERVATION not in var_types:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Active Inference models typically require observations",
                component="model",
                component_type="active_inference",
                details={"missing_component": "observation"}
            ))
        
        # Check for A matrix (observation model)
        if VariableType.LIKELIHOOD_MATRIX not in var_types:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No A matrix (observation model) found",
                component="model",
                component_type="active_inference",
                details={"missing_component": "likelihood_matrix"}
            ))
        
        # Check time specification for dynamic models
        has_temporal_connections = any(
            connection.description and 'temporal' in connection.description.lower()
            for connection in model.connections
        )
        
        if has_temporal_connections and not model.time_specification:
            result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Dynamic model should have time specification",
                component="model",
                component_type="active_inference",
                details={"suggestion": "Add time specification for temporal dynamics"}
            ))
        
        # Validate matrix dimensions consistency
        self._validate_matrix_dimensions(model, result)
    
    def _validate_matrix_dimensions(self, model: GNNInternalRepresentation, result: ValidationResult):
        """Validate Active Inference matrix dimension consistency."""
        # Get dimensions of different variable types
        hidden_states = [var for var in model.variables if var.var_type == VariableType.HIDDEN_STATE]
        observations = [var for var in model.variables if var.var_type == VariableType.OBSERVATION]
        a_matrices = [var for var in model.variables if var.var_type == VariableType.LIKELIHOOD_MATRIX]
        b_matrices = [var for var in model.variables if var.var_type == VariableType.TRANSITION_MATRIX]
        
        # Check A matrix dimensions
        for a_matrix in a_matrices:
            if len(a_matrix.dimensions) == 2:
                obs_dim, state_dim = a_matrix.dimensions
                
                # Check consistency with observations
                for obs_var in observations:
                    if obs_var.dimensions and obs_var.dimensions[0] != obs_dim:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="A matrix dimension inconsistent with observation variable",
                            component=a_matrix.name,
                            component_type="variable",
                            details={
                                "a_matrix_obs_dim": obs_dim,
                                "observation_dim": obs_var.dimensions[0],
                                "observation_var": obs_var.name
                            }
                        ))
                
                # Check consistency with hidden states
                for state_var in hidden_states:
                    if state_var.dimensions and state_var.dimensions[0] != state_dim:
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="A matrix dimension inconsistent with hidden state variable",
                            component=a_matrix.name,
                            component_type="variable",
                            details={
                                "a_matrix_state_dim": state_dim,
                                "state_dim": state_var.dimensions[0],
                                "state_var": state_var.name
                            }
                        ))
        
        # Check B matrix dimensions
        for b_matrix in b_matrices:
            if len(b_matrix.dimensions) >= 2:
                state_dim_t1, state_dim_t0 = b_matrix.dimensions[:2]
                
                if state_dim_t1 != state_dim_t0:
                    result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message="B matrix has different input/output state dimensions",
                        component=b_matrix.name,
                        component_type="variable",
                        details={
                            "input_state_dim": state_dim_t0,
                            "output_state_dim": state_dim_t1
                        }
                    )) 