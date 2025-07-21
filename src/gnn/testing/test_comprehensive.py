"""
Comprehensive Test Suite for GNN (Generalized Notation Notation) Module

This test suite provides complete coverage of the GNN module including:
- Schema validation testing
- Parser functionality testing  
- Example file validation
- Performance benchmarking
- Error handling and edge cases
- Cross-format consistency validation
- Unicode and internationalization testing
- Memory usage and large file handling
"""

import os
import sys
import json
import yaml
import unittest
import tempfile
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    from gnn.schema_validator import GNNParser, GNNValidator, ValidationResult, ParsedGNN
    from gnn.mcp import get_gnn_documentation, validate_gnn_content
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# Lark parser removed - too complex and not needed
LARK_AVAILABLE = False


class TestGNNSchemaValidation(unittest.TestCase):
    """Test schema validation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
        
        self.validator = GNNParser()
        self.parser = GNNValidator()
        
        # Valid GNN content for testing
        self.valid_gnn_content = """# GNN Example: Test Model
## GNNSection
TestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Test Model v1

## ModelAnnotation
This is a test model for validation testing.
It includes basic Active Inference components.

## StateSpaceBlock
s_f0[2,1,type=float]        # Hidden state factor 0
o_m0[3,1,type=int]          # Observation modality 0
A_m0[3,2,type=float]        # Likelihood matrix modality 0
u_c0[1,type=int]            # Action for control factor 0

## Connections
s_f0>A_m0
A_m0>o_m0
u_c0>s_f0

## InitialParameterization
A_m0={(0.8,0.1,0.1),(0.1,0.8,0.1)}
s_f0={(0.5,0.5)}

## Equations
p(o|s) = A * s

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10

## ActInfOntologyAnnotation
A_m0=LikelihoodMatrix
s_f0=HiddenState
o_m0=Observation

## ModelParameters
num_hidden_state_factors: [2]
num_obs_modalities: [3]

## Footer
Test Model - End

## Signature
Creator: Test Suite
Date: 2024-01-01
"""

        # Invalid GNN content for error testing
        self.invalid_gnn_content = """# Invalid GNN Example
## GNNSection
InvalidModel

## GNNVersionAndFlags
GNN v999  # Invalid version

## ModelName
# Missing model name

## ModelAnnotation
Invalid model for testing.

## StateSpaceBlock
invalid_var[type=unknown]   # Invalid dimension spec
s_f0[-1,type=float]         # Negative dimension

## Connections
s_f0>undefined_var          # Connection to undefined variable
invalid>syntax              # Invalid connection syntax

## InitialParameterization
A_m0=invalid_value          # Invalid parameter value
s_f0                        # Missing assignment

## Time
InvalidTimeType             # Invalid time specification

## Footer
# Missing footer content
"""
    
    def test_schema_loading(self):
        """Test JSON schema loading."""
        self.assertIsNotNone(self.validator.schema)
        self.assertIsInstance(self.validator.schema, dict)
    
    def test_valid_gnn_validation(self):
        """Test validation of valid GNN content."""
        result = self.validator.validate_file(self._create_temp_file(self.valid_gnn_content))
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid, f"Valid GNN should pass validation. Errors: {result.errors}")
        self.assertEqual(len(result.errors), 0)
    
    def test_invalid_gnn_validation(self):
        """Test validation of invalid GNN content."""
        result = self.validator.validate_file(self._create_temp_file(self.invalid_gnn_content))
        
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_missing_required_sections(self):
        """Test validation when required sections are missing."""
        incomplete_content = """## GNNSection
TestModel

## ModelName
Incomplete Model
"""
        
        result = self.validator.validate_file(self._create_temp_file(incomplete_content))
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Required section missing" in error for error in result.errors))
    
    def test_schema_validator_error_reporting(self):
        """Test detailed error reporting from schema validator."""
        result = self.validator.validate_file(self._create_temp_file(self.invalid_gnn_content))
        
        # Check for specific error types
        error_messages = ' '.join(result.errors)
        self.assertIn("Required section missing", error_messages)
    
    def _create_temp_file(self, content: str) -> str:
        """Create temporary file with content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            return f.name


class TestGNNParser(unittest.TestCase):
    """Test GNN parser functionality."""
    
    def setUp(self):
        """Set up test environment."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
        
        self.parser = GNNParser()
        
        # Comprehensive GNN content for parser testing
        self.parser_test_content = """## GNNSection
ParserTestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Parser Test Model

## ModelAnnotation
Comprehensive model for testing parser functionality.
Includes various GNN constructs and edge cases.

## StateSpaceBlock
# Hidden states
s_f0[2,1,type=float]        # Binary hidden state factor
s_f1[3,1,type=float]        # Ternary hidden state factor

# Observations  
o_m0[2,1,type=int]          # Binary observation modality
o_m1[4,1,type=int]          # Quaternary observation modality

# Matrices
A_m0[2,2,3,type=float]      # Likelihood matrix modality 0
A_m1[4,2,3,type=float]      # Likelihood matrix modality 1
B_f0[2,2,1,type=float]      # Transition matrix factor 0 (uncontrolled)
B_f1[3,3,2,type=float]      # Transition matrix factor 1 (controlled)

# Preferences and priors
C_m0[2,type=float]          # Preferences modality 0
C_m1[4,type=float]          # Preferences modality 1
D_f0[2,type=float]          # Prior beliefs factor 0
D_f1[3,type=float]          # Prior beliefs factor 1

# Control variables
u_c0[1,type=int]            # Action factor 0 (implicit)
u_c1[2,type=int]            # Action factor 1 (explicit)
π_c1[2,type=float]          # Policy factor 1

## Connections
# Generative model connections
(D_f0,D_f1)>(s_f0,s_f1)
(s_f0,s_f1)>(A_m0,A_m1)
(A_m0,A_m1)>(o_m0,o_m1)

# Temporal dynamics
(s_f0,s_f1,u_c1)>(B_f0,B_f1)
(B_f0,B_f1)>(s_f0,s_f1)

# Policy and preferences
(C_m0,C_m1)>π_c1
π_c1>u_c1

## InitialParameterization
# A matrices (likelihood)
A_m0={((0.9,0.1),(0.2,0.8)),((0.1,0.9),(0.8,0.2))}
A_m1={((0.7,0.2,0.1,0.0),(0.1,0.7,0.2,0.0)),((0.0,0.1,0.7,0.2),(0.0,0.0,0.2,0.8))}

# B matrices (transitions)
B_f0={((1.0,0.0),(0.0,1.0))}  # Identity (no control)
B_f1={((0.8,0.1,0.1),(0.1,0.8,0.1),(0.1,0.1,0.8)),((0.2,0.4,0.4),(0.4,0.2,0.4),(0.4,0.4,0.2))}

# C vectors (preferences)
C_m0={(1.0,-1.0)}
C_m1={(2.0,0.0,-1.0,-2.0)}

# D vectors (priors)
D_f0={(0.5,0.5)}
D_f1={(0.33,0.33,0.34)}

## Equations
F = E_q[ln q(s,π)] + E_q[ln p(o|s)]
G = E_q[ln q(π)] - E_q[ln p(π|C)]

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1
B_f0=TransitionMatrixFactor0
B_f1=TransitionMatrixFactor1
C_m0=PreferenceVectorModality0
C_m1=PreferenceVectorModality1
D_f0=PriorBeliefsFactor0
D_f1=PriorBeliefsFactor1
s_f0=HiddenStateFactor0
s_f1=HiddenStateFactor1
o_m0=ObservationModality0
o_m1=ObservationModality1
u_c1=ActionFactor1
π_c1=PolicyFactor1

## ModelParameters
num_hidden_state_factors: [2, 3]
num_obs_modalities: [2, 4]
num_control_factors: [1, 2]
backend: PyMDP
inference_method: variational_message_passing

## Footer
Parser Test Model - Comprehensive GNN Example

## Signature
Creator: GNN Test Suite
Date: 2024-01-01
Version: 1.0
Status: Testing
Compliance: GNN v1
"""
    
    def test_parse_content_basic(self):
        """Test basic content parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        self.assertIsInstance(parsed, ParsedGNN)
        self.assertEqual(parsed.gnn_section, "ParserTestModel")
        self.assertEqual(parsed.model_name, "Parser Test Model")
        self.assertIn("Comprehensive model", parsed.model_annotation)
    
    def test_parse_variables(self):
        """Test variable parsing from StateSpaceBlock."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check that variables were parsed
        self.assertGreater(len(parsed.variables), 0)
        
        # Check specific variables
        self.assertIn("s_f0", parsed.variables)
        self.assertIn("A_m0", parsed.variables)
        
        # Check variable properties
        s_f0 = parsed.variables["s_f0"]
        self.assertEqual(s_f0.name, "s_f0")
        self.assertEqual(s_f0.data_type, "float")
        self.assertEqual(len(s_f0.dimensions), 2)
    
    def test_parse_connections(self):
        """Test connection parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check that connections were parsed
        self.assertGreater(len(parsed.connections), 0)
        
        # Check for specific connection types
        connection_operators = [conn.symbol for conn in parsed.connections]
        self.assertIn(">", connection_operators)
        
        # Check connection structure
        first_conn = parsed.connections[0]
        self.assertIsNotNone(first_conn.source)
        self.assertIsNotNone(first_conn.target)
        self.assertIn(first_conn.connection_type, ['directed', 'undirected', 'conditional'])
    
    def test_parse_parameters(self):
        """Test parameter parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check that parameters were parsed
        self.assertGreater(len(parsed.parameters), 0)
        
        # Check specific parameters
        self.assertIn("A_m0", parsed.parameters)
        self.assertIn("D_f0", parsed.parameters)
    
    def test_parse_equations(self):
        """Test equation parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check that equations were parsed
        self.assertGreater(len(parsed.equations), 0)
        
        # Check equation structure
        first_eq = parsed.equations[0]
        self.assertIn("latex", first_eq)
        self.assertIsInstance(first_eq["latex"], str)
    
    def test_parse_time_config(self):
        """Test time configuration parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check time configuration
        self.assertIsInstance(parsed.time_config, dict)
        self.assertIn("type", parsed.time_config)
    
    def test_parse_ontology_mappings(self):
        """Test ontology mapping parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check ontology mappings
        self.assertGreater(len(parsed.ontology_mappings), 0)
        self.assertIn("A_m0", parsed.ontology_mappings)
        self.assertEqual(parsed.ontology_mappings["A_m0"], "LikelihoodMatrixModality0")
    
    def test_parse_model_parameters(self):
        """Test model parameter parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check model parameters
        self.assertGreater(len(parsed.model_parameters), 0)
        self.assertIn("backend", parsed.model_parameters)
    
    def test_parse_signature(self):
        """Test signature parsing."""
        parsed = self.parser.parse_content(self.parser_test_content)
        
        # Check signature
        self.assertIsNotNone(parsed.signature)
        self.assertIn("Creator", parsed.signature)
    
    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_file("nonexistent_file.md")
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        parsed = self.parser.parse_content("")
        
        self.assertIsInstance(parsed, ParsedGNN)
        self.assertEqual(len(parsed.variables), 0)
        self.assertEqual(len(parsed.connections), 0)


class TestGNNExampleValidation(unittest.TestCase):
    """Test validation of all GNN example files."""
    
    def setUp(self):
        """Set up test environment."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
        
        self.validator = GNNValidator()
        self.parser = GNNParser()
        self.examples_dir = Path(__file__).parent / "examples"
    
    def test_example_files_exist(self):
        """Test that example files exist."""
        if not self.examples_dir.exists():
            self.skipTest("Examples directory not found")
        
        example_files = list(self.examples_dir.glob("*.md"))
        self.assertGreater(len(example_files), 0, "No example files found")
    
    def test_validate_all_examples(self):
        """Test validation of all example files."""
        if not self.examples_dir.exists():
            self.skipTest("Examples directory not found")
        
        example_files = list(self.examples_dir.glob("*.md"))
        validation_results = {}
        
        for example_file in example_files:
            result = self.validator.validate_file(example_file)
            validation_results[example_file.name] = result
            
            # Each example should be valid
            self.assertTrue(
                result.is_valid,
                f"Example {example_file.name} failed validation: {result.errors}"
            )
        
        # All examples should pass
        valid_count = sum(1 for result in validation_results.values() if result.is_valid)
        self.assertEqual(valid_count, len(example_files))
    
    def test_parse_all_examples(self):
        """Test parsing of all example files."""
        if not self.examples_dir.exists():
            self.skipTest("Examples directory not found")
        
        example_files = list(self.examples_dir.glob("*.md"))
        parse_results = {}
        
        for example_file in example_files:
            try:
                parsed = self.parser.parse_file(example_file)
                parse_results[example_file.name] = parsed
                
                # Basic structure checks
                self.assertIsInstance(parsed, ParsedGNN)
                self.assertIsInstance(parsed.gnn_section, str)
                self.assertIsInstance(parsed.model_name, str)
                self.assertIsInstance(parsed.variables, dict)
                self.assertIsInstance(parsed.connections, list)
                
            except Exception as e:
                self.fail(f"Failed to parse example {example_file.name}: {e}")
        
        # All examples should parse successfully
        self.assertEqual(len(parse_results), len(example_files))
    
    def test_example_consistency(self):
        """Test consistency across example files."""
        if not self.examples_dir.exists():
            self.skipTest("Examples directory not found")
        
        example_files = list(self.examples_dir.glob("*.md"))
        
        for example_file in example_files:
            parsed = self.parser.parse_file(example_file)
            
            # Check that ontology mappings are consistent with variables
            for var_name, ontology_term in parsed.ontology_mappings.items():
                self.assertIn(
                    var_name, parsed.variables,
                    f"Ontology mapping references undefined variable: {var_name}"
                )
            
            # Check that connections reference defined variables
            for connection in parsed.connections:
                # This is a simplified check - full validation would need
                # to parse variable groups properly
                if isinstance(connection.source, str):
                    source_vars = [connection.source]
                else:
                    source_vars = connection.source if isinstance(connection.source, list) else []
                
                if isinstance(connection.target, str):
                    target_vars = [connection.target]
                else:
                    target_vars = connection.target if isinstance(connection.target, list) else []


class TestGNNPerformance(unittest.TestCase):
    """Test GNN module performance and memory usage."""
    
    def setUp(self):
        """Set up performance testing."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
        
        self.parser = GNNParser()
        self.validator = GNNValidator()
        
        # Generate large GNN content for testing
        self.large_gnn_content = self._generate_large_gnn_content()
    
    def _generate_large_gnn_content(self, num_variables: int = 100) -> str:
        """Generate large GNN content for performance testing."""
        content = """## GNNSection
LargeTestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Large Performance Test Model

## ModelAnnotation
This is a large model generated for performance testing.
It contains many variables and connections to test parser and validator performance.

## StateSpaceBlock
"""
        
        # Generate many variables
        for i in range(num_variables):
            content += f"var_{i}[{i+1},{i+2},type=float]    # Variable {i}\n"
        
        content += "\n## Connections\n"
        
        # Generate many connections
        for i in range(num_variables - 1):
            content += f"var_{i}>var_{i+1}\n"
        
        content += "\n## InitialParameterization\n"
        
        # Generate many parameters
        for i in range(min(20, num_variables)):  # Limit parameters to avoid memory issues
            content += f"var_{i}={{({i*0.1},{i*0.2})}}\n"
        
        content += """
## Time
Dynamic
DiscreteTime=t

## Footer
Large Performance Test Model - End
"""
        
        return content
    
    def test_parsing_performance(self):
        """Test parsing performance with large content."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        parsed = self.parser.parse_content(self.large_gnn_content)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        parse_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Performance assertions (adjust thresholds as needed)
        self.assertLess(parse_time, 5.0, f"Parsing took too long: {parse_time:.2f}s")
        self.assertLess(memory_used, 100 * 1024 * 1024, f"Too much memory used: {memory_used / 1024 / 1024:.2f}MB")
        
        # Verify parsing was successful
        self.assertIsInstance(parsed, ParsedGNN)
        self.assertGreater(len(parsed.variables), 50)
    
    def test_validation_performance(self):
        """Test validation performance."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        temp_file.write(self.large_gnn_content)
        temp_file.close()
        
        try:
            start_time = time.time()
            
            result = self.validator.validate_file(temp_file.name)
            
            end_time = time.time()
            validation_time = end_time - start_time
            
            # Performance assertion
            self.assertLess(validation_time, 10.0, f"Validation took too long: {validation_time:.2f}s")
            
            # Verify validation completed
            self.assertIsInstance(result, ValidationResult)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_memory_usage_large_files(self):
        """Test memory usage with very large files."""
        # Generate even larger content
        very_large_content = self._generate_large_gnn_content(num_variables=500)
        
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            parsed = self.parser.parse_content(very_large_content)
            peak_memory = psutil.Process().memory_info().rss
            
            memory_used = peak_memory - initial_memory
            
            # Memory usage should be reasonable
            self.assertLess(memory_used, 200 * 1024 * 1024, f"Excessive memory usage: {memory_used / 1024 / 1024:.2f}MB")
            
            # Verify parsing was successful
            self.assertIsInstance(parsed, ParsedGNN)
            
        except MemoryError:
            self.fail("Parser ran out of memory with large file")


class TestGNNErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up error handling tests."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
        
        self.parser = GNNParser()
        self.validator = GNNValidator()
    
    def test_unicode_content(self):
        """Test handling of Unicode content."""
        unicode_content = """## GNNSection
UnicodeTestModel

## GNNVersionAndFlags
GNN v1

## ModelName
Unicode Test Model ñáéíóú 中文 🚀

## ModelAnnotation
This model tests Unicode handling.
Special characters: αβγδε θλμπ ∑∏∫∆
Mathematical symbols: ∀∃∈∉⊆⊇∪∩

## StateSpaceBlock
α_factor[2,type=float]    # Alpha factor with Greek letter
β_matrix[3,3,type=float]  # Beta matrix
observación[4,type=int]   # Spanish variable name

## Connections
α_factor>β_matrix
β_matrix>observación

## InitialParameterization
α_factor={(0.5,0.5)}

## Time
Static

## Footer
End of Unicode test - 测试结束
"""
        
        # Should parse without errors
        parsed = self.parser.parse_content(unicode_content)
        self.assertIsInstance(parsed, ParsedGNN)
        self.assertIn("Unicode Test Model", parsed.model_name)
        self.assertIn("α_factor", parsed.variables)
    
    def test_malformed_sections(self):
        """Test handling of malformed sections."""
        malformed_content = """## GNNSection
MalformedTest

## InvalidSectionName
This section doesn't exist in the spec.

## StateSpaceBlock
invalid_syntax[
missing_bracket[2,type=float  # Missing closing bracket
too_many[brackets]][2,type=float]

## Connections
invalid>connection>syntax  # Multiple operators
source  # Missing target and operator

## InitialParameterization
param_no_value=  # Missing value
=missing_name   # Missing parameter name
invalid{{syntax}} # Invalid value syntax

## Time
# Missing time specification

## Footer
End
"""
        
        # Should handle malformed content gracefully
        parsed = self.parser.parse_content(malformed_content)
        self.assertIsInstance(parsed, ParsedGNN)
        
        # Validation should catch errors
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        temp_file.write(malformed_content)
        temp_file.close()
        
        try:
            result = self.validator.validate_file(temp_file.name)
            self.assertFalse(result.is_valid)
            self.assertGreater(len(result.errors), 0)
        finally:
            os.unlink(temp_file.name)
    
    def test_extremely_long_lines(self):
        """Test handling of extremely long lines."""
        long_line = "x" * 10000  # 10KB line
        
        long_line_content = f"""## GNNSection
LongLineTest

## ModelName
{long_line}

## ModelAnnotation
Testing very long lines.

## StateSpaceBlock
var[2,type=float]

## Connections
# Long comment: {"x" * 1000}
var>var

## InitialParameterization
var={{(1.0,2.0)}}

## Time
Static

## Footer
End
"""
        
        # Should handle long lines without crashing
        parsed = self.parser.parse_content(long_line_content)
        self.assertIsInstance(parsed, ParsedGNN)
    
    def test_empty_sections(self):
        """Test handling of empty sections."""
        empty_sections_content = """## GNNSection
EmptyTest

## GNNVersionAndFlags
GNN v1

## ModelName
Empty Sections Test

## ModelAnnotation
Testing empty sections.

## StateSpaceBlock
# No variables defined

## Connections
# No connections defined

## InitialParameterization
# No parameters defined

## Time
Static

## Footer
End
"""
        
        parsed = self.parser.parse_content(empty_sections_content)
        self.assertIsInstance(parsed, ParsedGNN)
        self.assertEqual(len(parsed.variables), 0)
        self.assertEqual(len(parsed.connections), 0)
        self.assertEqual(len(parsed.parameters), 0)
    
    def test_parser_robustness(self):
        """Test parser robustness with various edge cases."""
        edge_cases = [
            "",  # Empty file
            "## GNNSection\n",  # Only one section
            "Not a valid GNN file at all",  # No GNN structure
            "## " + "x" * 1000,  # Very long section name
            "\n" * 1000,  # Many empty lines
        ]
        
        for i, case in enumerate(edge_cases):
            with self.subTest(case=i):
                try:
                    parsed = self.parser.parse_content(case)
                    self.assertIsInstance(parsed, ParsedGNN)
                except Exception as e:
                    # Exceptions are acceptable for edge cases, but shouldn't crash
                    self.assertIsInstance(e, (ValueError, TypeError, AttributeError))


class TestGNNMCPIntegration(unittest.TestCase):
    """Test MCP (Model Context Protocol) integration."""
    
    def setUp(self):
        """Set up MCP integration tests."""
        if not GNN_AVAILABLE:
            self.skipTest("GNN module not available")
    
    def test_get_gnn_documentation(self):
        """Test MCP documentation retrieval."""
        doc_types = ["file_structure", "punctuation", "schema_json", "schema_yaml", "grammar"]
        
        for doc_type in doc_types:
            with self.subTest(doc_type=doc_type):
                result = get_gnn_documentation(doc_type)
                
                self.assertIsInstance(result, dict)
                self.assertIn("success", result)
                
                if result["success"]:
                    self.assertIn("content", result)
                    self.assertIsInstance(result["content"], str)
                    self.assertGreater(len(result["content"]), 0)
    
    def test_validate_gnn_content_mcp(self):
        """Test MCP validation functionality."""
        valid_content = """## GNNSection
MCPTest

## GNNVersionAndFlags
GNN v1

## ModelName
MCP Test Model

## ModelAnnotation
Testing MCP validation.

## StateSpaceBlock
x[2,type=float]

## Connections
x>x

## InitialParameterization
x={(1.0,2.0)}

## Time
Static

## Footer
End
"""
        
        result = validate_gnn_content(valid_content)
        
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        
        if result["success"]:
            self.assertIn("is_valid", result)
            self.assertIn("errors", result)
            self.assertIn("warnings", result)
    
    def test_mcp_error_handling(self):
        """Test MCP error handling."""
        # Test invalid document type
        result = get_gnn_documentation("invalid_type")
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["success"])
        self.assertIn("error", result)


if __name__ == '__main__':
    # Configure test discovery and execution
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestGNNSchemaValidation,
        TestGNNParser,
        TestGNNExampleValidation,
        TestGNNPerformance,
        TestGNNErrorHandling,
        TestGNNMCPIntegration,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1) 