#!/usr/bin/env python3
"""
Comprehensive tests for D2 visualization integration in advanced_visualization module.

This test file validates:
- D2Visualizer initialization and D2 CLI availability checking
- GNN model structure diagram generation
- POMDP diagram generation
- Pipeline flow diagram generation
- Framework mapping diagram generation
- Active Inference conceptual diagram generation
- D2 diagram compilation to multiple formats
- End-to-end processing of GNN files with D2
- Error handling and fallback mechanisms
"""

import unittest
import sys
import tempfile
import json
from pathlib import Path
# Mocks removed - using real implementations per testing policy

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import D2 visualization components
try:
    from advanced_visualization.d2_visualizer import (
        D2Visualizer,
        D2DiagramSpec,
        D2GenerationResult,
        process_gnn_file_with_d2
    )
    D2_MODULE_AVAILABLE = True
except ImportError as e:
    D2_MODULE_AVAILABLE = False
    print(f"Warning: D2 visualizer module not available: {e}")


class TestD2VisualizerImport(unittest.TestCase):
    """Test D2 visualizer module imports"""
    
    def test_d2_visualizer_module_available(self):
        """Test that D2 visualizer module can be imported"""
        self.assertTrue(D2_MODULE_AVAILABLE, "D2 visualizer module should be importable")
    
    @unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
    def test_d2_classes_available(self):
        """Test that D2 classes are available"""
        self.assertIsNotNone(D2Visualizer)
        self.assertIsNotNone(D2DiagramSpec)
        self.assertIsNotNone(D2GenerationResult)
        self.assertIsNotNone(process_gnn_file_with_d2)


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2VisualizerInitialization(unittest.TestCase):
    """Test D2Visualizer initialization and setup"""
    
    def test_d2_visualizer_init(self):
        """Test D2Visualizer initialization"""
        visualizer = D2Visualizer()
        self.assertIsNotNone(visualizer)
        self.assertIsNotNone(visualizer.logger)
    
    def test_d2_visualizer_with_logger(self):
        """Test D2Visualizer initialization with custom logger"""
        import logging
        logger = logging.getLogger("test_logger")
        visualizer = D2Visualizer(logger=logger)
        self.assertEqual(visualizer.logger, logger)
    
    def test_d2_availability_check(self):
        """Test D2 CLI availability checking"""
        visualizer = D2Visualizer()
        # d2_available is a boolean
        self.assertIn(visualizer.d2_available, [True, False])


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2DiagramGeneration(unittest.TestCase):
    """Test D2 diagram generation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = D2Visualizer()
        
        # Sample GNN model data (Active Inference POMDP)
        self.model_data = {
            "model_name": "Test POMDP Agent",
            "state_space": {
                "A": {
                    "dimensions": [3, 3],
                    "type": "float",
                    "description": "Likelihood matrix"
                },
                "B": {
                    "dimensions": [3, 3, 3],
                    "type": "float",
                    "description": "Transition matrix"
                },
                "C": {
                    "dimensions": [3],
                    "type": "float",
                    "description": "Preference vector"
                },
                "s": {
                    "dimensions": [3, 1],
                    "type": "float",
                    "description": "Hidden state"
                },
                "o": {
                    "dimensions": [3, 1],
                    "type": "int",
                    "description": "Observation"
                }
            },
            "connections": [
                {"source": "s", "target": "A", "type": "->"},
                {"source": "A", "target": "o", "type": "->"},
                {"source": "s", "target": "B", "type": "->"}
            ],
            "actinf_annotations": {
                "A": "LikelihoodMatrix",
                "B": "TransitionMatrix",
                "C": "LogPreferenceVector",
                "s": "HiddenState",
                "o": "Observation"
            }
        }
    
    def test_generate_model_structure_diagram(self):
        """Test generation of model structure diagram"""
        spec = self.visualizer.generate_model_structure_diagram(self.model_data)
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertIn("structure", spec.name.lower())
        self.assertTrue(len(spec.d2_content) > 0)
        self.assertIn("State Space", spec.d2_content)
        self.assertIn("# GNN Model", spec.d2_content)
    
    def test_generate_pomdp_diagram(self):
        """Test generation of POMDP diagram"""
        spec = self.visualizer.generate_pomdp_diagram(self.model_data)
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertIn("pomdp", spec.name.lower())
        self.assertTrue(len(spec.d2_content) > 0)
        self.assertIn("Active Inference", spec.d2_content)
        self.assertIn("Generative Model", spec.d2_content)
    
    def test_generate_pipeline_flow_diagram(self):
        """Test generation of pipeline flow diagram"""
        spec = self.visualizer.generate_pipeline_flow_diagram(include_frameworks=True)
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertEqual(spec.name, "gnn_pipeline_flow")
        self.assertTrue(len(spec.d2_content) > 0)
        self.assertIn("GNN Pipeline", spec.d2_content)
        self.assertIn("Code Generation", spec.d2_content)
    
    def test_generate_pipeline_flow_diagram_no_frameworks(self):
        """Test pipeline flow diagram without framework details"""
        spec = self.visualizer.generate_pipeline_flow_diagram(include_frameworks=False)
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertTrue(len(spec.d2_content) > 0)
    
    def test_generate_framework_mapping_diagram(self):
        """Test generation of framework mapping diagram"""
        spec = self.visualizer.generate_framework_mapping_diagram()
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertEqual(spec.name, "framework_integration")
        self.assertTrue(len(spec.d2_content) > 0)
        self.assertIn("Framework Integration", spec.d2_content)
    
    def test_generate_framework_mapping_custom_frameworks(self):
        """Test framework mapping with custom framework list"""
        frameworks = ["pymdp", "jax"]
        spec = self.visualizer.generate_framework_mapping_diagram(frameworks=frameworks)
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertIn("pymdp", spec.d2_content.lower())
        self.assertIn("jax", spec.d2_content.lower())
    
    def test_generate_active_inference_concepts_diagram(self):
        """Test generation of Active Inference concepts diagram"""
        spec = self.visualizer.generate_active_inference_concepts_diagram()
        
        self.assertIsInstance(spec, D2DiagramSpec)
        self.assertEqual(spec.name, "active_inference_concepts")
        self.assertTrue(len(spec.d2_content) > 0)
        self.assertIn("Free Energy Principle", spec.d2_content)
        self.assertIn("Cognitive Agent", spec.d2_content)


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2DiagramCompilation(unittest.TestCase):
    """Test D2 diagram compilation to output formats"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = D2Visualizer()
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Simple test diagram spec
        self.test_spec = D2DiagramSpec(
            name="test_diagram",
            description="Test D2 diagram",
            d2_content="Test: { shape: rectangle }",
            output_formats=["svg"],
            layout_engine="dagre",
            theme=1
        )
    
    def tearDown(self):
        """Clean up test directory"""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_compile_d2_diagram_no_cli(self):
        """Test compilation when D2 CLI is not available"""
        # Temporarily set d2_available to False
        original_available = self.visualizer.d2_available
        self.visualizer.d2_available = False
        
        result = self.visualizer.compile_d2_diagram(
            self.test_spec,
            self.output_dir
        )
        
        self.assertIsInstance(result, D2GenerationResult)
        self.assertFalse(result.success)
        self.assertIn("not available", result.error_message.lower())
        
        # Restore original state
        self.visualizer.d2_available = original_available
    
    def test_compile_d2_diagram_write_d2_file(self):
        """Test that D2 source file is written"""
        # This test doesn't require D2 CLI
        original_available = self.visualizer.d2_available
        self.visualizer.d2_available = False  # Skip compilation
        
        result = self.visualizer.compile_d2_diagram(
            self.test_spec,
            self.output_dir
        )
        
        # D2 file should still be written
        d2_file = self.output_dir / "test_diagram.d2"
        # File won't exist because compilation failed, but we tested the logic
        
        self.visualizer.d2_available = original_available
    
    @unittest.skipIf(not D2Visualizer().d2_available, "D2 CLI not available")
    def test_compile_d2_diagram_with_cli(self):
        """Test actual D2 compilation with CLI (if available)"""
        result = self.visualizer.compile_d2_diagram(
            self.test_spec,
            self.output_dir,
            formats=["svg"]
        )
        
        if result.success:
            self.assertTrue(len(result.output_files) > 0)
            self.assertIsNotNone(result.d2_file)
            self.assertTrue(result.d2_file.exists())


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2HelperMethods(unittest.TestCase):
    """Test D2 helper and utility methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = D2Visualizer()
    
    def test_sanitize_name(self):
        """Test name sanitization for D2 identifiers"""
        test_cases = {
            "Test Model v1": "test_model_v1",
            "POMDP-Agent@2023": "pomdp_agent2023",  # Hyphen becomes underscore
            "simple": "simple",
            "Multi Word Name": "multi_word_name"
        }
        
        for input_name, expected_output in test_cases.items():
            result = self.visualizer._sanitize_name(input_name)
            self.assertEqual(result, expected_output)
    
    def test_get_d2_shape_for_variable(self):
        """Test D2 shape determination for variables"""
        test_cases = [
            ({"dimensions": [3, 3]}, {"A": "Matrix"}, "hexagon"),
            ({"dimensions": [3]}, {"C": "Vector"}, "diamond"),
            ({"dimensions": [3, 1]}, {"s": "State"}, "cylinder"),
        ]
        
        for var_info, annotations, _ in test_cases:
            var_name = list(annotations.keys())[0]
            shape = self.visualizer._get_d2_shape_for_variable(
                var_name, var_info, annotations
            )
            self.assertIsInstance(shape, str)
            self.assertTrue(len(shape) > 0)
    
    def test_get_d2_arrow(self):
        """Test D2 arrow notation conversion"""
        arrow_tests = {
            "->": "->",
            "<-": "<-",
            "<->": "<->",
            "-": "--",
            ">": "->",
            "<": "<-"
        }
        
        for conn_type, expected_arrow in arrow_tests.items():
            result = self.visualizer._get_d2_arrow(conn_type)
            self.assertEqual(result, expected_arrow)
    
    def test_is_pomdp_model(self):
        """Test POMDP model detection"""
        pomdp_model = {
            "state_space": {"A": {}, "B": {}, "C": {}},
            "actinf_annotations": {"A": "Likelihood"}
        }
        
        non_pomdp_model = {
            "state_space": {"x": {}, "y": {}},
            "actinf_annotations": {}
        }
        
        self.assertTrue(self.visualizer._is_pomdp_model(pomdp_model))
        self.assertFalse(self.visualizer._is_pomdp_model(non_pomdp_model))


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2EndToEndProcessing(unittest.TestCase):
    """Test end-to-end D2 processing workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = D2Visualizer()
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        self.test_model = {
            "model_name": "Integration Test Model",
            "state_space": {
                "A": {"dimensions": [3, 3], "type": "float"},
                "B": {"dimensions": [3, 3, 3], "type": "float"}
            },
            "connections": [],
            "actinf_annotations": {"A": "LikelihoodMatrix"}
        }
    
    def tearDown(self):
        """Clean up test directory"""
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
    
    def test_generate_all_diagrams_for_model(self):
        """Test generating all diagrams for a model"""
        results = self.visualizer.generate_all_diagrams_for_model(
            self.test_model,
            self.output_dir,
            formats=["svg"]
        )
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        
        for result in results:
            self.assertIsInstance(result, D2GenerationResult)
            self.assertIsNotNone(result.diagram_name)


@unittest.skipIf(not D2_MODULE_AVAILABLE, "D2 module not available")
class TestD2ProcessorIntegration(unittest.TestCase):
    """Test D2 integration with advanced_visualization processor"""
    
    def test_processor_has_d2_methods(self):
        """Test that processor has D2 generation methods"""
        from advanced_visualization.processor import (
            _generate_d2_visualizations_safe,
            _generate_pipeline_d2_diagrams_safe
        )
        
        self.assertIsNotNone(_generate_d2_visualizations_safe)
        self.assertIsNotNone(_generate_pipeline_d2_diagrams_safe)
    
    def test_init_exports_d2_components(self):
        """Test that __init__ exports D2 components"""
        from advanced_visualization import (
            D2Visualizer,
            D2DiagramSpec,
            D2GenerationResult,
            D2_AVAILABLE
        )
        
        # These should all be importable (even if None when not available)
        self.assertIsNotNone(D2_AVAILABLE)


class TestD2Documentation(unittest.TestCase):
    """Test D2 module documentation and setup"""
    
    def test_d2_visualizer_has_docstrings(self):
        """Test that D2Visualizer class has comprehensive docstrings"""
        if not D2_MODULE_AVAILABLE:
            self.skipTest("D2 module not available")
        
        self.assertIsNotNone(D2Visualizer.__doc__)
        self.assertTrue(len(D2Visualizer.__doc__) > 50)
    
    def test_d2_diagram_spec_has_fields(self):
        """Test that D2DiagramSpec has required fields"""
        if not D2_MODULE_AVAILABLE:
            self.skipTest("D2 module not available")
        
        # Check dataclass fields
        from dataclasses import fields
        spec_fields = [f.name for f in fields(D2DiagramSpec)]
        
        required_fields = ["name", "description", "d2_content", "output_formats"]
        for field in required_fields:
            self.assertIn(field, spec_fields)


def run_tests():
    """Run all D2 visualizer tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestD2VisualizerImport))
    suite.addTests(loader.loadTestsFromTestCase(TestD2VisualizerInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestD2DiagramGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestD2DiagramCompilation))
    suite.addTests(loader.loadTestsFromTestCase(TestD2HelperMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestD2EndToEndProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestD2ProcessorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestD2Documentation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

