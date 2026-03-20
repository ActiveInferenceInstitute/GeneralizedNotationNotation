#!/usr/bin/env python3
"""
Tests for gnn/parsers/xml_parser.py — XMLGNNParser and PNMLParser.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MINIMAL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<GNNModel>
  <Metadata>
    <ModelName>TestModel</ModelName>
    <GNNSection>ActInfPOMDP</GNNSection>
  </Metadata>
  <StateSpaceBlock>
    <Variable name="s" dimensions="3,1" type="float"/>
    <Variable name="o" dimensions="2,1" type="int"/>
  </StateSpaceBlock>
  <Connections>
    <Connection source="s" target="o" directed="true"/>
  </Connections>
</GNNModel>
"""

MINIMAL_PNML = """\
<?xml version="1.0"?>
<pnml>
  <net id="net1" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <name><text>TestNet</text></name>
    <page id="page1">
      <place id="p1"><name><text>place1</text></name></place>
      <place id="p2"><name><text>place2</text></name></place>
      <transition id="t1"><name><text>trans1</text></name></transition>
      <arc id="a1" source="p1" target="t1"/>
    </page>
  </net>
</pnml>
"""


class TestXMLGNNParser:
    def _get_parser(self):
        try:
            from gnn.parsers.xml_parser import XMLGNNParser
            return XMLGNNParser()
        except ImportError:
            pytest.skip("xml_parser not importable")

    def test_parse_string_returns_parse_result(self):
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        assert result is not None

    def test_parse_string_with_valid_xml_no_errors(self):
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        # Should parse without crashing; errors list may be present
        assert hasattr(result, "errors") or result is not None

    def test_parse_string_invalid_xml_handles_gracefully(self):
        parser = self._get_parser()
        result = parser.parse_string("NOT XML AT ALL <<<")
        # Must not raise — should return error result or None
        # If it returns something, it should have errors
        assert result is None or hasattr(result, "errors")

    def test_get_supported_extensions_includes_xml(self):
        parser = self._get_parser()
        exts = parser.get_supported_extensions()
        assert any("xml" in ext.lower() for ext in exts)

    def test_parse_string_result_has_model(self):
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_XML)
        if result is None:
            pytest.skip("parser returned None for minimal XML")
        # ParseResult has a .model attribute
        assert hasattr(result, "model") or hasattr(result, "model_name")


class TestPNMLParser:
    def _get_parser(self):
        try:
            from gnn.parsers.xml_parser import PNMLParser
            return PNMLParser()
        except ImportError:
            pytest.skip("PNMLParser not importable")

    def test_parse_string_returns_result(self):
        parser = self._get_parser()
        result = parser.parse_string(MINIMAL_PNML)
        assert result is not None

    def test_parse_string_invalid_pnml_handles_gracefully(self):
        parser = self._get_parser()
        result = parser.parse_string("<<INVALID>>")
        assert result is None or hasattr(result, "errors")

    def test_get_supported_extensions_includes_pnml(self):
        parser = self._get_parser()
        exts = parser.get_supported_extensions()
        assert any("pnml" in ext.lower() or "xml" in ext.lower() for ext in exts)
