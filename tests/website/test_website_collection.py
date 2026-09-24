"""Import-stability pins for the website artifact-collection seam.

Pins: ``collect_website_data`` and its private collectors live in
``gnn.website.collection``, the package re-export from ``gnn.website`` is the
same object, the step catalogue import reaches the collection module, and
``WebsiteGenerator`` reaches the collection module through its delegating
``_collect_all_data`` path at generation time.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCollectionModuleHome:
    """The collection seam is importable at its new home and re-exported."""

    def test_collect_website_data_lives_in_collection(self) -> None:
        from gnn.website.collection import collect_website_data

        assert callable(collect_website_data)

    def test_package_reexport_is_same_object(self) -> None:
        import gnn.website as website
        import gnn.website.collection as collection

        assert website.collect_website_data is collection.collect_website_data

    def test_step_catalogue_import_reaches_collection(self) -> None:
        from gnn.website import PIPELINE_STEPS
        from gnn.website.collection import PIPELINE_STEPS as collection_steps

        assert collection_steps is PIPELINE_STEPS


class TestCollectionBehavior:
    """Moved collectors keep their contracts at the new home."""

    def test_summary_records_map_to_statuses(self) -> None:
        from gnn.website.collection import _step_statuses_from_summary

        statuses = _step_statuses_from_summary(
            {"steps": [{"script_name": "03_gnn.py", "status": "SUCCESS"}]}
        )
        assert statuses == {3: "ok"}

    def test_unusable_status_maps_to_error(self) -> None:
        from gnn.website.collection import _normalize_website_step_status

        assert _normalize_website_step_status(None) is None
        assert _normalize_website_step_status("PARTIAL SUCCESS") == "error"

    def test_generator_collects_through_collection_module(self, tmp_path: Any) -> None:
        from gnn.website import WebsiteGenerator

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        generator = WebsiteGenerator()
        data = generator._collect_all_data(
            tmp_path,
            input_dir,
            tmp_path / "site",
            tmp_path / "site" / "assets",
            {},
        )
        assert len(data["step_statuses"]) == 25
        assert data["processed_files"] == 0
        assert data["gui_navigation"] is False
