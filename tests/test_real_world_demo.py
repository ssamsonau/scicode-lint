"""Tests for real_world_demo module."""

import json
import tempfile
from pathlib import Path


class TestConfig:
    """Tests for config module."""

    def test_scientific_domains_defined(self) -> None:
        """Verify scientific domains are defined."""
        from real_world_demo.config import SCIENTIFIC_DOMAINS

        assert "biology" in SCIENTIFIC_DOMAINS
        assert "chemistry" in SCIENTIFIC_DOMAINS
        assert "medical" in SCIENTIFIC_DOMAINS
        assert "physics" in SCIENTIFIC_DOMAINS

    def test_ml_imports_defined(self) -> None:
        """Verify ML imports list is defined."""
        from real_world_demo.config import ML_IMPORTS

        assert "sklearn" in ML_IMPORTS
        assert "torch" in ML_IMPORTS
        assert "tensorflow" in ML_IMPORTS

    def test_directories_created(self) -> None:
        """Verify data directories exist."""
        from real_world_demo.config import COLLECTED_DIR, DATA_DIR

        assert DATA_DIR.exists()
        assert COLLECTED_DIR.exists()


class TestFilterPapers:
    """Tests for filter_papers module."""

    def test_matches_domain_biology(self) -> None:
        """Test domain matching for biology tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Protein Structure Prediction", "Gene Expression"]
        assert matches_domain(tasks) == "biology"

    def test_matches_domain_medical(self) -> None:
        """Test domain matching for medical tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Medical Image Segmentation", "Disease Classification"]
        assert matches_domain(tasks) == "medical"

    def test_matches_domain_none(self) -> None:
        """Test no match for non-scientific tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Image Classification", "Object Detection"]
        assert matches_domain(tasks) is None

    def test_matches_domain_empty(self) -> None:
        """Test empty tasks list."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        assert matches_domain([]) is None
        assert matches_domain(None) is None

    def test_matches_domain_specific_domains(self) -> None:
        """Test filtering to specific domains."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Protein Folding", "Drug Discovery"]
        # Should match biology when filtering for biology only
        assert matches_domain(tasks, domains=["biology"]) == "biology"
        # Should not match when filtering for physics only
        assert matches_domain(tasks, domains=["physics"]) is None

    def test_should_exclude_benchmark(self) -> None:
        """Test exclusion of benchmark tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import should_exclude

        tasks = ["Image Classification Benchmark", "Model Evaluation"]
        assert should_exclude(tasks) is True

    def test_should_exclude_false(self) -> None:
        """Test non-exclusion of scientific tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import should_exclude

        tasks = ["Protein Structure Prediction"]
        assert should_exclude(tasks) is False

    def test_normalize_github_url(self) -> None:
        """Test GitHub URL normalization."""
        from real_world_demo.sources.papers_with_code.filter_papers import normalize_github_url

        # Remove trailing slash
        assert (
            normalize_github_url("https://github.com/owner/repo/")
            == "https://github.com/owner/repo"
        )

        # Remove .git suffix
        assert (
            normalize_github_url("https://github.com/owner/repo.git")
            == "https://github.com/owner/repo"
        )

        # Convert git:// to https://
        assert (
            normalize_github_url("git://github.com/owner/repo") == "https://github.com/owner/repo"
        )

        # Convert SSH to HTTPS
        assert normalize_github_url("git@github.com:owner/repo") == "https://github.com/owner/repo"

    def test_filter_papers_basic(self) -> None:
        """Test basic paper filtering."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        papers = [
            {
                "paper_url": "https://paperswithcode.com/paper/1",
                "tasks": ["Protein Folding"],
                "title": "Paper 1",
            },
            {
                "paper_url": "https://paperswithcode.com/paper/2",
                "tasks": ["Image Classification"],
                "title": "Paper 2",
            },
        ]
        links = [
            {
                "paper_url": "https://paperswithcode.com/paper/1",
                "repo_url": "https://github.com/owner/repo1",
            },
        ]

        filtered = filter_papers(papers, links)

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Paper 1"
        # repo_urls are now embedded in paper records
        assert "repo_urls" in filtered[0]
        assert "https://github.com/owner/repo1" in filtered[0]["repo_urls"]

    def test_filter_papers_balanced_sampling(self) -> None:
        """Test balanced sampling across domains."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        # Create papers across multiple domains
        papers = []
        links = []
        # 10 biology papers
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "tasks": ["Protein Folding"],
                    "title": f"Bio Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "repo_url": f"https://github.com/owner/bio{i}",
                }
            )
        # 10 medical papers
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "tasks": ["Medical Imaging"],
                    "title": f"Medical Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "repo_url": f"https://github.com/owner/med{i}",
                }
            )

        # Request 10 papers with balanced sampling
        filtered = filter_papers(papers, links, limit=10, balanced=True)

        # Count domains
        domain_counts: dict[str, int] = {}
        for paper in filtered:
            d = paper.get("matched_domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

        # Should have roughly equal distribution (5 each)
        assert len(filtered) == 10
        assert domain_counts.get("biology", 0) == 5
        assert domain_counts.get("medical", 0) == 5

    def test_filter_papers_unbalanced(self) -> None:
        """Test unbalanced sampling (first-come-first-served)."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        # Create papers - biology first, then medical
        papers = []
        links = []
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "tasks": ["Protein Folding"],
                    "title": f"Bio Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "repo_url": f"https://github.com/owner/bio{i}",
                }
            )
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "tasks": ["Medical Imaging"],
                    "title": f"Medical Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "repo_url": f"https://github.com/owner/med{i}",
                }
            )

        # Request 10 papers WITHOUT balanced sampling
        filtered = filter_papers(papers, links, limit=10, balanced=False)

        assert len(filtered) == 10


class TestCloneRepos:
    """Tests for clone_repos module."""

    def test_repo_url_to_path(self) -> None:
        """Test converting repo URL to local path."""
        from real_world_demo.sources.papers_with_code.clone_repos import repo_url_to_path

        base_dir = Path("/tmp/repos")
        url = "https://github.com/owner/repo"
        path = repo_url_to_path(url, base_dir)

        assert path == base_dir / "owner__repo"

    def test_clone_result_to_dict(self) -> None:
        """Test CloneResult serialization."""
        from real_world_demo.sources.papers_with_code.clone_repos import CloneResult

        result = CloneResult(
            repo_url="https://github.com/owner/repo",
            success=True,
            repo_path=Path("/tmp/repos/owner__repo"),
        )
        d = result.to_dict()

        assert d["repo_url"] == "https://github.com/owner/repo"
        assert d["success"] is True
        assert d["repo_path"] == "/tmp/repos/owner__repo"
        assert d["error"] is None

    def test_clone_result_to_dict_with_error(self) -> None:
        """Test CloneResult serialization with error."""
        from real_world_demo.sources.papers_with_code.clone_repos import CloneResult

        result = CloneResult(
            repo_url="https://github.com/owner/repo",
            success=False,
            error="not_found",
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "not_found"
        assert d["repo_path"] is None


class TestFilterFiles:
    """Tests for filter_files module."""

    def test_extract_imports(self) -> None:
        """Test import extraction from Python code."""
        from real_world_demo.sources.papers_with_code.filter_files import extract_imports

        code = """
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch import nn
"""
        imports = extract_imports(code)

        assert "numpy" in imports
        assert "pandas" in imports
        assert "sklearn" in imports
        assert "torch" in imports

    def test_has_ml_imports_true(self) -> None:
        """Test ML import detection - positive case."""
        from real_world_demo.sources.papers_with_code.filter_files import has_ml_imports

        imports = {"numpy", "sklearn", "pandas"}
        has_ml, matched = has_ml_imports(imports)

        assert has_ml is True
        assert "sklearn" in matched

    def test_has_ml_imports_false(self) -> None:
        """Test ML import detection - negative case."""
        from real_world_demo.sources.papers_with_code.filter_files import has_ml_imports

        imports = {"os", "sys", "json"}
        has_ml, matched = has_ml_imports(imports)

        assert has_ml is False
        assert len(matched) == 0

    def test_has_scientific_imports(self) -> None:
        """Test scientific import detection."""
        from real_world_demo.sources.papers_with_code.filter_files import has_scientific_imports

        imports = {"numpy", "scipy", "matplotlib"}
        has_sci, matched = has_scientific_imports(imports)

        assert has_sci is True
        assert "numpy" in matched
        assert "scipy" in matched

    def test_matches_exclude_pattern_setup(self) -> None:
        """Test file exclusion for setup.py."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "setup.py"

        assert matches_exclude_pattern(file_path, repo_root) is True

    def test_matches_exclude_pattern_test_file(self) -> None:
        """Test file exclusion for test files."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "test_model.py"

        assert matches_exclude_pattern(file_path, repo_root) is True

    def test_matches_exclude_pattern_normal_file(self) -> None:
        """Test non-exclusion for normal files."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "model.py"

        assert matches_exclude_pattern(file_path, repo_root) is False


class TestGenerateManifest:
    """Tests for generate_manifest module."""

    def test_generate_unique_path(self) -> None:
        """Test unique path generation within repo directories."""
        from real_world_demo.sources.papers_with_code.generate_manifest import generate_unique_path

        seen_per_repo: dict[str, set[str]] = {}
        file_path = Path("/tmp/repo/model.py")
        repo_name = "owner__repo"

        repo_dir, name1 = generate_unique_path(file_path, repo_name, seen_per_repo)
        assert repo_dir == "owner__repo"
        assert name1 == "model.py"
        assert name1 in seen_per_repo[repo_name]

        # Second call with same file should get a unique name
        repo_dir, name2 = generate_unique_path(file_path, repo_name, seen_per_repo)
        assert repo_dir == "owner__repo"
        assert name2 == "model_1.py"
        assert name2 in seen_per_repo[repo_name]

    def test_load_repos_metadata(self) -> None:
        """Test loading repo metadata from papers file with embedded repo_urls."""
        from real_world_demo.sources.papers_with_code.generate_manifest import load_repos_metadata

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Papers format with embedded repo_urls
            papers = [
                {
                    "paper_url": "https://paperswithcode.com/paper1",
                    "title": "Paper 1",
                    "matched_domain": "biology",
                    "repo_urls": ["https://github.com/owner/repo1"],
                },
                {
                    "paper_url": "https://paperswithcode.com/paper2",
                    "title": "Paper 2",
                    "matched_domain": "medical",
                    "repo_urls": ["https://github.com/owner/repo2"],
                },
            ]
            json.dump(papers, f)
            f.flush()

            metadata = load_repos_metadata(Path(f.name))

            assert "owner__repo1" in metadata
            assert metadata["owner__repo1"]["domain"] == "biology"
            assert "owner__repo2" in metadata


class TestRunAnalysis:
    """Tests for run_analysis module."""

    def test_aggregate_findings_empty(self) -> None:
        """Test aggregation with no findings."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {"file_path": "file1.py", "success": True, "findings": [], "domain": "biology"},
            {"file_path": "file2.py", "success": True, "findings": [], "domain": "medical"},
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["analyzed_successfully"] == 2
        assert stats["files_with_findings"] == 0
        assert stats["total_findings"] == 0

    def test_aggregate_findings_with_findings(self) -> None:
        """Test aggregation with findings."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {
                "file_path": "file1.py",
                "success": True,
                "findings": [
                    {"pattern_id": "dl-001", "category": "data-leakage"},
                    {"pattern_id": "dl-002", "category": "data-leakage"},
                ],
                "domain": "biology",
                "repo_name": "repo1",
                "paper_url": "https://paper1",
            },
            {
                "file_path": "file2.py",
                "success": True,
                "findings": [{"pattern_id": "rs-001", "category": "reproducibility"}],
                "domain": "medical",
                "repo_name": "repo2",
                "paper_url": "https://paper2",
            },
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["files_with_findings"] == 2
        assert stats["total_findings"] == 3
        assert stats["findings_by_pattern"]["dl-001"] == 1
        assert stats["findings_by_pattern"]["dl-002"] == 1
        assert stats["findings_by_category"]["data-leakage"] == 2
        assert stats["findings_by_category"]["reproducibility"] == 1
        assert stats["repos_with_findings"] == 2
        assert stats["papers_with_findings"] == 2

    def test_aggregate_findings_failed_analysis(self) -> None:
        """Test aggregation handles failed analyses."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {"file_path": "file1.py", "success": False, "error": "timeout", "findings": []},
            {"file_path": "file2.py", "success": True, "findings": [], "domain": "biology"},
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["analyzed_successfully"] == 1

    def test_load_manifest(self) -> None:
        """Test loading manifest CSV."""
        import csv

        from real_world_demo.run_analysis import load_manifest

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "domain", "repo_name"])
            writer.writeheader()
            writer.writerow(
                {
                    "file_path": "files/test.py",
                    "domain": "biology",
                    "repo_name": "repo1",
                }
            )
            f.flush()

            manifest = load_manifest(Path(f.name))

            assert len(manifest) == 1
            assert manifest[0]["file_path"] == "files/test.py"
            assert manifest[0]["domain"] == "biology"


class TestPrefilterFiles:
    """Tests for prefilter_files module."""

    def test_prefilter_prompt_exists(self) -> None:
        """Test that prefilter prompt is defined."""
        from real_world_demo.sources.papers_with_code.prefilter_files import PREFILTER_PROMPT

        assert "ML" in PREFILTER_PROMPT or "scientific" in PREFILTER_PROMPT.lower()
        assert "YES" in PREFILTER_PROMPT
        assert "NO" in PREFILTER_PROMPT

    def test_prefilter_prompt_covers_categories(self) -> None:
        """Test that prefilter prompt covers all scicode-lint categories."""
        from real_world_demo.sources.papers_with_code.prefilter_files import PREFILTER_PROMPT

        prompt_lower = PREFILTER_PROMPT.lower()
        # Should cover main detection areas
        assert "training" in prompt_lower or "train" in prompt_lower
        assert "inference" in prompt_lower or "prediction" in prompt_lower
        assert "numerical" in prompt_lower or "computation" in prompt_lower
        assert "random" in prompt_lower  # reproducibility

    def test_load_qualifying_files_not_found(self) -> None:
        """Test error when qualifying files not found."""
        from pathlib import Path

        from real_world_demo.sources.papers_with_code.prefilter_files import load_qualifying_files

        try:
            load_qualifying_files(Path("/nonexistent/path.json"))
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_save_results(self) -> None:
        """Test saving prefilter results."""
        from real_world_demo.sources.papers_with_code.prefilter_files import save_results

        pipeline_files = [{"file_path": "a.py", "is_pipeline": True}]
        filtered_out = [{"file_path": "b.py", "is_pipeline": False}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_results(pipeline_files, filtered_out, output_dir)

            assert (output_dir / "pipeline_files.json").exists()
            assert (output_dir / "prefilter_excluded.json").exists()

            with open(output_dir / "pipeline_files.json") as f:
                saved = json.load(f)
            assert len(saved) == 1
            assert saved[0]["file_path"] == "a.py"


class TestRunPipeline:
    """Tests for run_pipeline module."""

    def test_check_prerequisites_filter(self) -> None:
        """Test prerequisites check for filter stage."""
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # Filter stage has no prerequisites
        assert check_prerequisites("filter") is True

    def test_check_prerequisites_clone_missing(self) -> None:
        """Test prerequisites check for clone stage with missing file."""
        from real_world_demo.config import DATA_DIR
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # Clone stage requires ai_science_papers.json (output from abstract_filter stage)
        ai_science_file = DATA_DIR / "ai_science_papers.json"
        if ai_science_file.exists():
            # If file exists, the check will pass
            assert check_prerequisites("clone") is True
        else:
            # If file doesn't exist, check will fail
            assert check_prerequisites("clone") is False

    def test_check_prerequisites_abstract_filter(self) -> None:
        """Test prerequisites check for abstract_filter stage."""
        from real_world_demo.config import DATA_DIR
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # abstract_filter stage requires filtered_papers.json
        filtered_file = DATA_DIR / "filtered_papers.json"
        if filtered_file.exists():
            assert check_prerequisites("abstract_filter") is True
        else:
            assert check_prerequisites("abstract_filter") is False


class TestDatabase:
    """Tests for database module."""

    def test_init_db_creates_tables(self) -> None:
        """Test database initialization creates all tables."""
        from real_world_demo.database import init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Check tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

            assert "papers" in tables
            assert "repos" in tables
            assert "files" in tables
            assert "analysis_runs" in tables
            assert "file_analyses" in tables
            assert "findings" in tables

            conn.close()

    def test_insert_paper(self) -> None:
        """Test inserting a paper record."""
        from real_world_demo.database import init_db, insert_paper

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            paper_data = {
                "paper_url": "https://paperswithcode.com/paper/test",
                "paper_title": "Test Paper",
                "arxiv_id": "2301.00001",
                "tasks": ["Protein Folding"],
                "domain": "biology",
            }
            paper_id = insert_paper(conn, paper_data)

            assert paper_id > 0

            # Inserting same paper should return existing ID
            paper_id2 = insert_paper(conn, paper_data)
            assert paper_id2 == paper_id

            conn.close()

    def test_get_or_create_repo(self) -> None:
        """Test getting or creating a repo record."""
        from real_world_demo.database import get_or_create_repo, init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            repo_data = {
                "repo_url": "https://github.com/owner/repo",
                "repo_name": "owner__repo",
                "domain": "biology",
                "paper_url": "https://paperswithcode.com/paper/test",
            }
            repo_id = get_or_create_repo(conn, repo_data)

            assert repo_id > 0

            # Getting same repo should return existing ID
            repo_id2 = get_or_create_repo(conn, repo_data)
            assert repo_id2 == repo_id

            conn.close()

    def test_insert_file(self) -> None:
        """Test inserting a file record."""
        from real_world_demo.database import get_or_create_repo, init_db, insert_file

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Create repo first
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/owner/repo",
                    "repo_name": "owner__repo",
                    "domain": "biology",
                },
            )

            file_data = {
                "file_path": "files/owner__repo/model.py",
                "original_path": "src/model.py",
                "ml_imports": ["torch", "sklearn"],
                "scientific_imports": ["numpy"],
                "file_size": 1000,
                "line_count": 50,
            }
            file_id = insert_file(conn, repo_id, file_data)

            assert file_id > 0

            conn.close()

    def test_analysis_run_lifecycle(self) -> None:
        """Test starting and completing an analysis run."""
        from real_world_demo.database import (
            complete_analysis_run,
            init_db,
            list_runs,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Start run
            run_id = start_analysis_run(
                conn,
                total_files=10,
                config={"max_files": 10},
                model_name="test-model",
                notes="Test run",
            )

            assert run_id > 0

            # Complete run
            complete_analysis_run(
                conn,
                run_id=run_id,
                analyzed=8,
                with_findings=3,
                total_findings=5,
                status="completed",
            )

            # Verify via list_runs
            runs = list_runs(conn, limit=1)
            assert len(runs) == 1
            assert runs[0].run_id == run_id
            assert runs[0].total_files == 10
            assert runs[0].analyzed_files == 8
            assert runs[0].files_with_findings == 3
            assert runs[0].total_findings == 5

            conn.close()

    def test_insert_findings(self) -> None:
        """Test inserting findings for a file analysis."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up hierarchy
            repo_id = get_or_create_repo(
                conn,
                {"repo_url": "https://github.com/o/r", "repo_name": "o__r", "domain": "biology"},
            )
            file_id = insert_file(conn, repo_id, {"file_path": "files/o__r/model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success", duration=1.5)

            # Insert findings
            findings = [
                {
                    "id": "dl-001",
                    "category": "data-leakage",
                    "severity": "high",
                    "confidence": 0.85,
                    "issue": "Test issue",
                    "location": {"lines": [10, 11], "snippet": "code"},
                },
                {
                    "id": "rep-002",
                    "category": "reproducibility",
                    "severity": "medium",
                    "confidence": 0.7,
                    "issue": "Another issue",
                    "location": {},
                },
            ]
            count = insert_findings(conn, analysis_id, findings)

            assert count == 2

            # Verify findings in DB
            cursor = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE file_analysis_id = ?",
                (analysis_id,),
            )
            assert cursor.fetchone()[0] == 2

            conn.close()

    def test_get_run_stats(self) -> None:
        """Test getting statistics for an analysis run."""
        from real_world_demo.database import (
            get_or_create_repo,
            get_run_stats,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Create data
            repo_id = get_or_create_repo(
                conn,
                {"repo_url": "https://github.com/o/r", "repo_name": "o__r", "domain": "biology"},
            )
            file_id = insert_file(conn, repo_id, {"file_path": "files/o__r/model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            findings = [
                {
                    "id": "dl-001",
                    "category": "data-leakage",
                    "severity": "high",
                    "confidence": 0.9,
                    "issue": "Test",
                }
            ]
            insert_findings(conn, analysis_id, findings)

            # Get stats
            stats = get_run_stats(conn, run_id)

            assert stats.run_id == run_id
            assert stats.total_repos == 1
            assert len(stats.by_category) >= 1

            conn.close()


class TestGenerateReport:
    """Tests for generate_report module."""

    def test_generate_markdown_report_empty_db(self) -> None:
        """Test report generation with empty database."""
        from real_world_demo.database import init_db
        from real_world_demo.generate_report import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            report = generate_markdown_report(conn)

            assert "No Analysis Data" in report
            conn.close()

    def test_generate_markdown_report_with_data(self) -> None:
        """Test report generation with actual data."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )
        from real_world_demo.generate_report import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up data
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/test/repo",
                    "repo_name": "test__repo",
                    "domain": "biology",
                },
            )
            file_id = insert_file(
                conn,
                repo_id,
                {
                    "file_path": "files/test__repo/model.py",
                    "original_path": "src/model.py",
                },
            )
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            insert_findings(
                conn,
                analysis_id,
                [
                    {
                        "id": "dl-001",
                        "category": "data-leakage",
                        "severity": "high",
                        "confidence": 0.85,
                        "issue": "Test issue",
                        "explanation": "Test explanation",
                        "location": {"lines": [10], "snippet": "test code"},
                    }
                ],
            )

            report = generate_markdown_report(conn, run_id)

            assert "Real-World Scientific ML Code Analysis Report" in report
            assert "data-leakage" in report
            assert "dl-001" in report
            assert "biology" in report
            conn.close()

    def test_get_example_findings(self) -> None:
        """Test getting example findings grouped by category."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )
        from real_world_demo.generate_report import get_example_findings

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up data
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/test/repo",
                    "repo_name": "test__repo",
                    "domain": "biology",
                },
            )
            file_id = insert_file(conn, repo_id, {"file_path": "model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            insert_findings(
                conn,
                analysis_id,
                [
                    {
                        "id": "dl-001",
                        "category": "data-leakage",
                        "severity": "high",
                        "confidence": 0.9,
                        "issue": "Issue 1",
                        "location": {},
                    },
                    {
                        "id": "rep-001",
                        "category": "reproducibility",
                        "severity": "medium",
                        "confidence": 0.8,
                        "issue": "Issue 2",
                        "location": {},
                    },
                ],
            )

            examples = get_example_findings(conn, run_id)

            assert "data-leakage" in examples
            assert "reproducibility" in examples
            assert len(examples["data-leakage"]) == 1
            assert examples["data-leakage"][0]["pattern_id"] == "dl-001"
            conn.close()


class TestModels:
    """Tests for Pydantic models."""

    def test_finding_model(self) -> None:
        """Test Finding model validation."""
        from real_world_demo.models import Finding, Severity

        finding = Finding(
            pattern_id="dl-001",
            category="data-leakage",
            severity=Severity.HIGH,
            confidence=0.85,
            issue="Data leakage detected",
        )

        assert finding.pattern_id == "dl-001"
        assert finding.severity == Severity.HIGH
        assert finding.confidence == 0.85

    def test_analysis_run_model(self) -> None:
        """Test AnalysisRun model."""
        from real_world_demo.models import AnalysisRun, RunStatus

        run = AnalysisRun(
            run_id=1,
            total_files=100,
            analyzed_files=95,
            files_with_findings=20,
            total_findings=35,
        )

        assert run.status == RunStatus.RUNNING
        assert run.total_files == 100

    def test_domain_stats_model(self) -> None:
        """Test DomainStats model."""
        from real_world_demo.models import DomainStats

        stats = DomainStats(
            domain="biology",
            total_files=50,
            analyzed_files=48,
            files_with_findings=10,
            total_findings=15,
            finding_rate=20.8,
        )

        assert stats.domain == "biology"
        assert stats.finding_rate == 20.8
