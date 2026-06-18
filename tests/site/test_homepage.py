from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_INDEX = REPO_ROOT / "site" / "index.html"
SITE_DOCS_REDIRECT = REPO_ROOT / "site" / "docs.html"
SITE_DOCS_INDEX = REPO_ROOT / "site" / "docs" / "index.html"
SITE_DOCS_ARCH = REPO_ROOT / "site" / "docs" / "architecture.html"
SITE_DOCS_DATASET = REPO_ROOT / "site" / "docs" / "dataset.html"
SITE_DOCS_TRAINING = REPO_ROOT / "site" / "docs" / "training.html"
SITE_CSS = REPO_ROOT / "site" / "assets" / "site.css"
SITE_JS = REPO_ROOT / "site" / "assets" / "site.js"
SITE_LOSS = REPO_ROOT / "site" / "assets" / "loss.png"
PAGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
README = REPO_ROOT / "readme.md"


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


class HomepageSmokeTests(unittest.TestCase):
    def test_core_site_files_exist(self) -> None:
        for path in (
            SITE_INDEX,
            SITE_DOCS_REDIRECT,
            SITE_DOCS_INDEX,
            SITE_DOCS_ARCH,
            SITE_DOCS_DATASET,
            SITE_DOCS_TRAINING,
            SITE_CSS,
            SITE_JS,
        ):
            self.assertTrue(path.exists(), f"missing {path.relative_to(REPO_ROOT)}")

    def test_index_references_shared_assets(self) -> None:
        html = read_text(SITE_INDEX)
        self.assertIn('href="assets/site.css"', html)
        self.assertIn('src="assets/site.js"', html)

    def test_index_contains_required_sections(self) -> None:
        html = read_text(SITE_INDEX)
        for marker in (
            'id="top"',
            'id="hero"',
            'id="metrics"',
            'id="architecture"',
            'id="quick-start"',
            'id="implementation-notes"',
            'id="training-curve"',
        ):
            self.assertIn(marker, html)

    def test_index_contains_explicit_project_links(self) -> None:
        html = read_text(SITE_INDEX)
        self.assertGreaterEqual(
            html.count("https://github.com/fangpin/siamese-pytorch"),
            3,
            "expected GitHub links in header, hero, and footer",
        )
        self.assertIn(
            "https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf",
            html,
        )
        self.assertIn("https://github.com/brendenlake/omniglot", html)
        self.assertIn("python3 train.py", html)
        self.assertIn('href="docs/"', html)

    def test_docs_index_contains_system_map_and_chapter_links(self) -> None:
        html = read_text(SITE_DOCS_INDEX)
        for marker in (
            'id="overview"',
            'id="repo-map"',
            'href="architecture.html"',
            'href="dataset.html"',
            'href="training.html"',
        ):
            self.assertIn(marker, html)
        self.assertIn("Omniglot", html)
        self.assertIn('href="../index.html"', html)
        self.assertIn("https://github.com/fangpin/siamese-pytorch", html)

    def test_docs_chapter_pages_cover_major_subsystems(self) -> None:
        arch_html = read_text(SITE_DOCS_ARCH)
        dataset_html = read_text(SITE_DOCS_DATASET)
        training_html = read_text(SITE_DOCS_TRAINING)

        self.assertIn('id="architecture-boundary"', arch_html)
        self.assertIn("model.py", arch_html)
        self.assertIn("forward_one", arch_html)

        self.assertIn('id="dataset-boundary"', dataset_html)
        self.assertIn("mydataset.py", dataset_html)
        self.assertIn("OmniglotTrain", dataset_html)
        self.assertIn("OmniglotTest", dataset_html)

        self.assertIn('id="training-boundary"', training_html)
        self.assertIn("train.py", training_html)
        self.assertIn("BCEWithLogitsLoss", training_html)
        self.assertIn("DataParallel", training_html)

    def test_site_js_defaults_to_english(self) -> None:
        script = read_text(SITE_JS)
        self.assertIn('const DEFAULT_LANGUAGE = "en";', script)
        self.assertIn("const translations =", script)

    def test_site_js_persists_language_choice(self) -> None:
        script = read_text(SITE_JS)
        self.assertIn("localStorage", script)
        self.assertIn("document.documentElement.lang", script)
        self.assertIn('querySelectorAll("[data-lang]")', script)
        self.assertIn("GitHub 仓库", script)
        self.assertIn("docsNav.training", script)

    def test_site_css_defines_console_layout(self) -> None:
        css = read_text(SITE_CSS)
        for marker in (
            ".page-shell",
            ".topbar",
            ".hero-grid",
            ".panel",
            ".stats-grid",
            ".architecture-visual",
            ".footer-links",
        ):
            self.assertIn(marker, css)

    def test_site_css_contains_mobile_breakpoints(self) -> None:
        css = read_text(SITE_CSS)
        self.assertIn("@media (max-width: 1100px)", css)
        self.assertIn("@media (max-width: 720px)", css)

    def test_loss_curve_is_copied_into_site_assets(self) -> None:
        self.assertTrue(SITE_LOSS.exists(), "missing site/assets/loss.png")
        self.assertGreater(SITE_LOSS.stat().st_size, 0)

    def test_pages_workflow_deploys_site_directory(self) -> None:
        workflow = read_text(PAGES_WORKFLOW)
        self.assertIn("actions/configure-pages", workflow)
        self.assertIn("actions/upload-pages-artifact", workflow)
        self.assertIn("actions/deploy-pages", workflow)
        self.assertIn("path: ./site", workflow)

    def test_readme_links_back_to_site_and_docs(self) -> None:
        readme = read_text(README)
        self.assertIn("https://fangpin.github.io/siamese-pytorch/", readme)
        self.assertIn("https://fangpin.github.io/siamese-pytorch/docs/", readme)


if __name__ == "__main__":
    unittest.main()
