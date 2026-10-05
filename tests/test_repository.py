from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_all_yaml_files_parse() -> None:
    for path in ROOT.glob("configs/**/*.y*ml"):
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path


def test_paper_tables_are_present() -> None:
    numbers = sorted(int(path.name.split("_")[1]) for path in (ROOT / "tables").glob("table_*.tex"))
    assert numbers == list(range(1, 20))


def test_reported_figures_are_present() -> None:
    assert len(list((ROOT / "figures").glob("figure_[1-4]_*.png"))) == 4


def test_final_training_grid_is_complete() -> None:
    configs = list((ROOT / "configs" / "training").glob("**/fold_*.yaml"))
    assert len(configs) == 50
    for path in configs:
        trainer = yaml.safe_load(path.read_text(encoding="utf-8"))["Trainer"]
        model = trainer["Model"]
        dataset = trainer["Dataset"]
        optimizer = model["UNetpp"]["optimizer"]
        scheduler = model["UNetpp"]["schedulers"]["StepLR"]
        assert model["classpath"] == "impactsynth.models:UNetpp"
        assert model["UNetpp"]["nb_channel"] == 5
        assert dataset["Patch"]["patch_size"] == [1, 320, 320]
        assert dataset["batch_size"] == 32
        assert optimizer["name"] == "AdamW"
        assert optimizer["lr"] == 0.001
        assert optimizer["weight_decay"] == 0.001
        assert scheduler["step_size"] == 10
        assert scheduler["gamma"] == 0.75
        assert trainer["it_validation"] == 2500
        assert trainer["it_lr_update"] == 2500  # one scheduler step per validation


def test_inference_uses_three_views() -> None:
    configs = list((ROOT / "configs" / "inference").glob("*_tta.yaml"))
    assert len(configs) == 4
    for path in configs:
        predictor = yaml.safe_load(path.read_text(encoding="utf-8"))["Predictor"]
        augmentations = predictor["Dataset"]["augmentations"]
        assert len(augmentations) == 2  # identity is implicit
        assert predictor["combine"] == "Concat"
        assert predictor["outputs_dataset"]["Head:Tanh"]["OutputDataset"]["name_class"] == "OutputDataset"


def test_metric_bundle_is_complete() -> None:
    assert len(list((ROOT / "results" / "raw" / "Evaluations").glob("**/Metric_TRAIN.json"))) == 828


def test_result_json_is_strictly_valid() -> None:
    def reject_non_finite(value: str) -> None:
        raise ValueError(value)

    for path in (ROOT / "results").glob("**/*.json"):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_non_finite)


def test_executable_files_have_no_cluster_paths() -> None:
    forbidden = tuple("/" + name + "/" for name in ("scratch", "home", "gpfs", "mnt"))
    roots = (ROOT / "configs", ROOT / "scripts", ROOT / "src")
    for base in roots:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert not any(item in text for item in forbidden), path


def test_script_tables_match_the_paper() -> None:
    outputs = {
        14: "table_14_task1_registration.tex",
        15: "table_15_task2_registration.tex",
        16: "table_16_task1_sam.tex",
        17: "table_17_task2_sam.tex",
        18: "table_18_task1_perceptual.tex",
        19: "table_19_task2_perceptual.tex",
    }
    pattern = re.compile(r"\\(?:b?mstd)\{([^}]+)\}\{([^}]+)\}\{([^}]*)\}")
    for number, output_name in outputs.items():
        reported_path = next((ROOT / "tables").glob(f"table_{number:02d}_*.tex"))
        reported = pattern.findall(reported_path.read_text(encoding="utf-8"))
        built = pattern.findall((ROOT / "tables" / "from_metrics" / output_name).read_text(encoding="utf-8"))
        assert reported
        assert all(cell in built for cell in reported), number


def test_documented_files_exist() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted(ROOT.glob("*/README.md")), ROOT / "Makefile"]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for path in set(re.findall(r"\b((?:scripts|configs|docs|tests|src)/[\w./-]+\.(?:py|md|yaml|yml))\b", text)):
            assert (ROOT / path).is_file(), f"{document.name} mentions {path}"
        for name in set(re.findall(r"\b((?:table_\d|figure_\d|published_tables)\w*\.py)\b", text)):
            assert list((ROOT / "scripts").rglob(name)), f"{document.name} mentions {name}"
