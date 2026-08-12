from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_engine_images_match_product_policy():
    cuda = _read("Dockerfile")
    rocm = _read("Dockerfile.rocm")
    cuda_multigpu = _read("Dockerfile.axolotl.cuda")

    assert "unsloth==${UNSLOTH_VERSION}" in cuda
    assert "requirements-axolotl-rocm.txt" in rocm
    assert "unsloth[amd]" not in rocm
    assert "requirements-axolotl-cuda.txt" in cuda_multigpu
    assert "accelerate.commands.launch" in cuda_multigpu
    assert "xformers==${XFORMERS_VERSION}" in cuda_multigpu
    assert 'torch.__version__.split("+", 1)[0]' in cuda_multigpu

    for dockerfile in (cuda, rocm, cuda_multigpu):
        assert "sft_format.py" in dockerfile
        assert "worker.py" in dockerfile
        assert "OPENREEF_THIRD_PARTY_NOTICES.md" in dockerfile
        assert "write_dependency_inventory.py" in dockerfile
        assert "requirements-build.txt" in dockerfile
        assert "requirements-worker-io.txt" in dockerfile


def test_direct_requirements_are_exactly_pinned():
    requirement_files = list(ROOT.glob("requirements*.txt"))
    assert requirement_files
    for path in requirement_files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            assert "==" in line, f"floating dependency in {path.name}: {line}"

    assert '"ogpu[service]>=${OGPU_VERSION}"' not in _read("Dockerfile")
    assert '"ogpu[service]==${OGPU_VERSION}"' in _read("Dockerfile")


def test_multigpu_compose_is_explicit_axolotl_opt_in():
    compose = yaml.safe_load(_read("docker-compose-nvidia-multigpu.yml"))
    finetune = compose["services"]["finetune"]
    device = finetune["deploy"]["resources"]["reservations"]["devices"][0]

    assert device["count"] == "all"
    assert any(
        item == "OPENREEF_TRAIN_ENGINE=axolotl"
        for item in finetune["environment"]
    )
    assert "cuda-multigpu-latest" in finetune["image"]


def test_default_nvidia_compose_keeps_single_gpu_but_allows_opt_in():
    text = _read("docker-compose-nvidia.yml")
    assert "count: ${OPENREEF_GPU_COUNT:-1}" in text
    assert "OPENREEF_TRAIN_ENGINE=${OPENREEF_TRAIN_ENGINE:-auto}" in text
    assert "OPENREEF_NUM_GPUS=${OPENREEF_NUM_GPUS:-}" in text


def test_workflow_covers_sources_and_never_pushes_on_main():
    workflow = yaml.load(
        _read(".github/workflows/docker-images.yml"), Loader=yaml.BaseLoader
    )
    trigger = workflow["on"]
    required_paths = {
        "Dockerfile",
        "Dockerfile.rocm",
        "Dockerfile.axolotl.cuda",
        "worker.py",
        "training_config.py",
        "sft_format.py",
        "requirements*.txt",
        "tests/**",
    }

    assert required_paths.issubset(set(trigger["push"]["paths"]))
    assert required_paths.issubset(set(trigger["pull_request"]["paths"]))

    plan_steps = workflow["jobs"]["plan"]["steps"]
    resolve = next(step for step in plan_steps if step.get("id") == "flags")
    script = resolve["run"]
    assert script.count("PUSH=true") == 1
    assert 'if [ "${{ github.event_name }}" = "workflow_dispatch" ]' in script
    assert "gpu_smoke_commit" in script


def test_actions_are_pinned_and_cuda_lock_is_used():
    workflow = _read(".github/workflows/docker-images.yml")
    assert "actions/checkout@v" not in workflow
    assert "docker/build-push-action@v" not in workflow
    assert "./scripts/load_cuda_pins.sh --gha" in workflow
    assert "steps.cuda_pins.outputs.build_args" in workflow
    assert "sbom:" in workflow


def test_dependency_refresh_opens_pr_without_publishing():
    workflow = _read(".github/workflows/refresh-unsloth-pins.yml")
    assert "resolve_unsloth_pins.py --write" in workflow
    assert "requirements-build.txt" in workflow
    assert "peter-evans/create-pull-request@" in workflow
    assert "docker/build-push-action" not in workflow


def test_ogpu_refresh_stays_on_the_compatible_api_line():
    resolver = _read("scripts/resolve_unsloth_pins.py")
    assert 'OGPU_COMPAT_SPEC = SpecifierSet(">=0.2.3,<0.3")' in resolver
    assert 'max_in_spec("ogpu", OGPU_COMPAT_SPEC)' in resolver
    dependabot = _read(".github/dependabot.yml")
    assert "dependency-name: ogpu" in dependabot
    assert 'versions: [\">=0.3\"]' in dependabot


def test_provider_updater_is_idle_only_and_has_rollback():
    updater = _read("scripts/provider-safe-update.sh")
    assert "training.active" in updater
    assert "updating" in updater
    assert "rollback" in updater
    assert "docker exec" in updater
    assert "prune_old_rollback_tags" in updater
    assert "tail -n +4" in updater
    assert "docker image prune" not in updater


def test_build_script_exposes_multigpu_without_changing_auto_default():
    script = _read("build_image.sh")
    assert "--cuda-multigpu" in script
    assert "DOCKERFILE=Dockerfile.axolotl.cuda" in script
    assert 'TAG_ROLLING="${REPO}:cuda-multigpu-latest"' in script


def test_build_script_rejects_unsafe_publication():
    script = ROOT / "build_image.sh"

    missing_sha = subprocess.run(
        [str(script), "--cuda", "--push"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_sha.returncode != 0
    assert "requires an immutable SHA tag" in missing_sha.stderr

    missing_smoke = subprocess.run(
        [
            str(script),
            "--cuda",
            "--tag",
            "cuda-deadbee",
            "--push",
            "--publish-latest",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_smoke.returncode != 0
    assert "requires --confirm-gpu-smoke" in missing_smoke.stderr
