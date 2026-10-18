import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.setup.cuda_arch import normalize_arches


ROOT = Path(__file__).resolve().parents[2]


class CudaArchitectureTests(unittest.TestCase):
    def test_original_targets_in_both_profiles(self):
        for version in ("12.1", "12.8"):
            for arch in ("7.0", "7.5", "8.0", "8.6", "8.9", "9.0"):
                with self.subTest(version=version, arch=arch):
                    self.assertEqual(normalize_arches(arch, version), arch)

    def test_blackwell_requires_cuda128(self):
        for arch in ("10.0", "12.0", "12.0+PTX"):
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, "--cuda128"):
                    normalize_arches(arch, "12.1")
                self.assertEqual(normalize_arches(arch, "12.8"), arch)

    def test_duplicate_targets_and_whitespace(self):
        self.assertEqual(normalize_arches(" 8.9;8.0 8.9\n9.0+PTX ", "12.1"),
                         "8.0;8.9;9.0+PTX")

    def test_rejects_invalid_or_unvalidated_targets(self):
        for value in ("", " ", "Ada", "sm_120", "12", "10.1", "8.9,12.0", "12.0+PTX+PTX"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_arches(value, "12.8")

    def test_default_version_comes_from_installed_torch(self):
        for version in ("12.1", "12.8"):
            torch = SimpleNamespace(version=SimpleNamespace(cuda=version))
            with self.subTest(version=version), patch.dict("sys.modules", {"torch": torch}):
                if version == "12.1":
                    with self.assertRaises(ValueError):
                        normalize_arches("12.0")
                else:
                    self.assertEqual(normalize_arches("12.0"), "12.0")


class RuntimeArchitectureTests(unittest.TestCase):
    def load_runtime(self, arch):
        env = os.environ.copy()
        env.pop("TORCH_CUDA_ARCH_LIST", None)
        if arch is not None:
            env["TORCH_CUDA_ARCH_LIST"] = arch
        with tempfile.TemporaryDirectory() as temp:
            conda_sh = Path(temp) / "etc/profile.d/conda.sh"
            conda_sh.parent.mkdir(parents=True)
            conda_sh.write_text("# Test Conda bootstrap; activation is not needed.\n")
            result = subprocess.run(
                ["bash", "-c", '''
set -eu
# galatea_load_conda invokes `conda info --base`; return the supplied test root.
conda() { printf '%s\\n' "$test_conda_root"; }
test_conda_root="$2"
source "$1/scripts/setup/runtime_env.sh"
galatea_load_conda "$1"
bash -c 'printf "%s" "${TORCH_CUDA_ARCH_LIST-not_exported}"'
''', "test", str(ROOT), temp],
                env=env, check=True, capture_output=True, text=True,
            )
        return result.stdout

    def test_unset_arch_is_exported_empty_for_gpu_detection(self):
        self.assertEqual(self.load_runtime(None), "")

    def test_explicit_empty_arch_is_preserved(self):
        self.assertEqual(self.load_runtime(""), "")

    def test_explicit_build_targets_are_preserved(self):
        self.assertEqual(self.load_runtime("8.9;12.0+PTX"), "8.9;12.0+PTX")


if __name__ == "__main__":
    unittest.main()
