"""Run every test module in this folder. Usage:  python tests/run_all.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Order matters only for readability: cheap and specific first.
ORDER = ["test_prompt.py", "test_tracking.py", "test_posture.py",
         "test_rules_engine.py", "test_integration.py",
         "test_api.py", "test_dashboard.py", "test_readme.py",
         "test_docs_rules.py"]

NOISE = ("FutureWarning", "warnings.warn", "INFO:", "W0000", "absl",
         "clean_up_tokenization", "Loading Florence-2", "Florence-2 ready",
         "TensorFlow Lite", "inference_feedback_manager", "Feedback manager")

def main():
    failed = []
    for name in ORDER:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        print(f"\n{'=' * 66}\n  {name}\n{'=' * 66}")
        p = subprocess.run([sys.executable, path], cwd=ROOT,
                           capture_output=True, text=True)
        out = "\n".join(ln for ln in (p.stdout + p.stderr).splitlines()
                        if not any(n in ln for n in NOISE))
        print(out.strip())
        if p.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 66}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    else:
        print("  ALL SUITES PASSED")
    print("=" * 66)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
