# Windows MACE quickstart

28 October 2026. Start here if you use Windows. A Mac is not required for onboarding or loader development.

For the complete assignment, milestones, implementation contracts and delivery criteria, read the [full task brief](IDRIS_MACE_TASK_BRIEF_2026-10-28.md). This quickstart is its setup companion.

**First deliverable:** complete steps 1–3 and send the packet verification output plus your Windows/WSL details. Then work through the proposed setup in step 4, retaining either its outputs or the exact failure. No training or trained-model inference is part of this guide.

## 1. Check Windows and WSL in PowerShell

Open ordinary PowerShell and run:

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
wsl --list --verbose
```

Keep the output. If a suitable Ubuntu distribution already shows version 2, reuse it with `wsl --distribution Ubuntu`, using its actual listed name. Preserve existing distributions and defaults.

If WSL is unavailable, Windows 11 or Windows 10 build 19041 or later can use this command in administrator PowerShell:

```powershell
wsl --install -d Ubuntu
```

Restart if requested, open Ubuntu and complete its initial username/password setup. Run `wsl --list --verbose` again in PowerShell and confirm version 2. If installation is blocked, send the command and error; stop that setup attempt.

[Microsoft WSL instructions](https://learn.microsoft.com/en-us/windows/wsl/install).

If you already downloaded the packet in Windows, this optional PowerShell command displays its checksum:

```powershell
Get-FileHash "$env:USERPROFILE\Downloads\mace-idris-handoff-2026-10-18-v2.tar.gz" -Algorithm SHA256
```

Expected SHA-256: `ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119`. [PowerShell hash documentation](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/get-filehash).

## 2. Prepare Ubuntu prerequisites

The proposed package target is **x86_64 Linux with glibc 2.28 or later**, using Python 3.12.13. Windows on ARM is not covered by this recipe.

Run in the Ubuntu terminal:

```bash
uname -m
getconf GNU_LIBC_VERSION
sudo apt-get update && sudo apt-get install -y curl git ca-certificates python3
```

Expect `x86_64` and glibc at least 2.28. Send other output before choosing another package target. Store the work in Ubuntu's home directory. [Microsoft filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

## 3. Download and verify the packet

Run this complete block in Ubuntu. It refuses to reuse an existing onboarding folder.

```bash
(
set -euo pipefail
onboarding="$HOME/group9-mace-onboarding-2026-10-28"
if [ -e "$onboarding" ]; then
  printf '%s\n' 'Folder already exists. Preserve it and report its location.' >&2
  exit 1
fi
test "$(uname -m)" = x86_64
python3 -c 'import platform; n,v=platform.libc_ver(); assert n=="glibc" and tuple(map(int,v.split(".")[:2])) >= (2,28), (n,v)'
mkdir "$onboarding"
cd "$onboarding"
curl -fL 'https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/mace-idris-handoff-2026-10-18-v2.tar.gz' -o mace-idris-handoff-2026-10-18-v2.tar.gz
printf '%s  %s\n' 'ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119' mace-idris-handoff-2026-10-18-v2.tar.gz | sha256sum -c -
tar -xzf mace-idris-handoff-2026-10-18-v2.tar.gz
python3 mace-idris-handoff/verify_bundle.py | tee packet-verification.txt
{ uname -a; getconf GNU_LIBC_VERSION; python3 --version; } > wsl-system.txt
)
```

Expected packet output: `PASS: 11 inventoried files; no model loaded`. The system Python in this step only verifies file bytes; it is not the proposed MACE runtime.

Send `packet-verification.txt`, `wsl-system.txt` and the PowerShell output from step 1. This completes the first access/setup milestone even while package work continues.

## 4. Prepare the proposed CPU environment

**Status: proposed and not executed on WSL.** Core package versions and the official CPU wheel are available. A wheel-only resolution failed because `python-hostlist==2.3.0` has no usable wheel. No pre-resolved Linux lock is published.

The recipe preserves the original 44 base-version pins, with the explicit Torch build change from `2.14.1` to `2.14.1+cpu`. The source-only hostlist package is pinned by its original source archive hash. Only that package has a planned source-build exception, using pinned setuptools without build isolation.

Idris generates and retains the lock on his own WSL environment. These commands install software on that PC; they do not load the molecular model. Stop and retain any failed command/output rather than changing package versions.

Run this complete block in Ubuntu after step 3:

```bash
(
set -euo pipefail
cd "$HOME/group9-mace-onboarding-2026-10-28"
for path in source tools .venv-mace-wsl; do
  if [ -e "$path" ]; then
    printf 'Preserve existing %s and report the completed stage.\n' "$path" >&2
    exit 1
  fi
done
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch post-midterm-mace-readiness --filter=blob:none 'https://github.com/Shedrackeze002/idl-molecular-force-consistency.git' source
git -C source switch -c idris-mace-loader adf1b13819015cbfb1f2fbb79a79df132e7a4545
git -C source rev-parse HEAD > source-commit.txt
printf '%s  %s\n' '005d1e58ff6f3087993c1747ecd4b51164bb5e8c0614f9f4b23feca01ebcbdfa' source/configs/mace/wsl2-cpu-2026-10-28/requirements.in | sha256sum -c -
curl -LsSf https://astral.sh/uv/0.11.16/install.sh -o uv-install.sh
UV_INSTALL_DIR="$PWD/tools" UV_NO_MODIFY_PATH=1 sh uv-install.sh
uv_bin="$PWD/tools/uv"
"$uv_bin" --version
"$uv_bin" python install 3.12.13
"$uv_bin" venv --python 3.12.13 .venv-mace-wsl
python_bin="$PWD/.venv-mace-wsl/bin/python"
"$uv_bin" pip install --python "$python_bin" --no-deps 'https://files.pythonhosted.org/packages/95/9c/c510029fc6ef33a6275cd2c5d3cecd6613dfd6aa401d57c54f1c18852ccf/setuptools-84.0.0-py3-none-any.whl#sha256=51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670'
"$uv_bin" pip compile source/configs/mace/wsl2-cpu-2026-10-28/requirements.in --python .venv-mace-wsl/bin/python --no-build-isolation --only-binary :all: --no-binary python-hostlist --generate-hashes --default-index https://pypi.org/simple --output-file wsl-requirements.lock 2>&1 | tee lock-build.txt
"$python_bin" - <<'PY'
from pathlib import Path
import re
source=Path('source/configs/mace/wsl2-cpu-2026-10-28/requirements.in').read_text()
locked=Path('wsl-requirements.lock').read_text()
pattern=r'^([A-Za-z0-9_.-]+)(?:==| @ )'
expected=set(re.findall(pattern,source,re.M))
actual=set(re.findall(pattern,locked,re.M))
assert expected == actual, 'Dependency set changed; retain the lock for review'
assert not any(n.startswith(('nvidia','triton')) for n in actual), 'Unexpected GPU dependency'
print('Proposed lock retains the expected 44-package set')
PY
"$uv_bin" pip sync --python "$python_bin" --require-hashes --no-build-isolation --only-binary :all: --no-binary python-hostlist wsl-requirements.lock 2>&1 | tee package-install.txt
"$uv_bin" pip check --python "$python_bin" 2>&1 | tee package-check.txt
comparison_exit=0
"$python_bin" mace-idris-handoff/verify_environment.py > mac-reference-comparison.json 2> mac-reference-comparison-stderr.txt || comparison_exit=$?
printf '%s\n' "$comparison_exit" > mac-reference-comparison-exit.txt
printf '%s\n' 'Setup outputs saved. Send them for environment review; do not run the model yet.'
)
```

[Official uv installation](https://docs.astral.sh/uv/getting-started/installation/) and [package command reference](https://docs.astral.sh/uv/reference/cli/).

The archived verifier compares against the original Mac identity. Its expected Torch-version difference and platform flag must remain visible; its output is not a Linux acceptance certificate or a prediction result.

Send the generated lock, source commit, build/install/check logs and all `mac-reference-comparison` files. If a command fails, send its exact command, output and completed stage. Preserve the folder for diagnosis.

## 5. Continue the assigned work

Continue through milestones I02–I08 in the [full task brief](IDRIS_MACE_TASK_BRIEF_2026-10-28.md), including test-tool installation, implementation, platform admission, bounded validation, independent audit, report and integration handoff. Olufunke owns integration. No repeat training is required.

The original Mac scoring remains reference evidence. The accepted 26 October numerical specification remains unchanged. Actual WSL model loading and prediction comparison need a reviewed, bounded platform amendment before execution; its tolerances are not changed by this guide.
