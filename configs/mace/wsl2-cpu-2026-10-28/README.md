# Proposed WSL2 CPU setup

Start with the [Windows quickstart](../../../docs/team/WINDOWS_MACE_QUICKSTART_2026-10-28.md).

Target: x86_64 Linux, glibc at least 2.28, Python 3.12.13 and CPU float64 inference after separate numerical approval. This configuration is proposed, not a verified WSL installation.

The input preserves all 44 original base-version pins. Torch uses the official `2.14.1+cpu` build instead of original Mac `2.14.1`. The two direct artifacts are hash-pinned: CPU Torch wheel and source-only `python-hostlist==2.3.0` archive.

Input SHA-256: `005d1e58ff6f3087993c1747ecd4b51164bb5e8c0614f9f4b23feca01ebcbdfa`.

The wheel-only resolution failed because hostlist has no usable wheel. No Linux lock is published. The guide asks Idris to generate one in WSL, with pinned setuptools84.0.0 bootstrapped and isolation disabled for the planned hostlist source build. All other packages are restricted to wheels; an unexpected dependency set blocks installation pending review.

The source `setup.py` was inspected without execution and declares no runtime install dependencies or extension modules. Metadata-Version2.1 alone was not treated as proof. Actual building, installation, imports and prediction agreement remain untested.

Keep original packet/model bytes and identities unchanged. The archived Mac environment verifier remains a comparison tool, never Linux acceptance by relabelling.
