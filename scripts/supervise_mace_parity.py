"""Public refusals; privileged supervision lives in the external trust root."""


def supervise(*args, **kwargs):
    raise RuntimeError("Direct project supervision is refused; use the external trusted launcher")


def _supervise(*args, **kwargs):
    raise RuntimeError("Direct project supervision is refused; use the external trusted launcher")


def main():
    raise SystemExit("Direct supervisor main() is refused; use python -I -S "
                     "<external-trusted-launcher> --entry supervisor --policy <approved-policy> "
                     "--policy-sha256 <independently-pinned-digest>; see docs/run-mace-parity.md")


if __name__ == "__main__":
    raise SystemExit(main())
