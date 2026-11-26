# Large report files

The two large width/rate replacement JSON reports use Git LFS. One is the historical invalid report; the corrected report is authoritative. Their exact exported sizes and SHA256 identities are recorded in [the research export manifest](../../reports/research-export-manifest.json).

Install Git LFS for your platform, then initialize it in your own checkout and fetch the report payloads:

```sh
git lfs install --local
git lfs pull
git lfs fsck
```

A text file beginning with `version https://git-lfs.github.com/spec/v1` is a pointer, not the report payload. Compare the hydrated file to the manifest before reading it as evidence. Complete original raw run archives are retained separately; see [the reproduction note](../REPRODUCTION.md).
