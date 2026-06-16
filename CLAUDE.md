## File Safety

- NEVER delete files or directories without first listing their contents and asking for explicit confirmation, especially hidden files (e.g., .thumbnails, .git). Always use `ls -la` before any `rm -rf`.

## Implementation Limits

- If a feature fails after 2 implementation attempts, STOP and propose an alternative approach or reduced scope rather than iterating on the same strategy. Ask the user whether to continue or pivot.

## Pre-Deployment Checklist

- Before any deploy or push: (1) verify all config files are UTF-8 encoded (not UTF-16 LE), (2) run `pip freeze > requirements.txt` to regenerate if encoding is suspect, (3) confirm the deployment target's runtime constraints (e.g., Android API level, Python version, network access).

## Python Encoding Hygiene

- When writing or editing any Python-related config files (requirements.txt, setup.cfg, pyproject.toml), always explicitly write with UTF-8 encoding. Verify with `file -I <filename>` after writes.
