# Releasing localcaption

Releases are fully automated via GitHub Actions and **PyPI Trusted
Publishing (OIDC)**. No PyPI API tokens are stored in GitHub Secrets.

## One-time setup (≈ 3 minutes)

Do this once, before the first release.

### 1. Register the project on PyPI as a "pending publisher"

Trusted Publishing lets you claim a project name on PyPI without having to
do a manual upload first.

1. Sign in to <https://pypi.org> (create the account if needed).
2. Go to **Account settings → Publishing → Add a new pending publisher**.
3. Fill in:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `localcaption` |
   | Owner | `jatinkrmalik` |
   | Repository name | `localcaption` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

PyPI will now trust uploads coming from `release.yml` running in the
`pypi` environment of `jatinkrmalik/localcaption`, and from nothing else.

### 2. Create the matching GitHub environment

The workflow gates the publish job behind a GitHub Environment. This is
where you can later add manual approval if you want a "are you sure?"
checkpoint on every release.

1. In the repo: **Settings → Environments → New environment**.
2. Name it exactly `pypi` (must match the value you used on PyPI).
3. Optional: add yourself as a **Required reviewer** to gate every
   release on a manual approval click.

That's it for one-time setup.

## Cutting a release

```bash
# 1. Make sure main is green and all the changes you want are merged.
git checkout main
git pull

# 2. Bump the version in pyproject.toml (X.Y.Z, no leading 'v').
$EDITOR pyproject.toml

# 3. Move the [Unreleased] block in CHANGELOG.md under a new [X.Y.Z] header.
$EDITOR CHANGELOG.md

# 4. Commit and push.
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): X.Y.Z"
git push origin main

# 5. Tag and push.
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The push of the `v*` tag triggers `.github/workflows/release.yml`, which
runs three jobs:

1. **build** — produces `dist/*.whl` + `dist/*.tar.gz`, runs `twine check
   --strict`, and verifies that `pyproject.toml` version matches the tag
   (hard failure if not).
2. **publish-pypi** — uploads to <https://pypi.org> via OIDC.
3. **github-release** — creates a GitHub Release page with auto-generated
   notes and the wheel + sdist attached. Runs in parallel with publish-pypi.

About 1–2 minutes end to end. Watch it with:

```bash
gh run watch --repo jatinkrmalik/localcaption
```

## Verifying a release

After CI goes green:

```bash
# Should list the new version
pip index versions localcaption

# Install it in a throwaway env to smoke-test
pipx install --force --suffix=-test localcaption
localcaption-test --version
```

## Recovering from a bad release

PyPI does **not** allow re-uploading the same version, even if you delete
it. The recipe is:

1. **Yank** the bad version on pypi.org (Project page → Manage → Yank).
   Yanked versions stay installable by exact pin (so anyone who already
   pinned to it isn't broken) but disappear from `pip install localcaption`
   resolvers.
2. Bump to the next patch version in `pyproject.toml`.
3. Tag and push as normal.
