#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

TOKEN="${GITHUB_PAT:-${GITHUB_TOKEN:-}}"
GIT_AUTH_ARGS=()
FORCE_MIRROR=0

if [[ -n "$TOKEN" ]]; then
  AUTH_HEADER="$(printf 'x-access-token:%s' "$TOKEN" | base64 | tr -d '\n')"
  GIT_AUTH_ARGS=(
    -c credential.helper=
    -c "http.https://github.com/.extraheader=AUTHORIZATION: basic $AUTH_HEADER"
  )
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This folder is not a git repository: $ROOT_DIR"
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Git remote 'origin' is not configured."
  exit 1
fi

BRANCH="$(git branch --show-current)"
if [[ "$BRANCH" != "main" ]]; then
  echo "Current branch is '$BRANCH'. Please switch to 'main' before syncing."
  exit 1
fi

if [[ "${1:-}" == "--force-mirror" ]]; then
  FORCE_MIRROR=1
  shift
fi

COMMIT_MSG="${1:-Update website $(date '+%Y-%m-%d %H:%M:%S')}"
REMOTE_MAIN_EXISTS=0

if git "${GIT_AUTH_ARGS[@]}" ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
  REMOTE_MAIN_EXISTS=1
fi

echo "Staging local changes..."
git add -A

if git diff --cached --quiet; then
  echo "No new local changes to commit."
else
  echo "Creating commit: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG"
fi

if [[ "$FORCE_MIRROR" -eq 1 ]]; then
  if [[ "$REMOTE_MAIN_EXISTS" -eq 1 ]]; then
    echo "Force-mirror mode: pushing local main to overwrite remote main..."
    PUSH_CMD=(git "${GIT_AUTH_ARGS[@]}" push --force-with-lease origin main)
  else
    echo "Force-mirror mode on an empty remote: creating origin/main..."
    PUSH_CMD=(git "${GIT_AUTH_ARGS[@]}" push -u origin main)
  fi
else
  if [[ "$REMOTE_MAIN_EXISTS" -eq 1 ]]; then
    echo "Fetching latest origin/main..."
    git "${GIT_AUTH_ARGS[@]}" fetch origin main
    echo "Rebasing onto origin/main..."
    git "${GIT_AUTH_ARGS[@]}" pull --rebase origin main
    echo "Pushing to origin/main..."
    PUSH_CMD=(git "${GIT_AUTH_ARGS[@]}" push origin main)
  else
    echo "Remote main does not exist yet. Creating origin/main..."
    PUSH_CMD=(git "${GIT_AUTH_ARGS[@]}" push -u origin main)
  fi
fi

if ! "${PUSH_CMD[@]}"; then
  echo
  echo "Push failed."
  echo "If this repository needs GitHub token auth, rerun like this:"
  echo "GITHUB_PAT=your_token_here ./sync_site.sh"
  exit 1
fi

echo
echo "Sync finished."
git log -1 --oneline
