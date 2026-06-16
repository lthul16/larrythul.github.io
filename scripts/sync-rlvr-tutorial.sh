#!/usr/bin/env bash
#
# sync-rlvr-tutorial.sh — copy the RLVR recurrence tutorial authored in the
# ../rlvr-tutorial repo into this Hugo site as real, committable files.
#
# Same idea as scripts/sync-blog.sh: the CI build has no access to the source repo
# and Hugo won't follow symlinks that escape the project root, so publishable content
# must live here as real files. rlvr-tutorial/blog is the source of truth; this site
# holds a published snapshot.
#
# This post differs from the forecaster post: its figures live in a nested tree
# (rlvr-tutorial/figures/...) and are referenced from the markdown as ../figures/<path>.
# A leaf bundle can't reach outside itself with ../, so we (1) copy only the figures the
# post actually references, preserving their subpath under <bundle>/figures/, and
# (2) rewrite ../figures/ -> figures/ in the synced markdown so the links resolve.
#
# Usage:
#   scripts/sync-rlvr-tutorial.sh
#   RLVR_TUTORIAL=/path scripts/sync-rlvr-tutorial.sh   # override source repo location
#
set -euo pipefail

# --- locate repos -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_REPO="${RLVR_TUTORIAL:-$SITE_ROOT/../rlvr-tutorial}"

if [[ ! -d "$SRC_REPO/blog" ]]; then
  echo "error: source repo not found: $SRC_REPO/blog" >&2
  echo "       set RLVR_TUTORIAL to the rlvr-tutorial repo path." >&2
  exit 1
fi
SRC_REPO="$(cd "$SRC_REPO" && pwd)"

# --- post definition --------------------------------------------------------
# The markdown references figures relative to the repo root as ../figures/<path>.
SRC_MD="$SRC_REPO/blog/rlvr-recurrence.md"
SLUG="teaching-a-small-language-model-to-solve-recurrences-with-rlvr-from-first-principles"
DEST_DIR="$SITE_ROOT/content/posts/$SLUG"
DEST_MD="$DEST_DIR/index.md"

if [[ ! -f "$SRC_MD" ]]; then
  echo "error: source markdown not found: $SRC_MD" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

# --- markdown ---------------------------------------------------------------
[[ -L "$DEST_MD" ]] && { echo "  removing stale symlink: $DEST_MD"; rm -f "$DEST_MD"; }
cp "$SRC_MD" "$DEST_MD"
# Rewrite ../figures/ -> figures/ so the images resolve as bundle resources.
sed -i '' 's#](\.\./figures/#](figures/#g' "$DEST_MD"
# Publish on this site (source repo may keep draft: true for its own preview).
sed -i '' 's/^draft:[[:space:]]*true/draft: false/' "$DEST_MD"
echo "  md  -> content/posts/$SLUG/index.md  (rewrote ../figures/ -> figures/, draft: false)"

# --- figures (only the referenced ones, preserving structure) ---------------
# Collect figures/<path> targets from the rewritten markdown.
# (read loop instead of mapfile for macOS bash 3.2 compatibility.)
refs=()
while IFS= read -r line; do
  [[ -n "$line" ]] && refs+=("$line")
done < <(grep -oE '\]\(figures/[^)]+\)' "$DEST_MD" | sed -E 's/^\]\(//; s/\)$//' | sort -u)

if [[ ${#refs[@]} -eq 0 ]]; then
  echo "  warn: no figure references found in $DEST_MD" >&2
else
  fig_count=0
  for rel in "${refs[@]}"; do
    src_fig="$SRC_REPO/$rel"           # ../figures/<path> == repo/figures/<path>
    dst_fig="$DEST_DIR/$rel"
    if [[ ! -f "$src_fig" ]]; then
      echo "  error: referenced figure missing in source: $src_fig" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$dst_fig")"
    [[ -L "$dst_fig" ]] && rm -f "$dst_fig"
    cp "$src_fig" "$dst_fig"
    fig_count=$((fig_count + 1))
  done
  echo "  png -> content/posts/$SLUG/figures/  ($fig_count referenced figures)"
fi

# --- draft notice -----------------------------------------------------------
if grep -qE '^draft:[[:space:]]*true' "$DEST_MD"; then
  echo "  warn: draft is still true after sync patch" >&2
fi

echo "synced 1 post from $SRC_REPO/blog"
echo "next: review 'git -C \"$SITE_ROOT\" status' / 'git diff', then commit."
