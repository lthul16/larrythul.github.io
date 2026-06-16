#!/usr/bin/env bash
#
# sync-blog.sh — copy blog posts authored in the (private) ls-research repo into this
# Hugo site as real, committable files.
#
# Why this exists: the GitHub Pages CI builds on a clean runner that has NO access to
# ls-research, and Hugo will not follow a symlink that escapes the project root. So the
# publishable content must live here as real files. ls-research/blog is the source of
# truth; this site holds a published snapshot. Run this after editing a post in
# ls-research, then review `git diff` and commit.
#
# Usage:
#   scripts/sync-blog.sh            # sync all posts defined below
#   LS_RESEARCH_BLOG=/path scripts/sync-blog.sh   # override source location
#
set -euo pipefail

# --- locate repos -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_BLOG="${LS_RESEARCH_BLOG:-$SITE_ROOT/../ls-research/blog}"

if [[ ! -d "$SRC_BLOG" ]]; then
  echo "error: source blog dir not found: $SRC_BLOG" >&2
  echo "       set LS_RESEARCH_BLOG to the ls-research/blog path." >&2
  exit 1
fi
SRC_BLOG="$(cd "$SRC_BLOG" && pwd)"

# --- post registry ----------------------------------------------------------
# One line per post:  <source-markdown> | <source-figures-dir> | <dest-bundle-slug>
# The markdown is copied to <dest>/index.md; every *.png in the figures dir is copied
# flat into <dest>/ (the markdown references figures by bare filename).
# Content is not checked in on this branch; run this script when ready to publish.
POSTS=(
  "forecaster_architecture.md|figures|learned-alphabet-market-signals-tokenizing-time-series-for-llm-style-forecasting"
)

# --- sync -------------------------------------------------------------------
synced=0
for entry in "${POSTS[@]}"; do
  IFS='|' read -r src_md src_figs slug <<<"$entry"

  src_md_path="$SRC_BLOG/$src_md"
  src_figs_path="$SRC_BLOG/$src_figs"
  dest_dir="$SITE_ROOT/content/posts/$slug"

  if [[ ! -f "$src_md_path" ]]; then
    echo "error: source markdown not found: $src_md_path" >&2
    exit 1
  fi

  mkdir -p "$dest_dir"

  # Replace any existing symlink (the old, broken approach) with a real file.
  dest_md="$dest_dir/index.md"
  if [[ -L "$dest_md" ]]; then
    echo "  removing stale symlink: $dest_md"
    rm -f "$dest_md"
  fi

  cp "$src_md_path" "$dest_md"
  echo "  md  -> content/posts/$slug/index.md"

  if [[ -d "$src_figs_path" ]]; then
    fig_count=0
    shopt -s nullglob
    for png in "$src_figs_path"/*.png; do
      dest_png="$dest_dir/$(basename "$png")"
      # Replace stale symlinks with real files; copying onto a symlink would otherwise
      # follow it and clobber the source.
      [[ -L "$dest_png" ]] && rm -f "$dest_png"
      cp "$png" "$dest_png"
      fig_count=$((fig_count + 1))
    done
    shopt -u nullglob
    echo "  png -> content/posts/$slug/  ($fig_count figures)"
  else
    echo "  warn: figures dir not found, skipping images: $src_figs_path" >&2
  fi

  synced=$((synced + 1))
done

echo "synced $synced post(s) from $SRC_BLOG"
echo "next: review 'git -C \"$SITE_ROOT\" status' / 'git diff', then commit."
