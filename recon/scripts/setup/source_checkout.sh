#!/usr/bin/env bash
galatea_checkout_source() {
  local url="$1" dest="$2" revision="$3" assets="${4:-}" stage backup
  if [ -e "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD)" = "$revision" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")" || return
  stage="$(mktemp -d "${dest}.install.XXXXXX")" || return
  if ! (git init -q "$stage" && git -C "$stage" remote add origin "$url" &&
        git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 -C "$stage" fetch --depth 1 origin "$revision" &&
        GIT_LFS_SKIP_SMUDGE=1 git -C "$stage" checkout --detach FETCH_HEAD); then
    rm -rf "$stage"
    echo "Could not fetch pinned source $url. Existing sources were preserved; check network access and rerun." >&2
    return 1
  fi
  if [ -e "$dest" ] || [ -L "$dest" ]; then
    if [ -n "$assets" ] && [ -d "$dest/$assets" ]; then
      cp -a --reflink=auto "$dest/$assets/." "$stage/$assets" || { rm -rf "$stage"; return 1; }
    fi
    backup="$(mktemp -d "${dest}.backup.XXXXXX")" || { rm -rf "$stage"; return 1; }
    mv "$dest" "$backup/source" || { rm -rf "$stage"; return 1; }
    if ! mv "$stage" "$dest"; then
      mv "$backup/source" "$dest"
      return 1
    fi
    echo "Updated $dest to $revision; previous sources and local changes saved in $backup/source"
  else
    mv "$stage" "$dest" || return
  fi
}
