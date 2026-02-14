#!/bin/sh
if [ -z "$husky_skip_init" ]; then
  husky_skip_init=1
  export husky_skip_init
  command_exists () {
    command -v "$1" >/dev/null 2>&1
  }
  if command_exists pnpm; then
    :
  elif command_exists npm; then
    :
  else
    echo "husky > can't find pnpm or npm" >&2
    exit 1
  fi
  . "$(dirname "$0")/husky.local.sh" 2>/dev/null || true
fi
