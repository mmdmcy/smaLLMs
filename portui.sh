#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/.portui-runtime/portui.sh" --manifest-dir "$SCRIPT_DIR/portui" "$@"