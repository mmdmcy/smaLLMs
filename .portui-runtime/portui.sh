#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_MANIFEST_DIR="$SCRIPT_DIR/examples/demo"
DEFAULT_WORKSPACE_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_SELF_MANIFEST_DIR="$SCRIPT_DIR/.portui"

MANIFEST_DIR=""
WORKSPACE_DIR=""
MODE="auto"
LIST_ONLY=0
LIST_PROJECTS=0
RUN_ACTION_ID=""
PROJECT_ID=""
INSTALL_PROJECT_DIR=""
INIT_PROJECT_DIR=""

PORTUI_OS=""
PORTUI_VAR_KEYS=""
PORTUI_ACTION_LIST_FILE=""
PORTUI_PROJECT_LIST_FILE=""

PORTUI_MANIFEST_NAME=""
PORTUI_MANIFEST_DESCRIPTION=""
CURRENT_PROJECT_DIR=""
CURRENT_PROJECT_ID=""
CURRENT_WORKSPACE_DIR=""

usage() {
    cat <<'EOF'
Usage: sh ./portui.sh [--manifest-dir DIR | --workspace DIR] [--project PROJECT_ID] [--list-projects] [--list] [--run ACTION_ID]

PortUI opens project-local terminal menus from portui/ or .portui/ manifests.
It does not build executables; the launcher scripts are the portable entrypoints.

Options:
  --manifest-dir DIR   Path to one PortUI manifest directory.
  --workspace DIR      Path to a workspace containing project manifests in repo/portui or repo/.portui.
  --project ID         Project id inside workspace mode.
  --install-project DIR
                      Install or update project-local PortUI runtime files in a repo that already has portui/ or .portui.
  --init-project DIR  Create a starter PortUI app in a repo, then install the project-local runtime.
  --list-projects      Print discovered workspace projects and exit.
  --list               Print actions for the selected manifest or project and exit.
  --run ACTION_ID      Run a specific action non-interactively.
  --help               Show this help.
EOF
}

quote_single() {
    printf "%s" "$1" | sed "s/'/'\\\\''/g"
}

set_named_var() {
    key=$1
    value=$2
    case "$key" in
        ''|*[!A-Za-z0-9_]*)
            return 1
            ;;
    esac

    escaped=$(quote_single "$value")
    eval "PORTUI_VAR_$key='$escaped'"

    case " $PORTUI_VAR_KEYS " in
        *" $key "*) ;;
        *) PORTUI_VAR_KEYS="$PORTUI_VAR_KEYS $key" ;;
    esac
}

get_named_var() {
    key=$1
    case "$key" in
        ''|*[!A-Za-z0-9_]*)
            printf '%s' ""
            return
            ;;
    esac

    eval "printf '%s' \"\${PORTUI_VAR_$key-}\""
}

escape_sed_replacement() {
    printf "%s" "$1" | sed 's/[&|]/\\&/g'
}

expand_text() {
    text=$1
    pass=0

    while [ "$pass" -lt 8 ]; do
        changed=0
        for key in $PORTUI_VAR_KEYS; do
            token="{{$key}}"
            value=$(get_named_var "$key")
            safe_value=$(escape_sed_replacement "$value")
            updated=$(printf "%s" "$text" | sed "s|$token|$safe_value|g")
            if [ "$updated" != "$text" ]; then
                changed=1
                text=$updated
            fi
        done

        if [ "$changed" -eq 0 ]; then
            break
        fi

        pass=$((pass + 1))
    done

    printf "%s" "$text"
}

append_pipe_value() {
    var_name=$1
    next_value=$2
    eval "current_value=\${$var_name-}"
    if [ -n "$current_value" ]; then
        merged="$current_value|$next_value"
    else
        merged="$next_value"
    fi
    escaped=$(quote_single "$merged")
    eval "$var_name='$escaped'"
}

is_truthy() {
    value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$value" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

detect_os() {
    uname_value=$(uname -s 2>/dev/null || printf '%s' "unknown")
    case "$uname_value" in
        Linux) PORTUI_OS="linux" ;;
        Darwin) PORTUI_OS="macos" ;;
        *) PORTUI_OS="unknown" ;;
    esac
}

detect_local_manifest_dir() {
    if [ -f "$DEFAULT_SELF_MANIFEST_DIR/manifest.env" ]; then
        printf '%s\n' "$DEFAULT_SELF_MANIFEST_DIR"
        return 0
    fi
    if [ -f "$SCRIPT_DIR/portui/manifest.env" ]; then
        printf '%s\n' "$SCRIPT_DIR/portui"
        return 0
    fi
    return 1
}

resolve_dir() {
    target=$1
    if [ ! -d "$target" ]; then
        printf '%s\n' "Missing directory: $target" >&2
        exit 1
    fi
    CDPATH= cd -- "$target" && pwd
}

copy_runtime_file() {
    source_path=$1
    target_path=$2
    cp "$source_path" "$target_path" || exit 1
}

write_project_shim_sh() {
    target_path=$1
    manifest_leaf=$2

    cat > "$target_path" <<EOF
#!/bin/sh

set -eu

SCRIPT_DIR=\$(CDPATH= cd -- "\$(dirname -- "\$0")" && pwd)
exec sh "\$SCRIPT_DIR/.portui-runtime/portui.sh" --manifest-dir "\$SCRIPT_DIR/$manifest_leaf" "\$@"
EOF
    chmod +x "$target_path" || exit 1
}

write_project_shim_ps1() {
    target_path=$1
    manifest_leaf=$2

    cat > "$target_path" <<EOF
\$scriptDir = Split-Path -Parent \$MyInvocation.MyCommand.Path
& (Join-Path \$scriptDir '.portui-runtime\portui.ps1') -ManifestDir (Join-Path \$scriptDir '$manifest_leaf') @args
exit \$LASTEXITCODE
EOF
}

write_project_shim_cmd() {
    target_path=$1
    manifest_leaf=$2

    cat > "$target_path" <<EOF
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0\.portui-runtime\portui.ps1" -ManifestDir "%~dp0$manifest_leaf" %*
EOF
}

detect_project_manifest_dir_in_repo() {
    project_dir=$1
    if [ -f "$project_dir/portui/manifest.env" ]; then
        printf '%s\n' "$project_dir/portui"
        return 0
    fi
    if [ -f "$project_dir/.portui/manifest.env" ]; then
        printf '%s\n' "$project_dir/.portui"
        return 0
    fi

    printf '%s\n' "Project does not contain portui/manifest.env or .portui/manifest.env: $project_dir" >&2
    exit 1
}

install_project_runtime() {
    project_dir=$(resolve_dir "$1")
    manifest_dir=$(detect_project_manifest_dir_in_repo "$project_dir")
    manifest_leaf=$(basename "$manifest_dir")
    runtime_dir="$project_dir/.portui-runtime"

    mkdir -p "$runtime_dir" || exit 1

    copy_runtime_file "$SCRIPT_DIR/portui.sh" "$runtime_dir/portui.sh"
    copy_runtime_file "$SCRIPT_DIR/portui.ps1" "$runtime_dir/portui.ps1"
    copy_runtime_file "$SCRIPT_DIR/portui.cmd" "$runtime_dir/portui.cmd"
    if [ -f "$SCRIPT_DIR/VERSION" ]; then
        copy_runtime_file "$SCRIPT_DIR/VERSION" "$runtime_dir/VERSION"
    fi
    chmod +x "$runtime_dir/portui.sh" || exit 1

    write_project_shim_sh "$project_dir/portui.sh" "$manifest_leaf"
    write_project_shim_ps1 "$project_dir/portui.ps1" "$manifest_leaf"
    write_project_shim_cmd "$project_dir/portui.cmd" "$manifest_leaf"

    printf '%s\n' "Installed PortUI runtime into $project_dir"
    printf '%s\n' "Manifest: $manifest_dir"
    printf '%s\n' "Run from the project root with ./portui.sh, .\\portui.ps1, or portui.cmd"
}

init_project_files() {
    project_dir=$1
    project_name=$(basename "$project_dir")
    manifest_dir="$project_dir/portui"
    actions_dir="$manifest_dir/actions"

    if [ -f "$project_dir/portui/manifest.env" ] || [ -f "$project_dir/.portui/manifest.env" ]; then
        printf '%s\n' "Project already has a PortUI app definition: $project_dir" >&2
        exit 1
    fi

    mkdir -p "$actions_dir" || exit 1

    cat > "$manifest_dir/manifest.env" <<EOF
NAME=$project_name PortUI
DESCRIPTION=Starter PortUI app for $project_name.
VARIABLE_repo={{projectDir}}
EOF

    cat > "$actions_dir/01-doctor.env" <<'EOF'
ID=doctor
TITLE=Doctor
DESCRIPTION=Print the current project, workspace, and OS values.
TIMEOUT_SECONDS=20
CWD={{projectDir}}
POSIX_PROGRAM=sh
POSIX_ARGS=-c|printf '%s\n' 'project={{projectId}}' 'workspace={{workspaceDir}}' 'os={{os}}'
WINDOWS_PROGRAM=powershell
WINDOWS_ARGS=-NoProfile|-Command|Write-Output 'project={{projectId}}'; Write-Output 'workspace={{workspaceDir}}'; Write-Output 'os={{os}}'
EOF

    cat > "$actions_dir/02-list-files.env" <<'EOF'
ID=list-files
TITLE=List Files
DESCRIPTION=List the files in the project root.
TIMEOUT_SECONDS=20
CWD={{projectDir}}
POSIX_PROGRAM=ls
POSIX_ARGS=-la|.
WINDOWS_PROGRAM=powershell
WINDOWS_ARGS=-NoProfile|-Command|Get-ChildItem -Force .
EOF

    cat > "$actions_dir/03-git-status.env" <<'EOF'
ID=git-status
TITLE=Git Status
DESCRIPTION=Show a compact git status when the project is a git repository.
TIMEOUT_SECONDS=30
CWD={{projectDir}}
PROGRAM=git
ARGS=status|--short|--branch
EOF
}

init_project_runtime() {
    project_dir=$(resolve_dir "$1")
    init_project_files "$project_dir"
    install_project_runtime "$project_dir"
    printf '%s\n' "Created starter PortUI app in $project_dir/portui"
}

project_dir_from_manifest_dir() {
    manifest_dir=$1
    manifest_base=$(basename "$manifest_dir")
    case "$manifest_base" in
        portui|.portui)
            dirname "$manifest_dir"
            ;;
        *)
            printf '%s\n' "$manifest_dir"
            ;;
    esac
}

project_id_from_manifest_dir() {
    project_dir=$(project_dir_from_manifest_dir "$1")
    basename "$project_dir"
}

read_manifest_summary() {
    manifest_dir=$1
    manifest_file="$manifest_dir/manifest.env"
    SUMMARY_NAME=$(project_id_from_manifest_dir "$manifest_dir")
    SUMMARY_DESCRIPTION=""

    if [ ! -f "$manifest_file" ]; then
        return
    fi

    while IFS= read -r raw_line || [ -n "$raw_line" ]; do
        line=$(printf "%s" "$raw_line" | tr -d '\r')
        case "$line" in
            ''|'#'*)
                continue
                ;;
        esac

        key=${line%%=*}
        value=${line#*=}
        case "$key" in
            NAME)
                SUMMARY_NAME=$value
                ;;
            DESCRIPTION)
                SUMMARY_DESCRIPTION=$value
                ;;
        esac
    done < "$manifest_file"
}

reset_variable_state() {
    PORTUI_VAR_KEYS=""
}

init_builtin_variables() {
    home_dir=${HOME-}
    current_dir=$(pwd)
    current_project_dir=$(project_dir_from_manifest_dir "$MANIFEST_DIR")
    current_project_id=$(project_id_from_manifest_dir "$MANIFEST_DIR")

    if [ -n "$CURRENT_WORKSPACE_DIR" ]; then
        workspace_dir_value=$CURRENT_WORKSPACE_DIR
    else
        workspace_dir_value=$(dirname "$current_project_dir")
    fi

    CURRENT_PROJECT_DIR=$current_project_dir
    CURRENT_PROJECT_ID=$current_project_id

    set_named_var "home" "$home_dir" || exit 1
    set_named_var "cwd" "$current_dir" || exit 1
    set_named_var "os" "$PORTUI_OS" || exit 1
    set_named_var "manifestDir" "$MANIFEST_DIR" || exit 1
    set_named_var "projectDir" "$current_project_dir" || exit 1
    set_named_var "projectId" "$current_project_id" || exit 1
    set_named_var "workspaceDir" "$workspace_dir_value" || exit 1

    if [ "$PORTUI_OS" = "windows" ]; then
        set_named_var "pathSep" "\\" || exit 1
        set_named_var "listSep" ";" || exit 1
        set_named_var "exeSuffix" ".exe" || exit 1
    else
        set_named_var "pathSep" "/" || exit 1
        set_named_var "listSep" ":" || exit 1
        set_named_var "exeSuffix" "" || exit 1
    fi
}

parse_manifest_line() {
    line=$1
    case "$line" in
        ''|'#'*)
            return
            ;;
    esac

    key=${line%%=*}
    value=${line#*=}

    case "$key" in
        NAME) PORTUI_MANIFEST_NAME=$value ;;
        DESCRIPTION) PORTUI_MANIFEST_DESCRIPTION=$value ;;
        VARIABLE_*)
            variable_name=${key#VARIABLE_}
            set_named_var "$variable_name" "$value" || {
                printf '%s\n' "Invalid variable name in manifest: $variable_name" >&2
                exit 1
            }
            ;;
    esac
}

load_manifest() {
    manifest_file="$MANIFEST_DIR/manifest.env"
    if [ ! -f "$manifest_file" ]; then
        printf '%s\n' "Missing manifest file: $manifest_file" >&2
        exit 1
    fi

    PORTUI_MANIFEST_NAME=$CURRENT_PROJECT_ID
    PORTUI_MANIFEST_DESCRIPTION=""

    while IFS= read -r raw_line || [ -n "$raw_line" ]; do
        line=$(printf "%s" "$raw_line" | tr -d '\r')
        parse_manifest_line "$line"
    done < "$manifest_file"

    for key in $PORTUI_VAR_KEYS; do
        value=$(get_named_var "$key")
        expanded=$(expand_text "$value")
        set_named_var "$key" "$expanded" || exit 1
    done
}

build_action_list() {
    actions_dir="$MANIFEST_DIR/actions"
    if [ ! -d "$actions_dir" ]; then
        printf '%s\n' "Missing actions directory: $actions_dir" >&2
        exit 1
    fi

    if [ -n "${PORTUI_ACTION_LIST_FILE-}" ] && [ -f "$PORTUI_ACTION_LIST_FILE" ]; then
        rm -f "$PORTUI_ACTION_LIST_FILE"
    fi

    PORTUI_ACTION_LIST_FILE=$(mktemp)
    find "$actions_dir" -type f -name '*.env' | sort > "$PORTUI_ACTION_LIST_FILE"
}

build_project_list() {
    if [ ! -d "$WORKSPACE_DIR" ]; then
        printf '%s\n' "Missing workspace directory: $WORKSPACE_DIR" >&2
        exit 1
    fi

    if [ -n "${PORTUI_PROJECT_LIST_FILE-}" ] && [ -f "$PORTUI_PROJECT_LIST_FILE" ]; then
        rm -f "$PORTUI_PROJECT_LIST_FILE"
    fi

    PORTUI_PROJECT_LIST_FILE=$(mktemp)

    for candidate in "$WORKSPACE_DIR"/* "$WORKSPACE_DIR"/.[!.]* "$WORKSPACE_DIR"/..?*; do
        [ -d "$candidate" ] || continue

        for manifest_candidate in "$candidate/portui" "$candidate/.portui"; do
            manifest_file="$manifest_candidate/manifest.env"
            if [ -f "$manifest_file" ]; then
                resolved_manifest_dir=$(resolve_dir "$manifest_candidate")
                printf '%s\n' "$resolved_manifest_dir" >> "$PORTUI_PROJECT_LIST_FILE"
            fi
        done
    done

    sort -u "$PORTUI_PROJECT_LIST_FILE" -o "$PORTUI_PROJECT_LIST_FILE"
}

project_count() {
    count=0
    if [ -z "${PORTUI_PROJECT_LIST_FILE-}" ] || [ ! -f "$PORTUI_PROJECT_LIST_FILE" ]; then
        printf '%s' "0"
        return
    fi

    while IFS= read -r manifest_dir || [ -n "$manifest_dir" ]; do
        [ -n "$manifest_dir" ] || continue
        count=$((count + 1))
    done < "$PORTUI_PROJECT_LIST_FILE"

    printf '%s' "$count"
}

cleanup() {
    if [ -n "${PORTUI_ACTION_LIST_FILE-}" ] && [ -f "$PORTUI_ACTION_LIST_FILE" ]; then
        rm -f "$PORTUI_ACTION_LIST_FILE"
    fi
    if [ -n "${PORTUI_PROJECT_LIST_FILE-}" ] && [ -f "$PORTUI_PROJECT_LIST_FILE" ]; then
        rm -f "$PORTUI_PROJECT_LIST_FILE"
    fi
}

trap cleanup EXIT INT TERM

reset_action_state() {
    ACTION_ID=""
    ACTION_TITLE=""
    ACTION_DESCRIPTION=""
    ACTION_TIMEOUT_SECONDS="30"
    ACTION_INTERACTIVE="0"
    ACTION_PROGRAM=""
    ACTION_ARGS=""
    ACTION_CWD=""
    ACTION_ENV=""
    ACTION_POSIX_PROGRAM=""
    ACTION_POSIX_ARGS=""
    ACTION_POSIX_CWD=""
    ACTION_POSIX_ENV=""
    ACTION_LINUX_PROGRAM=""
    ACTION_LINUX_ARGS=""
    ACTION_LINUX_CWD=""
    ACTION_LINUX_ENV=""
    ACTION_MACOS_PROGRAM=""
    ACTION_MACOS_ARGS=""
    ACTION_MACOS_CWD=""
    ACTION_MACOS_ENV=""
    ACTION_WINDOWS_PROGRAM=""
    ACTION_WINDOWS_ARGS=""
    ACTION_WINDOWS_CWD=""
    ACTION_WINDOWS_ENV=""
}

parse_action_line() {
    line=$1
    case "$line" in
        ''|'#'*)
            return
            ;;
    esac

    key=${line%%=*}
    value=${line#*=}

    case "$key" in
        ID) ACTION_ID=$value ;;
        TITLE) ACTION_TITLE=$value ;;
        DESCRIPTION) ACTION_DESCRIPTION=$value ;;
        TIMEOUT_SECONDS) ACTION_TIMEOUT_SECONDS=$value ;;
        INTERACTIVE) ACTION_INTERACTIVE=$value ;;
        PROGRAM) ACTION_PROGRAM=$value ;;
        ARGS) ACTION_ARGS=$value ;;
        CWD) ACTION_CWD=$value ;;
        ENV_*) append_pipe_value "ACTION_ENV" "${key#ENV_}=$value" ;;
        POSIX_PROGRAM) ACTION_POSIX_PROGRAM=$value ;;
        POSIX_ARGS) ACTION_POSIX_ARGS=$value ;;
        POSIX_CWD) ACTION_POSIX_CWD=$value ;;
        POSIX_ENV_*) append_pipe_value "ACTION_POSIX_ENV" "${key#POSIX_ENV_}=$value" ;;
        LINUX_PROGRAM) ACTION_LINUX_PROGRAM=$value ;;
        LINUX_ARGS) ACTION_LINUX_ARGS=$value ;;
        LINUX_CWD) ACTION_LINUX_CWD=$value ;;
        LINUX_ENV_*) append_pipe_value "ACTION_LINUX_ENV" "${key#LINUX_ENV_}=$value" ;;
        MACOS_PROGRAM) ACTION_MACOS_PROGRAM=$value ;;
        MACOS_ARGS) ACTION_MACOS_ARGS=$value ;;
        MACOS_CWD) ACTION_MACOS_CWD=$value ;;
        MACOS_ENV_*) append_pipe_value "ACTION_MACOS_ENV" "${key#MACOS_ENV_}=$value" ;;
        WINDOWS_PROGRAM) ACTION_WINDOWS_PROGRAM=$value ;;
        WINDOWS_ARGS) ACTION_WINDOWS_ARGS=$value ;;
        WINDOWS_CWD) ACTION_WINDOWS_CWD=$value ;;
        WINDOWS_ENV_*) append_pipe_value "ACTION_WINDOWS_ENV" "${key#WINDOWS_ENV_}=$value" ;;
    esac
}

load_action() {
    action_file=$1
    reset_action_state

    while IFS= read -r raw_line || [ -n "$raw_line" ]; do
        line=$(printf "%s" "$raw_line" | tr -d '\r')
        parse_action_line "$line"
    done < "$action_file"

    if [ -z "$ACTION_ID" ]; then
        printf '%s\n' "Action file is missing ID: $action_file" >&2
        exit 1
    fi

    if [ -z "$ACTION_TITLE" ]; then
        ACTION_TITLE=$ACTION_ID
    fi
}

apply_variant() {
    prefix=$1
    label=$2

    eval "variant_program=\${ACTION_${prefix}_PROGRAM-}"
    eval "variant_args=\${ACTION_${prefix}_ARGS-}"
    eval "variant_cwd=\${ACTION_${prefix}_CWD-}"
    eval "variant_env=\${ACTION_${prefix}_ENV-}"

    if [ -n "$variant_program" ]; then
        RESOLVED_PROGRAM=$variant_program
    fi
    if [ -n "$variant_args" ]; then
        RESOLVED_ARGS=$variant_args
    fi
    if [ -n "$variant_cwd" ]; then
        RESOLVED_CWD=$variant_cwd
    fi
    if [ -n "$variant_env" ]; then
        if [ -n "$RESOLVED_ENV" ]; then
            RESOLVED_ENV="$RESOLVED_ENV|$variant_env"
        else
            RESOLVED_ENV=$variant_env
        fi
    fi

    if [ -n "$variant_program$variant_args$variant_cwd$variant_env" ]; then
        RESOLUTION_SOURCE="$RESOLUTION_SOURCE -> $label"
    fi
}

expand_env_pairs() {
    raw_pairs=$1
    expanded_pairs=""

    if [ -z "$raw_pairs" ]; then
        printf '%s' ""
        return
    fi

    old_ifs=$IFS
    IFS='|'
    set -- $raw_pairs
    IFS=$old_ifs

    for pair in "$@"; do
        env_key=${pair%%=*}
        env_value=${pair#*=}
        env_value=$(expand_text "$env_value")
        if [ -n "$expanded_pairs" ]; then
            expanded_pairs="$expanded_pairs|$env_key=$env_value"
        else
            expanded_pairs="$env_key=$env_value"
        fi
    done

    printf '%s' "$expanded_pairs"
}

resolve_action() {
    RESOLVED_PROGRAM=$ACTION_PROGRAM
    RESOLVED_ARGS=$ACTION_ARGS
    RESOLVED_CWD=$ACTION_CWD
    RESOLVED_ENV=$ACTION_ENV
    RESOLUTION_SOURCE="base"

    if [ "$PORTUI_OS" != "windows" ]; then
        apply_variant "POSIX" "posix"
    fi

    case "$PORTUI_OS" in
        linux) apply_variant "LINUX" "linux" ;;
        macos) apply_variant "MACOS" "macos" ;;
        windows) apply_variant "WINDOWS" "windows" ;;
    esac

    if [ -z "$RESOLVED_PROGRAM" ]; then
        printf '%s\n' "Action $ACTION_ID does not resolve to a runnable program on $PORTUI_OS" >&2
        exit 1
    fi

    RESOLVED_PROGRAM=$(expand_text "$RESOLVED_PROGRAM")
    RESOLVED_ARGS=$(expand_text "$RESOLVED_ARGS")

    if [ -n "$RESOLVED_CWD" ]; then
        RESOLVED_CWD=$(expand_text "$RESOLVED_CWD")
    else
        RESOLVED_CWD=$(pwd)
    fi

    RESOLVED_ENV=$(expand_env_pairs "$RESOLVED_ENV")
}

quote_display_part() {
    value=$1
    case "$value" in
        ''|*[!A-Za-z0-9_./:=+-]*)
            escaped=$(printf "%s" "$value" | sed 's/"/\\"/g')
            printf '"%s"' "$escaped"
            ;;
        *)
            printf '%s' "$value"
            ;;
    esac
}

display_command() {
    quote_display_part "$RESOLVED_PROGRAM"
    old_ifs=$IFS
    IFS='|'
    if [ -n "$RESOLVED_ARGS" ]; then
        set -- $RESOLVED_ARGS
    else
        set --
    fi
    IFS=$old_ifs

    for arg in "$@"; do
        printf ' '
        quote_display_part "$arg"
    done
}

run_resolved_action() {
    if is_truthy "$ACTION_INTERACTIVE"; then
        start_epoch=$(date +%s)

        (
            cd "$RESOLVED_CWD" || exit 1

            old_ifs=$IFS
            IFS='|'
            if [ -n "$RESOLVED_ENV" ]; then
                set -- $RESOLVED_ENV
            else
                set --
            fi
            IFS=$old_ifs

            for pair in "$@"; do
                env_key=${pair%%=*}
                env_value=${pair#*=}
                export "$env_key=$env_value"
            done
            export PORTUI_INTERACTIVE=1

            old_ifs=$IFS
            IFS='|'
            if [ -n "$RESOLVED_ARGS" ]; then
                set -- $RESOLVED_ARGS
            else
                set --
            fi
            IFS=$old_ifs

            exec "$RESOLVED_PROGRAM" "$@"
        )
        exit_code=$?

        end_epoch=$(date +%s)
        duration=$((end_epoch - start_epoch))
        printf '\n'
        printf '%s\n' "Status: exit code $exit_code"
        printf '%s\n' "Duration: ${duration}s"
        printf '\n'
        return "$exit_code"
    fi

    output_file=$(mktemp)
    timeout_flag_file=$(mktemp)
    start_epoch=$(date +%s)

    (
        cd "$RESOLVED_CWD" || exit 1

        old_ifs=$IFS
        IFS='|'
        if [ -n "$RESOLVED_ENV" ]; then
            set -- $RESOLVED_ENV
        else
            set --
        fi
        IFS=$old_ifs

        for pair in "$@"; do
            env_key=${pair%%=*}
            env_value=${pair#*=}
            export "$env_key=$env_value"
        done

        old_ifs=$IFS
        IFS='|'
        if [ -n "$RESOLVED_ARGS" ]; then
            set -- $RESOLVED_ARGS
        else
            set --
        fi
        IFS=$old_ifs

        exec "$RESOLVED_PROGRAM" "$@"
    ) >"$output_file" 2>&1 &
    command_pid=$!

    timed_out=0
    if [ -n "$ACTION_TIMEOUT_SECONDS" ] && [ "$ACTION_TIMEOUT_SECONDS" -gt 0 ] 2>/dev/null; then
        (
            sleep "$ACTION_TIMEOUT_SECONDS"
            if kill -0 "$command_pid" 2>/dev/null; then
                printf '%s' "1" > "$timeout_flag_file"
                kill "$command_pid" 2>/dev/null
                sleep 1
                kill -9 "$command_pid" 2>/dev/null
            fi
        ) >/dev/null 2>&1 </dev/null &
        watchdog_pid=$!
    else
        watchdog_pid=""
    fi

    wait "$command_pid"
    exit_code=$?

    if [ -n "$watchdog_pid" ]; then
        if [ -s "$timeout_flag_file" ]; then
            timed_out=1
        elif kill -0 "$watchdog_pid" 2>/dev/null; then
            kill "$watchdog_pid" 2>/dev/null
        fi
    fi

    end_epoch=$(date +%s)
    duration=$((end_epoch - start_epoch))

    printf '\n'
    if [ "$timed_out" -eq 1 ]; then
        printf '%s\n' "Status: timed out after ${ACTION_TIMEOUT_SECONDS}s"
    else
        printf '%s\n' "Status: exit code $exit_code"
    fi
    printf '%s\n' "Duration: ${duration}s"
    printf '\n'
    cat "$output_file"
    printf '\n'

    rm -f "$output_file" "$timeout_flag_file"

    if [ "$timed_out" -eq 1 ]; then
        return 124
    fi

    return "$exit_code"
}

load_manifest_context() {
    manifest_dir=$1
    workspace_dir=$2

    MANIFEST_DIR=$(resolve_dir "$manifest_dir")
    CURRENT_WORKSPACE_DIR=$workspace_dir
    reset_variable_state
    init_builtin_variables
    load_manifest
    build_action_list
}

list_actions() {
    count=0
    printf '%s\n' "$PORTUI_MANIFEST_NAME"
    if [ -n "$PORTUI_MANIFEST_DESCRIPTION" ]; then
        printf '%s\n' "$PORTUI_MANIFEST_DESCRIPTION"
    fi
    printf '%s\n' "Project: $CURRENT_PROJECT_ID"
    if [ -n "$CURRENT_WORKSPACE_DIR" ]; then
        printf '%s\n' "Workspace: $CURRENT_WORKSPACE_DIR"
    fi
    printf '\n'

    while IFS= read -r action_file || [ -n "$action_file" ]; do
        [ -n "$action_file" ] || continue
        load_action "$action_file"
        count=$((count + 1))
        printf '%2d. %s [%s]\n' "$count" "$ACTION_TITLE" "$ACTION_ID"
        if [ -n "$ACTION_DESCRIPTION" ]; then
            printf '    %s\n' "$ACTION_DESCRIPTION"
        fi
    done < "$PORTUI_ACTION_LIST_FILE"
}

list_projects() {
    count=0
    printf '%s\n' "PortUI Workspace"
    printf '%s\n' "$WORKSPACE_DIR"
    printf '\n'

    while IFS= read -r manifest_dir || [ -n "$manifest_dir" ]; do
        [ -n "$manifest_dir" ] || continue
        count=$((count + 1))
        project_id=$(project_id_from_manifest_dir "$manifest_dir")
        read_manifest_summary "$manifest_dir"
        printf '%2d. %s [%s]\n' "$count" "$SUMMARY_NAME" "$project_id"
        if [ -n "$SUMMARY_DESCRIPTION" ]; then
            printf '    %s\n' "$SUMMARY_DESCRIPTION"
        fi
    done < "$PORTUI_PROJECT_LIST_FILE"

    if [ "$count" -eq 0 ]; then
        printf '%s\n' "No PortUI projects found."
        exit 1
    fi
}

find_project_manifest_dir() {
    target_id=$1

    while IFS= read -r manifest_dir || [ -n "$manifest_dir" ]; do
        [ -n "$manifest_dir" ] || continue
        if [ "$(project_id_from_manifest_dir "$manifest_dir")" = "$target_id" ]; then
            printf '%s\n' "$manifest_dir"
            return 0
        fi
    done < "$PORTUI_PROJECT_LIST_FILE"

    return 1
}

run_action_by_id() {
    target_id=$1
    matched_file=""

    while IFS= read -r action_file || [ -n "$action_file" ]; do
        [ -n "$action_file" ] || continue
        load_action "$action_file"
        if [ "$ACTION_ID" = "$target_id" ]; then
            matched_file=$action_file
            break
        fi
    done < "$PORTUI_ACTION_LIST_FILE"

    if [ -z "$matched_file" ]; then
        printf '%s\n' "No action with id: $target_id" >&2
        exit 1
    fi

    load_action "$matched_file"
    resolve_action

    printf '%s\n' "$ACTION_TITLE"
    if [ -n "$ACTION_DESCRIPTION" ]; then
        printf '%s\n' "$ACTION_DESCRIPTION"
    fi
    printf '%s\n' "Project: $CURRENT_PROJECT_ID"
    printf '%s\n' "Working directory: $RESOLVED_CWD"
    printf '%s\n' "Resolution: $RESOLUTION_SOURCE"
    printf '%s' "Command: "
    display_command
    printf '\n'
    if is_truthy "$ACTION_INTERACTIVE"; then
        printf '%s\n' "I/O: interactive terminal"
    fi

    if [ -n "$RESOLVED_ENV" ]; then
        printf '%s\n' "Environment overrides:"
        old_ifs=$IFS
        IFS='|'
        set -- $RESOLVED_ENV
        IFS=$old_ifs
        for pair in "$@"; do
            printf '  %s\n' "$pair"
        done
    fi

    run_resolved_action
}

interactive_action_menu() {
    allow_back=$1

    while :; do
        printf '\n%s\n' "$PORTUI_MANIFEST_NAME"
        if [ -n "$PORTUI_MANIFEST_DESCRIPTION" ]; then
            printf '%s\n' "$PORTUI_MANIFEST_DESCRIPTION"
        fi
        printf '%s\n' "Project: $CURRENT_PROJECT_ID"
        printf '%s\n' "Project directory: $CURRENT_PROJECT_DIR"
        if [ -n "$CURRENT_WORKSPACE_DIR" ]; then
            printf '%s\n' "Workspace: $CURRENT_WORKSPACE_DIR"
        fi
        printf '%s\n\n' "OS: $PORTUI_OS"

        count=0
        while IFS= read -r action_file || [ -n "$action_file" ]; do
            [ -n "$action_file" ] || continue
            load_action "$action_file"
            count=$((count + 1))
            printf '%2d. %s [%s]\n' "$count" "$ACTION_TITLE" "$ACTION_ID"
            if [ -n "$ACTION_DESCRIPTION" ]; then
                printf '    %s\n' "$ACTION_DESCRIPTION"
            fi
        done < "$PORTUI_ACTION_LIST_FILE"

        if [ "$count" -eq 0 ]; then
            printf '%s\n' "No actions found."
            exit 1
        fi

        if [ "$allow_back" -eq 1 ]; then
            printf '\n%s' "Select an action number, b to go back, or q to quit: "
        else
            printf '\n%s' "Select an action number, or q to quit: "
        fi
        IFS= read -r selection

        case "$selection" in
            q|Q|quit|exit)
                exit 0
                ;;
            b|B|back)
                if [ "$allow_back" -eq 1 ]; then
                    return 0
                fi
                ;;
        esac

        case "$selection" in
            ''|*[!0-9]*)
                printf '%s\n' "Invalid selection."
                continue
                ;;
        esac

        if [ "$selection" -lt 1 ] || [ "$selection" -gt "$count" ]; then
            printf '%s\n' "Selection out of range."
            continue
        fi

        current_index=0
        selected_file=""
        while IFS= read -r action_file || [ -n "$action_file" ]; do
            [ -n "$action_file" ] || continue
            current_index=$((current_index + 1))
            if [ "$current_index" -eq "$selection" ]; then
                selected_file=$action_file
                break
            fi
        done < "$PORTUI_ACTION_LIST_FILE"

        if [ -z "$selected_file" ]; then
            printf '%s\n' "Unable to resolve selection."
            continue
        fi

        load_action "$selected_file"
        resolve_action

        printf '\n%s\n' "$ACTION_TITLE"
        if [ -n "$ACTION_DESCRIPTION" ]; then
            printf '%s\n' "$ACTION_DESCRIPTION"
        fi
        printf '%s\n' "Project: $CURRENT_PROJECT_ID"
        printf '%s\n' "Working directory: $RESOLVED_CWD"
        printf '%s\n' "Resolution: $RESOLUTION_SOURCE"
        printf '%s' "Command: "
        display_command
        printf '\n'
        if is_truthy "$ACTION_INTERACTIVE"; then
            printf '%s\n' "I/O: interactive terminal"
        fi

        if [ -n "$RESOLVED_ENV" ]; then
            printf '%s\n' "Environment overrides:"
            old_ifs=$IFS
            IFS='|'
            set -- $RESOLVED_ENV
            IFS=$old_ifs
            for pair in "$@"; do
                printf '  %s\n' "$pair"
            done
        fi

        printf '\n%s' "Run this action? [Y/n]: "
        IFS= read -r confirmation
        case "$confirmation" in
            n|N|no|NO)
                continue
                ;;
        esac

        run_resolved_action
        printf '\n%s' "Press Enter to return to the menu."
        IFS= read -r _
    done
}

interactive_workspace_menu() {
    while :; do
        printf '\n%s\n' "PortUI Workspace"
        printf '%s\n' "$WORKSPACE_DIR"
        printf '%s\n\n' "OS: $PORTUI_OS"

        count=0
        while IFS= read -r manifest_dir || [ -n "$manifest_dir" ]; do
            [ -n "$manifest_dir" ] || continue
            count=$((count + 1))
            project_id=$(project_id_from_manifest_dir "$manifest_dir")
            read_manifest_summary "$manifest_dir"
            printf '%2d. %s [%s]\n' "$count" "$SUMMARY_NAME" "$project_id"
            if [ -n "$SUMMARY_DESCRIPTION" ]; then
                printf '    %s\n' "$SUMMARY_DESCRIPTION"
            fi
        done < "$PORTUI_PROJECT_LIST_FILE"

        if [ "$count" -eq 0 ]; then
            printf '%s\n' "No PortUI projects found."
            exit 1
        fi

        printf '\n%s' "Select a project number, or q to quit: "
        IFS= read -r selection

        case "$selection" in
            q|Q|quit|exit)
                exit 0
                ;;
        esac

        case "$selection" in
            ''|*[!0-9]*)
                printf '%s\n' "Invalid selection."
                continue
                ;;
        esac

        if [ "$selection" -lt 1 ] || [ "$selection" -gt "$count" ]; then
            printf '%s\n' "Selection out of range."
            continue
        fi

        current_index=0
        selected_manifest=""
        while IFS= read -r manifest_dir || [ -n "$manifest_dir" ]; do
            [ -n "$manifest_dir" ] || continue
            current_index=$((current_index + 1))
            if [ "$current_index" -eq "$selection" ]; then
                selected_manifest=$manifest_dir
                break
            fi
        done < "$PORTUI_PROJECT_LIST_FILE"

        if [ -z "$selected_manifest" ]; then
            printf '%s\n' "Unable to resolve selection."
            continue
        fi

        load_manifest_context "$selected_manifest" "$WORKSPACE_DIR"
        interactive_action_menu 1
    done
}

select_mode() {
    detect_os

    if [ -n "$MANIFEST_DIR" ]; then
        MANIFEST_DIR=$(resolve_dir "$MANIFEST_DIR")
        MODE="manifest"
        return
    fi

    if [ -n "$WORKSPACE_DIR" ]; then
        WORKSPACE_DIR=$(resolve_dir "$WORKSPACE_DIR")
        build_project_list
        MODE="workspace"
        return
    fi

    local_manifest_dir=$(detect_local_manifest_dir 2>/dev/null || true)
    if [ -n "$local_manifest_dir" ]; then
        MANIFEST_DIR=$(resolve_dir "$local_manifest_dir")
        MODE="manifest"
        return
    fi

    WORKSPACE_DIR=$(resolve_dir "$DEFAULT_WORKSPACE_DIR")
    build_project_list
    discovered_projects=$(project_count)

    if [ "$discovered_projects" -gt 0 ]; then
        MODE="workspace"
        return
    fi

    if [ -n "$PROJECT_ID" ] || [ "$LIST_PROJECTS" -eq 1 ]; then
        printf '%s\n' "No PortUI workspace projects were discovered under $WORKSPACE_DIR" >&2
        exit 1
    fi

    MANIFEST_DIR=$(resolve_dir "$DEFAULT_MANIFEST_DIR")
    MODE="manifest"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --manifest-dir)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--manifest-dir requires a value" >&2
                exit 1
            fi
            MANIFEST_DIR=$1
            ;;
        --workspace)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--workspace requires a value" >&2
                exit 1
            fi
            WORKSPACE_DIR=$1
            ;;
        --project)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--project requires a value" >&2
                exit 1
            fi
            PROJECT_ID=$1
            ;;
        --install-project)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--install-project requires a value" >&2
                exit 1
            fi
            INSTALL_PROJECT_DIR=$1
            ;;
        --init-project)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--init-project requires a value" >&2
                exit 1
            fi
            INIT_PROJECT_DIR=$1
            ;;
        --list-projects)
            LIST_PROJECTS=1
            ;;
        --list)
            LIST_ONLY=1
            ;;
        --run)
            shift
            if [ "$#" -eq 0 ]; then
                printf '%s\n' "--run requires an action id" >&2
                exit 1
            fi
            RUN_ACTION_ID=$1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf '%s\n' "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ -n "$INIT_PROJECT_DIR" ]; then
    if [ -n "$INSTALL_PROJECT_DIR" ] || [ -n "$MANIFEST_DIR" ] || [ -n "$WORKSPACE_DIR" ] || [ -n "$PROJECT_ID" ] || [ "$LIST_PROJECTS" -eq 1 ] || [ "$LIST_ONLY" -eq 1 ] || [ -n "$RUN_ACTION_ID" ]; then
        printf '%s\n' "--init-project cannot be combined with other runtime selection or action flags" >&2
        exit 1
    fi

    init_project_runtime "$INIT_PROJECT_DIR"
    exit 0
fi

if [ -n "$INSTALL_PROJECT_DIR" ]; then
    if [ -n "$MANIFEST_DIR" ] || [ -n "$WORKSPACE_DIR" ] || [ -n "$PROJECT_ID" ] || [ "$LIST_PROJECTS" -eq 1 ] || [ "$LIST_ONLY" -eq 1 ] || [ -n "$RUN_ACTION_ID" ]; then
        printf '%s\n' "--install-project cannot be combined with runtime selection or action flags" >&2
        exit 1
    fi

    install_project_runtime "$INSTALL_PROJECT_DIR"
    exit 0
fi

if [ -n "$MANIFEST_DIR" ] && { [ -n "$WORKSPACE_DIR" ] || [ -n "$PROJECT_ID" ] || [ "$LIST_PROJECTS" -eq 1 ]; }; then
    printf '%s\n' "--manifest-dir cannot be combined with workspace options" >&2
    exit 1
fi

select_mode

if [ "$MODE" = "manifest" ]; then
    if [ "$LIST_PROJECTS" -eq 1 ] || [ -n "$PROJECT_ID" ]; then
        printf '%s\n' "Project selection is only available in workspace mode" >&2
        exit 1
    fi

    CURRENT_WORKSPACE_DIR=""
    load_manifest_context "$MANIFEST_DIR" "$CURRENT_WORKSPACE_DIR"

    if [ "$LIST_ONLY" -eq 1 ]; then
        list_actions
        exit 0
    fi

    if [ -n "$RUN_ACTION_ID" ]; then
        run_action_by_id "$RUN_ACTION_ID"
        exit $?
    fi

    interactive_action_menu 0
    exit 0
fi

workspace_projects=$(project_count)
if [ "$workspace_projects" -eq 0 ]; then
    printf '%s\n' "No PortUI projects found in workspace: $WORKSPACE_DIR" >&2
    exit 1
fi

if [ "$LIST_PROJECTS" -eq 1 ]; then
    list_projects
    exit 0
fi

if [ -n "$PROJECT_ID" ]; then
    selected_manifest=$(find_project_manifest_dir "$PROJECT_ID") || {
        printf '%s\n' "No project with id: $PROJECT_ID" >&2
        exit 1
    }

    load_manifest_context "$selected_manifest" "$WORKSPACE_DIR"

    if [ "$LIST_ONLY" -eq 1 ]; then
        list_actions
        exit 0
    fi

    if [ -n "$RUN_ACTION_ID" ]; then
        run_action_by_id "$RUN_ACTION_ID"
        exit $?
    fi

    interactive_action_menu 1
    exit 0
fi

if [ "$LIST_ONLY" -eq 1 ] || [ -n "$RUN_ACTION_ID" ]; then
    printf '%s\n' "Workspace mode requires --project when using --list or --run" >&2
    exit 1
fi

interactive_workspace_menu
