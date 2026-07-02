#!/usr/bin/env bash
#
# dir_size_tree.sh
#
# Logs a tree diagram of a directory, sized and SORTED with the heaviest
# directories/files first at every level. Sizing is done with `du`, which
# is disk-accurate (correctly dedupes hard-linked files) — unlike `tree
# --du`, which sums file sizes and can badly over-count when hard links
# are present (e.g. Docker layers, rsync --link-dest backups).
#
# Usage:
#   ./dir_size_tree.sh [TARGET_DIR] [MAX_DEPTH] [OUTPUT_FILE]
#
# Examples:
#   ./dir_size_tree.sh                     # scan current dir, unlimited depth, print to stdout
#   ./dir_size_tree.sh /var/log            # scan /var/log
#   ./dir_size_tree.sh /var/log 3          # limit recursion to 3 levels deep
#   ./dir_size_tree.sh /var/log 3 out.txt  # also save the report to out.txt
#
# Notes:
#   - Sizes are human-readable (K/M/G) as reported by `du`.
#   - At every directory level, entries are sorted largest-first.
#   - Recomputes `du` per directory to sort each level independently, so
#     very large/deep trees may take a while. Use MAX_DEPTH to limit scope.
#   - `tree` (if installed) is still shown afterward as a quick visual
#     reference, but its totals are NOT used for sizing — see note below.

set -uo pipefail

TARGET_DIR="${1:-.}"
MAX_DEPTH="${2:-}"
OUTPUT_FILE="${3:-}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: '$TARGET_DIR' is not a valid directory." >&2
    exit 1
fi

TARGET_DIR="$(realpath "$TARGET_DIR")"
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

# Recursively print a du-based tree, sorted largest-first at each level.
print_du_tree() {
    local dir="$1"
    local prefix="$2"
    local depth="$3"

    if [ -n "$MAX_DEPTH" ] && [ "$depth" -gt "$MAX_DEPTH" ]; then
        return
    fi

    local entries=()
    while IFS=$'\t' read -r size path; do
        [ "$path" = "$dir" ] && continue   # skip the du summary line for $dir itself
        entries+=("${size}"$'\t'"${path}")
    done < <(du -a -h --max-depth=1 "$dir" 2>/dev/null | sort -rh)

    local count=${#entries[@]}
    local i=0
    for entry in "${entries[@]}"; do
        i=$((i + 1))
        local size="${entry%%$'\t'*}"
        local path="${entry#*$'\t'}"
        local name
        name="$(basename "$path")"

        local connector="├──"
        local next_prefix="│   "
        if [ "$i" -eq "$count" ]; then
            connector="└──"
            next_prefix="    "
        fi

        if [ -d "$path" ]; then
            echo "${prefix}${connector} [${size}]  ${name}/"
            print_du_tree "$path" "${prefix}${next_prefix}" "$((depth + 1))"
        else
            echo "${prefix}${connector} [${size}]  ${name}"
        fi
    done
}

report() {
    echo "Directory size report (sorted, largest first)"
    echo "Target : $TARGET_DIR"
    echo "Date   : $TIMESTAMP"
    echo "========================================================"
    echo

    local total_size
    total_size="$(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)"
    echo "[${total_size}]  $(basename "$TARGET_DIR")/"
    print_du_tree "$TARGET_DIR" "" 1

    echo
    echo "========================================================"
    echo "Total size (du, accurate): $total_size"

    if command -v tree >/dev/null 2>&1; then
        echo
        echo "========================================================"
        echo "Visual reference only (tree --du) — NOT used for sizing."
        echo "Its totals can over-count if hard-linked files are present."
        echo "========================================================"
        echo
        if [ -n "$MAX_DEPTH" ]; then
            tree -a -h --du -L "$MAX_DEPTH" "$TARGET_DIR"
        else
            tree -a -h --du "$TARGET_DIR"
        fi
    fi
}

if [ -n "$OUTPUT_FILE" ]; then
    report | tee "$OUTPUT_FILE"
    echo
    echo "Report saved to: $OUTPUT_FILE"
else
    report
fi