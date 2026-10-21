
#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_NEF="data/P3a_L273R/P3a_L273R.nef"
INPUT_NMRPIPE="data/P3a_L273R/nmrpipe_shifts.tab"
INPUT_CCPN="data/P3a_L273R/ccpn_resonances.txt"
INPUT_PRED="data/P3a_L273R/shiftx2.cs"

OUTPUT="output/test.txt"
LOG="output/test.log"
CONFIG="config/config_yaml_2.txt"
LOG_MATRIX_PICKLE="output/log_prob_matrix.csv"
# REQUIRES MANUAL ADDITION OF THIS OUTPUT IN THE CODE BECAUSE IT'S JUST DIAGNOSTIC

RUNS=100
SEEDS=(0 1 2 3 4 5 6 7 8 9)
RESULTS_DIR="baseline_outputs"

mkdir -p "$RESULTS_DIR" "$(dirname "$OUTPUT")"

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

run_snaps() {
    local input="$1"
    local shift_type="$2"

    # Prevent accidentally hashing stale files
    rm -f "$OUTPUT" "$LOG_MATRIX_PICKLE"

    uv run python/SNAPS.py \
        "$input" \
        "$INPUT_PRED" \
        "$OUTPUT" \
        -c "$CONFIG" \
        -l "$LOG" \
        --shift_type "$shift_type" \
        --pred_type shiftx2 \
        --out_type snaps

    # Ensure both files were generated
    [[ -f "$OUTPUT" ]] || {
        echo "ERROR: Missing output: $OUTPUT" >&2
        exit 1
    }

    [[ -f "$LOG_MATRIX_PICKLE" ]] || {
        echo "ERROR: Missing matrix: $LOG_MATRIX_PICKLE" >&2
        exit 1
    }
}

get_hash() {
    md5sum "$OUTPUT" | cut -d ' ' -f 1
}

get_matrix_hash() {
    md5sum "$LOG_MATRIX_PICKLE" | cut -d ' ' -f 1
}

save_result() {
    local dir="$1"
    local hash matrix_hash

    hash=$(get_hash)
    matrix_hash=$(get_matrix_hash)

    echo "$hash" >> "$dir/hashes.txt"
    echo "$matrix_hash" >> "$dir/matrix_hashes.txt"
    echo "$hash $matrix_hash" >> "$dir/hash_pairs.txt"

    # Save one copy of each unique output and matrix
    if [[ ! -f "$dir/${hash}.txt" ]]; then
        cp "$OUTPUT" "$dir/${hash}.txt"
    fi

    if [[ ! -f "$dir/${matrix_hash}.csv" ]]; then
        cp "$LOG_MATRIX_PICKLE" "$dir/${matrix_hash}.csv"
    fi

    RESULT_HASH="$hash"
    RESULT_MATRIX_HASH="$matrix_hash"
}

prepare_dir() {
    local dir="$1"

    mkdir -p "$dir"

    : > "$dir/hashes.txt"
    : > "$dir/matrix_hashes.txt"
    : > "$dir/hash_pairs.txt"
}

print_summary() {
    local dir="$1"

    echo
    echo "Unique output hashes:"
    sort "$dir/hashes.txt" | uniq -c | sort -nr

    echo
    echo "Unique matrix hashes:"
    sort "$dir/matrix_hashes.txt" | uniq -c | sort -nr

    echo
    echo "Unique matrix/output combinations:"
    sort -u "$dir/hash_pairs.txt" | wc -l

    echo
}

# --------------------------------------------------
# Test: Different PYTHONHASHSEED values
# --------------------------------------------------

test_seeds() {
    local input="$1"
    local shift_type="$2"
    local name="$3"

    local dir="$RESULTS_DIR/$name/seeds"

    prepare_dir "$dir"

    echo "========================================"
    echo "Different seeds: $name"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        PYTHONHASHSEED="$seed" \
            run_snaps "$input" "$shift_type"

        save_result "$dir"

        echo "Seed $seed:"
        echo "  Output: $RESULT_HASH"
        echo "  Matrix: $RESULT_MATRIX_HASH"

        rg DR "$OUTPUT" || true
    done

    print_summary "$dir"
}

# --------------------------------------------------
# Test: Repeated runs
# --------------------------------------------------

test_repeated() {
    local input="$1"
    local shift_type="$2"
    local name="$3"
    local seed_mode="$4"

    local dir="$RESULTS_DIR/$name/$seed_mode"

    prepare_dir "$dir"

    echo "========================================"
    echo "$name: $RUNS runs ($seed_mode)"
    echo "========================================"

    for ((i = 1; i <= RUNS; i++)); do

        if [[ "$seed_mode" == "seed_0" ]]; then
            PYTHONHASHSEED=0 \
                run_snaps "$input" "$shift_type"
        else
            # Randomized Python hash seed
            (
                unset PYTHONHASHSEED
                run_snaps "$input" "$shift_type"
            )
        fi

        save_result "$dir"

        # Only show progress when running in a terminal
        if [[ -t 1 ]]; then
            printf "\r\033[KRun %3d/%d: %s | Matrix: %s" \
                "$i" "$RUNS" \
                "$RESULT_HASH" "$RESULT_MATRIX_HASH"
        fi
    done

    if [[ -t 1 ]]; then
        echo
    fi

    print_summary "$dir"
}

# --------------------------------------------------
# NEF tests
# --------------------------------------------------

test_seeds \
    "$INPUT_NEF" nef nef

test_repeated \
    "$INPUT_NEF" nef nef random

test_repeated \
    "$INPUT_NEF" nef nef seed_0

# --------------------------------------------------
# NMRPipe tests
# --------------------------------------------------

test_seeds \
    "$INPUT_NMRPIPE" nmrpipe nmrpipe

test_repeated \
    "$INPUT_NMRPIPE" nmrpipe nmrpipe random

test_repeated \
    "$INPUT_NMRPIPE" nmrpipe nmrpipe seed_0

# --------------------------------------------------
# CCPN tests
# --------------------------------------------------

test_seeds \
    "$INPUT_CCPN" ccpn ccpn

test_repeated \
    "$INPUT_CCPN" ccpn ccpn random

test_repeated \
    "$INPUT_CCPN" ccpn ccpn seed_0

# --------------------------------------------------
# Final summary
# --------------------------------------------------

echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"

printf "%-25s %10s %10s %10s\n" \
    "Test" "Outputs" "Matrices" "Runs"

echo "------------------------------------------------------------"

for name in nef nmrpipe ccpn; do
    for mode in seeds random seed_0; do

        dir="$RESULTS_DIR/$name/$mode"

        outputs=$(sort -u "$dir/hashes.txt" | wc -l)
        matrices=$(sort -u "$dir/matrix_hashes.txt" | wc -l)
        total=$(wc -l < "$dir/hash_pairs.txt")

        printf "%-25s %10d %10d %10d\n" \
            "$name/$mode" "$outputs" "$matrices" "$total"
    done
done

echo
echo "========================================"
echo "INTERPRETATION"
echo "========================================"

echo "1 matrix, 1 output:"
echo "  Deterministic results."
echo
echo "1 matrix, multiple outputs:"
echo "  Nondeterminism occurs after matrix generation."
echo
echo "Multiple matrices, multiple outputs:"
echo "  Nondeterminism occurs at or before matrix generation."
echo
echo "Multiple matrices, 1 output:"
echo "  Different serialized matrices produce the same output."
echo
echo "NOTE: Matrix hashes compare pickle bytes, not necessarily"
echo "the underlying numerical values."
