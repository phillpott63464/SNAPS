
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

    uv run python/SNAPS.py \
        "$input" \
        "$INPUT_PRED" \
        "$OUTPUT" \
        -c "$CONFIG" \
        -l "$LOG" \
        --shift_type "$shift_type" \
        --pred_type shiftx2 \
        --out_type snaps
}

get_hash() {
    md5sum "$OUTPUT" | cut -d ' ' -f 1
}

print_summary() {
    local hash_file="$1"

    echo "Unique outputs:"
    sort "$hash_file" | uniq -c | sort -nr
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
    local hashes="$dir/hashes.txt"

    mkdir -p "$dir"
    : > "$hashes"

    echo "========================================"
    echo "Different seeds: $name"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        PYTHONHASHSEED="$seed" \
            run_snaps "$input" "$shift_type"

        hash=$(get_hash)
        echo "$hash" >> "$hashes"

        cp "$OUTPUT" "$dir/${hash}.txt"

        echo "Seed $seed: $hash"
        rg DR "$OUTPUT" || true
    done

    print_summary "$hashes"
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
    local hashes="$dir/hashes.txt"

    mkdir -p "$dir"
    : > "$hashes"

    echo "========================================"
    echo "$name: $RUNS runs ($seed_mode)"
    echo "========================================"

    for ((i = 1; i <= RUNS; i++)); do

        if [[ "$seed_mode" == "seed_0" ]]; then
            PYTHONHASHSEED=0 \
                run_snaps "$input" "$shift_type"
        else
            # Ensure Python uses a randomized hash seed
            (
                unset PYTHONHASHSEED
                run_snaps "$input" "$shift_type"
            )
        fi

        hash=$(get_hash)
        echo "$hash" >> "$hashes"

        cp "$OUTPUT" "$dir/${hash}.txt"

        if [[ -t 1 ]]; then
          printf "\r\033[KRun %3d/%d: %s" \
            "$i" "$RUNS" "$hash"
        fi
    done

    if [[ -t 1 ]]; then
      echo
    fi

    print_summary "$hashes"
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

printf "%-30s %10s %10s\n" \
    "Test" "Unique" "Total"

echo "----------------------------------------------------"

for name in nef nmrpipe ccpn; do
    for mode in seeds random seed_0; do

        hashes="$RESULTS_DIR/$name/$mode/hashes.txt"

        unique=$(sort -u "$hashes" | wc -l)
        total=$(wc -l < "$hashes")

        printf "%-30s %10d %10d\n" \
            "$name/$mode" "$unique" "$total"
    done
done
