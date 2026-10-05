#!/bin/bash
# bench/bench.sh - Runs the compiled V0/V1 executables located in build/, scans scales, and outputs CSV results.
#
# Prerequisites: Run the following commands in the project root directory first:
#   mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 && make -j
#
# Usage (run from the project root directory): bash bench/bench.sh

set -e
BUILD_DIR="build"
SIZES=(512 1024 2048 4096)
TILES=(8 16 32)
OUT_CSV="results/results.csv"

mkdir -p results

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: $BUILD_DIR not found; please run the CMake build first."
    exit 1
fi

echo "version,tile_size,N,avg_ms,gflops" > "$OUT_CSV"

echo "=== run V0 ==="
for N in "${SIZES[@]}"; do
    OUT=$("$BUILD_DIR/v0_naive" $N)
    echo "$OUT"
    MS=$(echo "$OUT" | grep -oP 'avg_time=\K[0-9.]+')
    GF=$(echo "$OUT" | grep -oP 'GFLOPS=\K[0-9.]+')
    echo "V0,NA,$N,$MS,$GF" >> "$OUT_CSV"
done

echo "=== run V1 (tile size comparison) ==="
for T in "${TILES[@]}"; do
    for N in "${SIZES[@]}"; do
        OUT=$("$BUILD_DIR/v1_tile${T}" $N)
        echo "$OUT"
        MS=$(echo "$OUT" | grep -oP 'avg_time=\K[0-9.]+')
        GF=$(echo "$OUT" | grep -oP 'GFLOPS=\K[0-9.]+')
        echo "V1,$T,$N,$MS,$GF" >> "$OUT_CSV"
    done
done

echo "=== run V2 (register blocking, TM/TN comparison) ==="
REG_TILES=(4x4 8x8)
for T in "${REG_TILES[@]}"; do
    for N in "${SIZES[@]}"; do
        OUT=$("$BUILD_DIR/v2_reg${T}" $N)
        echo "$OUT"
        MS=$(echo "$OUT" | grep -oP 'avg_time=\K[0-9.]+')
        GF=$(echo "$OUT" | grep -oP 'GFLOPS=\K[0-9.]+')
        echo "V2,$T,$N,$MS,$GF" >> "$OUT_CSV"
    done
done

echo "=== run V3 (vectorized + double buffering) ==="
for N in "${SIZES[@]}"; do
    OUT=$("$BUILD_DIR/v3_vecdbuf" $N)
    echo "$OUT"
    MS=$(echo "$OUT" | grep -oP 'avg_time=\K[0-9.]+')
    GF=$(echo "$OUT" | grep -oP 'GFLOPS=\K[0-9.]+')
    echo "V3,NA,$N,$MS,$GF" >> "$OUT_CSV"
done

echo "finish, results written to $OUT_CSV"