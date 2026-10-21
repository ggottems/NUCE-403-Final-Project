#!/bin/bash -l
#SBATCH --job-name=pebble_dem
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --threads-per-core=1
#SBATCH --core-spec=8
#SBATCH --time=00:20:00
#SBATCH --signal=INT@180
#SBATCH --output=%x-%j.out

# Benchmark: sbatch -A YOUR_PROJECT frontier_dem.sbatch
# Production: sbatch -A YOUR_PROJECT -p extended -t 24:00:00 --export=ALL,MODE=run,DEM_THREADS=16 frontier_dem.sbatch
# Continue:   sbatch -A YOUR_PROJECT -p extended -t 24:00:00 --export=ALL,MODE=resume,DEM_THREADS=16 frontier_dem.sbatch
# The account is supplied at submission rather than hardcoded here.
# Default case: 8 fixed release points (centre + 7 on a ring of radius 15 at z=65),
# each pebble pre-accelerated to vz=-3 and released every >=0.35 t* per point.
# Original single slow inlet: --export=ALL,DEM_INLET_COUNT=1,DEM_INLET_VZ=0,DEM_POUR_INTERVAL=0.5,...

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this file with sbatch from your Frontier working directory.}"

module load miniforge3
DEM_ENV="${DEM_ENV:-$SLURM_SUBMIT_DIR/.conda-dem-frontier}"
if [[ ! -x "$DEM_ENV/bin/python3" ]]; then
    echo "Missing Python environment: $DEM_ENV. See FRONTIER_DEM_RUN.md." >&2
    exit 2
fi
conda activate "$DEM_ENV"
set -u

DEM_SCRIPT="${DEM_SCRIPT:-$SLURM_SUBMIT_DIR/pebble_dem_optimized.py}"
if [[ ! -f "$DEM_SCRIPT" ]]; then
    echo "Missing solver: $DEM_SCRIPT. Copy it here or set DEM_SCRIPT at submission." >&2
    exit 2
fi

MODE="${MODE:-benchmark}"
DEM_N="${DEM_N:-53386}"
DEM_R="${DEM_R:-21}"
DEM_DP="${DEM_DP:-1}"
DEM_THREADS="${DEM_THREADS:-16}"
DEM_PREFIX="${DEM_PREFIX:-pb50k_8inlet}"
DEM_CHECKPOINT="${DEM_CHECKPOINT:-${DEM_PREFIX}_restart.npz}"
DEM_INLET_Z="${DEM_INLET_Z:-65}"
DEM_INLET_COUNT="${DEM_INLET_COUNT:-8}"     # Centre + (count-1) points on a ring.
DEM_INLET_RADIUS="${DEM_INLET_RADIUS:-15}"  # Ring radius, input length units.
DEM_INLET_VZ="${DEM_INLET_VZ:--3}"          # Release velocity; terminal speed is 1/drag = 3.33.
DEM_POUR_INTERVAL="${DEM_POUR_INTERVAL:-0.35}"
DEM_INLETS="${DEM_INLETS:-}"                # Optional explicit list "x y z; x y z; ..." (overrides the ring).
read -r -a DEM_EXTRA <<< "${DEM_EXTRA_ARGS:-}"  # Optional extra solver flags, e.g. "--max-settle-time 300".

# Reserve a single Python process with all 56 normally available physical
# cores. Numba controls its active thread count inside that process.
export NUMBA_NUM_THREADS=56
export NUMBA_THREADING_LAYER=omp
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

dem_launch() {
    srun --nodes=1 --ntasks=1 --cpus-per-task=56 \
        --threads-per-core=1 --cpu-bind=cores \
        "$DEM_ENV/bin/python3" -u "$DEM_SCRIPT" "$@"
}

valid_threads() {
    if [[ ! "$1" =~ ^[0-9]+$ ]] || ((10#$1 < 1 || 10#$1 > 56)); then
        echo "Thread count must be an integer from 1 to 56: $1" >&2
        exit 2
    fi
}

INLET_ARGS=()
if [[ -n "$DEM_INLETS" ]]; then
    IFS=';' read -r -a DEM_POINTS <<< "$DEM_INLETS"
    for DEM_POINT in "${DEM_POINTS[@]}"; do
        read -r -a DEM_XYZ <<< "$DEM_POINT"
        if ((${#DEM_XYZ[@]} != 3)); then
            echo "Each DEM_INLETS entry needs X Y Z: '$DEM_POINT'" >&2
            exit 2
        fi
        INLET_ARGS+=(--inlet "${DEM_XYZ[@]}")
    done
else
    if [[ ! "$DEM_INLET_COUNT" =~ ^[1-9][0-9]*$ ]]; then
        echo "DEM_INLET_COUNT must be a positive integer: $DEM_INLET_COUNT" >&2
        exit 2
    fi
    while read -r DEM_X DEM_Y; do
        INLET_ARGS+=(--inlet "$DEM_X" "$DEM_Y" "$DEM_INLET_Z")
    done < <(awk -v n="$DEM_INLET_COUNT" -v r="$DEM_INLET_RADIUS" 'BEGIN {
        print "0 0"; pi = atan2(0, -1)
        for (k = 0; k < n-1; k++) printf "%.6f %.6f\n", r*cos(2*pi*k/(n-1)), r*sin(2*pi*k/(n-1)) }')
fi
FEED_ARGS=(--inlet-vz "$DEM_INLET_VZ" --pour-interval "$DEM_POUR_INTERVAL" "${INLET_ARGS[@]}")

echo "Job=${SLURM_JOB_ID:-unknown}; mode=$MODE; directory=$SLURM_SUBMIT_DIR"
echo "Python: $DEM_ENV/bin/python3"
echo "One Python process; CPU threads; snapshots and checkpoints enabled in run mode."
echo "Feed: ${FEED_ARGS[*]}"

case "$MODE" in
    benchmark)
        DEM_BENCH_DIR="frontier_dem_benchmark_${SLURM_JOB_ID:-manual}"
        mkdir -p "$DEM_BENCH_DIR"
        read -r -a DEM_SWEEP <<< "${DEM_THREAD_COUNTS:-1 2 4 8 16 28 56}"
        if ((${#DEM_SWEEP[@]} == 0)); then
            echo "DEM_THREAD_COUNTS must contain at least one integer." >&2
            exit 2
        fi
        for DEM_T in "${DEM_SWEEP[@]}"; do
            valid_threads "$DEM_T"
            export OMP_NUM_THREADS="$DEM_T"
            echo "Benchmarking $DEM_T threads"
            dem_launch --benchmark --N "$DEM_N" --R "$DEM_R" --dp "$DEM_DP" \
                --backend numba --threads "$DEM_T" "${FEED_ARGS[@]}" "${DEM_EXTRA[@]}" \
                2>&1 | tee "$DEM_BENCH_DIR/threads_${DEM_T}.txt"
        done
        echo "Benchmark logs: $DEM_BENCH_DIR"
        grep -h "^ESTIMATE" "$DEM_BENCH_DIR"/threads_*.txt || true
        ;;
    run|resume)
        valid_threads "$DEM_THREADS"
        export OMP_NUM_THREADS="$DEM_THREADS"
        if [[ "$MODE" == run ]]; then
            if [[ -e "$DEM_CHECKPOINT" || -e "${DEM_PREFIX}.dat" ]]; then
                echo "Output/checkpoint already exists. Use MODE=resume or choose a new DEM_PREFIX and DEM_CHECKPOINT." >&2
                exit 2
            fi
            dem_launch --N "$DEM_N" --R "$DEM_R" --dp "$DEM_DP" \
                --backend numba --threads "$DEM_THREADS" "${FEED_ARGS[@]}" \
                --load-fraction 0.2 --clearance 1e-5 \
                --snapshots "${DEM_PREFIX}_snapshots" --snapshot-every 1000 \
                --checkpoint "$DEM_CHECKPOINT" --checkpoint-interval 300 \
                --history-interval 5 --log-file "${DEM_PREFIX}.log" -o "${DEM_PREFIX}.dat" "${DEM_EXTRA[@]}"
        else
            if [[ ! -f "$DEM_CHECKPOINT" ]]; then
                echo "Missing checkpoint: $DEM_CHECKPOINT" >&2
                exit 2
            fi
            dem_launch --resume "$DEM_CHECKPOINT" --backend numba --threads "$DEM_THREADS" \
                --checkpoint "$DEM_CHECKPOINT" --snapshots "${DEM_PREFIX}_snapshots" \
                --log-file "${DEM_PREFIX}.log" -o "${DEM_PREFIX}.dat" "${DEM_EXTRA[@]}"
        fi
        ;;
    *)
        echo "MODE must be benchmark, run, or resume." >&2
        exit 2
        ;;
esac
