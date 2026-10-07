# Slurm 잡 안에서 pixi 환경으로 Python을 돌릴 때의 공통 세팅. sbatch 스크립트에서 `source`해서 쓴다.
#
#   cp ~/.claude/skills/pixi/templates/slurm_env.sh scripts/slurm_env.sh   # 프로젝트에 복사
#   # sbatch 안에서 (프로젝트 루트에서 제출하는 게 규칙이라 상대경로가 맞다):
#   source scripts/slurm_env.sh
#   pyrun train.py --lr 1e-3          # 간단한 경우: pixi run python ... (wrapper 프로세스가 끼어듦)
#   # 또는
#   pixi_activate                     # 시그널/멀티노드/torchrun: PATH에 환경을 직접 올림 (wrapper 없음)
#   python train.py ...
#
# 선택 변수 (source 전에 export):
#   PIXI_ENV=default      pixi 환경 이름
#   SCRATCH_CACHE=...     HF/torch/triton 캐시 위치 (기본 /scratch/$USER/cache)
#
# 사전 조건: 로그인 노드에서 미리 `pixi install --locked` 해둘 것. 잡 안에서는 설치/solve를 하지 않는다.

# --- 작업 디렉터리 ------------------------------------------------------------
# sbatch는 스크립트를 spool로 복사해 실행하므로 $0 기준 경로는 믿을 수 없다. 제출 위치 = 프로젝트 루트.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pixi.toml || -f pyproject.toml ]] || { echo "[slurm_env] pixi manifest 없음: $PWD (프로젝트 루트에서 sbatch 했나?)" >&2; exit 1; }
command -v pixi >/dev/null || export PATH="$HOME/.pixi/bin:$PATH"   # batch 셸엔 로그인 셸 PATH가 없을 수 있다

# --- Python 출력/진단 -------------------------------------------------------------
export PYTHONUNBUFFERED=1      # print가 즉시 로그 파일로 (없으면 블록 버퍼링돼서 로그가 한참 뒤에야 찍힘, 선점/kill 시 유실)
export PYTHONFAULTHANDLER=1    # segfault/SIGABRT 시 Python traceback 덤프
export PYTHONNOUSERSITE=1      # ~/.local/lib/pythonX 의 패키지가 환경을 오염시키는 것 방지 (재현성)
unset PYTHONHOME

# --- pixi 호출 방식 -----------------------------------------------------------------
# --as-is = --frozen + --no-install: lock을 건드리지 않고 환경도 설치/수정하지 않는다.
#   → 컴퓨트 노드 여러 곳이 NFS의 같은 .pixi/envs·pixi.lock을 동시에 만지는 사고 방지, 시작 지연도 최소.
# --color never / --no-progress: 로그 파일에 ANSI 코드와 진행바 찌꺼기가 안 남게.
PIXI_ENV="${PIXI_ENV:-default}"
export PIXI_COLOR=never
PIXI_RUN=(pixi run --as-is --color never --no-progress -e "$PIXI_ENV")

pyrun()  { "${PIXI_RUN[@]}" python "$@"; }   # pyrun script.py args...
pxrun()  { "${PIXI_RUN[@]}" "$@"; }          # pxrun torchrun ... / pxrun <task> / pxrun pytest

# `pixi run`은 중간에 wrapper 프로세스가 끼므로, 시그널(USR1/TERM trap)을 python에 직접 보내거나
# srun으로 task마다 python/torchrun을 띄울 땐 환경을 현재 셸에 활성화하고 바로 실행하는 쪽이 낫다.
pixi_activate() {
    eval "$(pixi shell-hook --as-is -s bash -e "$PIXI_ENV")"
}

# --- 스레드/캐시 ---------------------------------------------------------------------
# 안 정하면 BLAS/OpenMP가 노드 전체 코어 수만큼 스레드를 만들어 cgroup 한도에서 병목이 난다.
# (DDP 템플릿처럼 먼저 export한 값이 있으면 그걸 존중)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# 큰 캐시는 /home(NFS)이 아니라 /scratch로. triton/torch extension 캐시는 동시 잡이 같은 NFS 경로를 쓰면 깨지기도 한다.
if [[ -d /scratch ]]; then
    SCRATCH_CACHE="${SCRATCH_CACHE:-/scratch/$USER/cache}"
    export HF_HOME="${HF_HOME:-$SCRATCH_CACHE/huggingface}"
    export TORCH_HOME="${TORCH_HOME:-$SCRATCH_CACHE/torch}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_CACHE/triton/${SLURM_JOB_ID:-local}}"   # 잡별 분리: 동시 컴파일 충돌 방지
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$SCRATCH_CACHE/torch_extensions}"
    mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
fi

# --- 사전 점검: 환경이 안 깔려 있으면 여기서 바로 죽는다 --------------------------------
# (--as-is는 미설치 환경에서 "python: command not found"라는 알아보기 힘든 에러만 낸다)
if ! "${PIXI_RUN[@]}" python -c "import sys; print('[slurm_env] python:', sys.executable)" 2>/dev/null; then
    echo "[slurm_env] pixi 환경 '$PIXI_ENV'이 설치되지 않았거나 lock이 어긋남. 로그인 노드에서 먼저:  pixi install --locked -e $PIXI_ENV" >&2
    exit 1
fi
# torch가 있으면 GPU가 실제로 보이는지 로그에 남긴다 (GPU 잡에서 gres 누락/CPU 빌드 조기 발견)
"${PIXI_RUN[@]}" python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(f"[slurm_env] torch {torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} n={torch.cuda.device_count()}")
except ImportError:
    pass
PY
