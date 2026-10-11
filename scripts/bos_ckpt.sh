#!/usr/bin/env bash
# ============================================================================
# bos_ckpt.sh — 上传/下载 model checkpoint 到 BOS
#
# BOS 根路径 : ${BOS_USER_ROOT}ckpts/   (BOS_USER_ROOT 默认 bos://umi-gripper/Orion/, 可用环境变量覆盖)
# 底层命令   : bcecmd bos cp <SRC> <DST> -y   (注: bcecmd v0.5.10 无 -p 参数,
#               本脚本用 -y 自动确认, 并手动拼出完整路径, 等价于 -p/--parents 的效果)
#
# 用法:
#   ./bos_ckpt.sh upload   <model_path_or_name> [ckpt] [选项]
#   ./bos_ckpt.sh download <model_name>         [ckpt] [选项]
#   ./bos_ckpt.sh list     [model_name]
#
# 参数:
#   model_path_or_name  本地模型目录(完整路径) 或 模型名(位于 LOCAL_ROOT 下)
#                       例: /root/data/model_outputs/run_umi_xxx/0724_042505
#                           run_umi_shenzhen_relative_6d_full_20000step
#   ckpt                可选, 默认 latest。支持:
#                       latest       -> 最高步数的 steps_<N>_pytorch_model.pt
#                       final        -> final_model/pytorch_model.pt (远端 final_pytorch_model.pt)
#                       <数字>        -> 如 20000 => steps_20000_pytorch_model.pt
#                       <完整/相对名>  -> 如 steps_20000_pytorch_model.pt
#                       <子串>        -> 如 steps_20000 / latest.ckpt (模糊匹配)
#
# 选项:
#   --timestamp TS      当模型名下有多个时间戳子目录时, 指定使用哪一个 (默认最新且有 ckpt 的)
#   --as NAME           upload 时指定远端模型名 (默认 = 本地模型名)
#   --out DIR           download 时本地目标根目录 (默认 = ~/.models/<model_name>)
#   --store DIR         本地模型库根目录, download 默认存放处 (默认 ~/.models)
#   --local-root DIR    训练输出根目录, upload 数据源 (默认 /root/data/model_outputs)
#   --no-sidecar        不附带 config.yaml / dataset_statistics.json
#   -n, --dry-run       只打印将执行的 bcecmd 命令, 不真正传输
#   -h, --help          显示帮助
#
# 远端布局 (扁平, 以模型名为 key):
#   bos://umi-gripper/Orion/ckpts/<model_name>/steps_<N>_pytorch_model.pt
#   bos://umi-gripper/Orion/ckpts/<model_name>/final_pytorch_model.pt
#   bos://umi-gripper/Orion/ckpts/<model_name>/config.yaml            (可选附带)
#   bos://umi-gripper/Orion/ckpts/<model_name>/dataset_statistics.json (可选附带)
#   注: download 会递归列出 <model_name>/ 下的 .pt/.ckpt, 因此也兼容历史嵌套布局
#       (如 <model_name>/checkpoints/latest.ckpt)。
#
# 例:
#   ./bos_ckpt.sh upload run_umi_shenzhen_relative_6d_full_20000step
#   ./bos_ckpt.sh upload /root/data/model_outputs/run_umi_xxx/0724_042505 final
#   ./bos_ckpt.sh download run_umi_shenzhen_relative_6d_full_20000step 20000
#   ./bos_ckpt.sh download umi_0529_ckpts latest.ckpt
#   ./bos_ckpt.sh list run_umi_shenzhen_relative_6d_full_20000step
# ============================================================================

set -uo pipefail

BOS_USER_ROOT="${BOS_USER_ROOT:-bos://umi-gripper/Orion/}"   # 用户命名空间根 (其下 ckpts/ src/ ...); 环境变量可覆盖
BOS_ROOT="${BOS_USER_ROOT}ckpts/"
LOCAL_ROOT="/root/data/model_outputs"     # 训练输出根目录 (upload 的数据源)
MODEL_STORE="${HOME}/.models"            # 本地模型库 (download 默认存放处)
BCE="bcecmd"
SIDECARS=(config.yaml dataset_statistics.json)

# ---- 输出工具 ----
info() { printf '\033[1;34m>>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# 用 bcecmd 执行 (或 dry-run 时仅打印)
run_bce() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '\033[1;32m[dry-run]\033[0m %s %s\n' "$BCE" "$*"
  else
    "$BCE" "$@"
  fi
}

# ============================================================================
# 本地解析
# ============================================================================

# 打印某根目录下的模型名
list_dir_models() {
  find "$1" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort
}
list_local_models() { list_dir_models "$LOCAL_ROOT"; }
list_store_models() { list_dir_models "$MODEL_STORE"; }

# 打印某模型目录(base 路径)下含 checkpoints/ 或 final_model/ 的运行子目录名
list_local_runs() {
  local base="$1" d
  [[ -d "$base" ]] || return 0
  for d in "$base"/*/; do
    [[ -d "$d" ]] || continue
    { [[ -d "${d}checkpoints" ]] || [[ -d "${d}final_model" ]]; } && basename "$d"
  done | sort
}

# 打印某 model_dir 下的 ckpt 显示名 (steps_*.pt 文件名 + final)
list_local_ckpts() {
  local model_dir="$1" f
  if [[ -d "$model_dir/checkpoints" ]]; then
    for f in "$model_dir/checkpoints"/steps_*_pytorch_model.pt; do
      [[ -e "$f" ]] && basename "$f"
    done
  fi
  [[ -f "$model_dir/final_model/pytorch_model.pt" ]] && echo "final"
}

# 解析本地模型: 设置全局 RES_MODEL_DIR / RES_MODEL_NAME
# (用全局变量而非 stdout, 这样内部的 die 能终止整个脚本, 不被子 shell 吞掉)
resolve_local_model() {
  local arg="$1" ts="${2:-}" base
  RES_MODEL_DIR=""; RES_MODEL_NAME=""
  if [[ "$arg" == */* || "$arg" == /* ]]; then
    # 当作路径
    local d="${arg%/}"
    if [[ -d "$d/checkpoints" || -d "$d/final_model" ]]; then
      RES_MODEL_DIR="$d"; RES_MODEL_NAME="$(basename "$(dirname "$d")")"; return
    elif [[ -d "$d" ]]; then
      # 当作模型名目录, 进入多时间戳逻辑
      _pick_run "$d" "$(basename "$d")" "$ts"; return
    else
      die "路径不存在: $arg"
    fi
  else
    # 当作模型名: 先在 LOCAL_ROOT 找, 再在 MODEL_STORE 找
    local found=""
    if [[ -d "$LOCAL_ROOT/$arg" ]]; then found="$LOCAL_ROOT/$arg"
    elif [[ -d "$MODEL_STORE/$arg" ]]; then found="$MODEL_STORE/$arg"; info "从本地模型库读取: $found"
    fi
    if [[ -z "$found" ]]; then
      warn "本地未找到模型: $arg"
      info "训练输出 ($LOCAL_ROOT):";   list_local_models | sed 's/^/    /' >&2
      info "本地模型库 ($MODEL_STORE):"; list_store_models | sed 's/^/    /' >&2
      die "模型不存在: $arg"
    fi
    base="$found"
    # 若 base 自身就是运行目录(含 checkpoints)
    if [[ -d "$base/checkpoints" || -d "$base/final_model" ]]; then
      RES_MODEL_DIR="$base"; RES_MODEL_NAME="$arg"; return
    fi
    _pick_run "$base" "$arg" "$ts"
  fi
}

# 从模型名目录里挑一个运行子目录; 设置 RES_MODEL_DIR / RES_MODEL_NAME
# - 默认挑"最新且有 ckpt 的"运行; 显式 --timestamp 时用指定的
_pick_run() {
  local base="$1" mname="$2" ts="${3:-}"
  local all_runs run candidates=""
  all_runs="$(list_local_runs "$base")"
  [[ -n "$all_runs" ]] || die "模型 $mname 下没有 checkpoints/ 或 final_model/ 子目录 ($base)"
  # 只保留真正含 ckpt 的运行
  while read -r run; do
    [[ -n "$(list_local_ckpts "$base/$run")" ]] && candidates+="${run}"$'\n'
  done <<<"$all_runs"
  candidates="${candidates%$'\n'}"   # 去掉末尾换行, 否则 tail -n1 会取到空行
  if [[ -z "$candidates" ]]; then
    warn "模型 $mname 的所有运行目录都没有 ckpt"
    info "运行目录:"; printf '    %s\n' $all_runs >&2
    die "没有可用 ckpt (checkpoints/ 为空): $base"
  fi
  if [[ -n "$ts" ]]; then
    run="$ts"
    grep -qxF "$run" <<<"$all_runs" || {
      warn "时间戳 $ts 不存在"
      info "可用时间戳:"; printf '    %s\n' $all_runs >&2
      die "时间戳不存在: $ts"
    }
  else
    run="$(tail -n1 <<<"$candidates")"
    local n; n="$(grep -c . <<<"$candidates")"
    [[ "$n" -gt 1 ]] && info "模型 $mname 有 $n 个含 ckpt 的时间戳, 默认最新: $run (--timestamp 指定其它)"
  fi
  RES_MODEL_DIR="$base/$run"; RES_MODEL_NAME="$mname"
}

# ============================================================================
# ckpt 名解析
# ============================================================================

# 从一堆 ckpt 名(可能是文件名或相对路径)里选出步数最大的 steps_*.pt;
# 没有 steps_ 则回退 final / final_pytorch_model.pt / latest.ckpt / pytorch_model.pt
latest_ckpt() {
  local best=-1 best_rel="" rel n b
  for rel in "$@"; do
    b="$(basename "$rel")"
    case "$b" in steps_*_pytorch_model.pt) ;; *) continue ;; esac
    n="${b#steps_}"; n="${n%_pytorch_model.pt}"   # 取出中间的步数
    case "$n" in ''|*[!0-9]*) continue ;; esac    # 非纯数字则跳过
    (( n > best )) && { best=$n; best_rel="$rel"; }
  done
  if [[ -z "$best_rel" ]]; then
    for rel in "$@"; do
      b="$(basename "$rel")"
      case "$b" in
        final|final_pytorch_model.pt|latest.ckpt|pytorch_model.pt) best_rel="$rel"; break ;;
      esac
    done
  fi
  echo "$best_rel"
}

# 设置全局 RES_CKPT; 失败 die
resolve_ckpt() {
  local arg="$1"; shift
  local avail=("$@")
  RES_CKPT=""
  [[ ${#avail[@]} -gt 0 ]] || die "没有可用的 ckpt"
  case "$arg" in
    ""|latest)
      RES_CKPT="$(latest_ckpt "${avail[@]}")"
      [[ -n "$RES_CKPT" ]] || { warn "没有 steps_*_pytorch_model.pt, 无法判断 latest"; _ckpts_not_found "$arg" "${avail[@]}"; }
      ;;
    final)
      local c b found=0
      for c in "${avail[@]}"; do
        b="$(basename "$c")"
        case "$b" in final|final_pytorch_model.pt|latest.ckpt|pytorch_model.pt) RES_CKPT="$c"; found=1; break ;; esac
      done
      [[ "$found" == 1 ]] || _ckpts_not_found "$arg" "${avail[@]}"
      ;;
    *)
      # 纯数字 -> steps_<N>_pytorch_model.pt
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        local want="steps_${arg}_pytorch_model.pt" c
        for c in "${avail[@]}"; do [[ "$c" == "$want" || "$(basename "$c")" == "$want" ]] && { RES_CKPT="$c"; return; } done
      fi
      # 完整名/相对路径精确匹配, 否则子串模糊匹配
      local c hit=""
      for c in "${avail[@]}"; do [[ "$c" == "$arg" ]] && { RES_CKPT="$c"; return; } done
      for c in "${avail[@]}"; do [[ "$c" == *"$arg"* ]] && { hit="$c"; break; } done
      [[ -n "$hit" ]] && { RES_CKPT="$hit"; return; }
      _ckpts_not_found "$arg" "${avail[@]}"
      ;;
  esac
}

_ckpts_not_found() {
  local arg="$1"; shift
  warn "未找到 ckpt: '$arg'"
  info "可用 ckpt:"
  printf '    %s\n' "$@" >&2
  die "ckpt 不存在: $arg"
}

# 显示名 -> 远端对象名 (仅 upload 用, 扁平布局)
ckpt_remote_name() {
  case "$1" in
    final) echo "final_pytorch_model.pt" ;;
    *)     echo "$1" ;;
  esac
}

# ============================================================================
# 远端解析
# ============================================================================

# 打印 BOS_ROOT 下顶层条目 (模型/独立ckpt)
list_remote_top() {
  $BCE bos ls "$BOS_ROOT" 2>/dev/null \
    | awk 'NF{print $NF}' \
    | grep -E '^[A-Za-z0-9._-]+/?$' \
    | sed 's:/$::' | sort -u
}

# 递归打印某 model_name 下的 ckpt 相对路径 (.pt / .ckpt), 排除目录标记与小文件
list_remote_ckpts() {
  local model="$1"
  $BCE bos ls -r "$BOS_ROOT$model/" 2>/dev/null \
    | awk 'NF{print $NF}' \
    | grep -E '\.(pt|ckpt)$' \
    | sed 's#^/##' \
    | grep -v '/$' \
    | sort -u
}

# ============================================================================
# 子命令
# ============================================================================

do_upload() {
  local arg="$1" ckpt="${2:-latest}" ts="${TIMESTAMP:-}"
  local model_dir model_name remote_name="${AS_NAME:-}"
  resolve_local_model "$arg" "$ts"
  model_dir="$RES_MODEL_DIR"; model_name="$RES_MODEL_NAME"
  [[ -n "$remote_name" ]] || remote_name="$model_name"
  info "本地模型目录: $model_dir"
  info "远端模型名  : $remote_name"

  local avail ck
  mapfile -t avail < <(list_local_ckpts "$model_dir")
  [[ ${#avail[@]} -gt 0 ]] || die "运行目录下没有 ckpt (checkpoints/ 为空或无 steps_*_pytorch_model.pt): $model_dir"
  resolve_ckpt "$ckpt" "${avail[@]}"
  ck="$RES_CKPT"
  info "选中 ckpt  : $ck"

  local local_pt remote_obj
  remote_obj="$(ckpt_remote_name "$ck")"
  if [[ "$ck" == "final" ]]; then
    local_pt="$model_dir/final_model/pytorch_model.pt"
  else
    local_pt="$model_dir/checkpoints/$ck"
  fi
  [[ -f "$local_pt" ]] || die "ckpt 文件不存在: $local_pt"

  info "上传 $(basename "$local_pt")"
  run_bce bos cp "$local_pt" "$BOS_ROOT$remote_name/$remote_obj" -y
  if [[ "${NO_SIDECAR:-0}" != "1" ]]; then
    local s
    for s in "${SIDECARS[@]}"; do
      if [[ -f "$model_dir/$s" ]]; then
        info "附带 $s"
        run_bce bos cp "$model_dir/$s" "$BOS_ROOT$remote_name/$s" -y
      fi
    done
  fi
  info "完成 -> $BOS_ROOT$remote_name/$remote_obj"
}

do_download() {
  local remote_name="$1" ckpt="${2:-latest}"
  # 允许传入完整 bos 路径, 取 BOS_ROOT 之后的第一段作为模型名
  if [[ "$remote_name" == "$BOS_ROOT"* ]]; then
    remote_name="${remote_name#"$BOS_ROOT"}"
    remote_name="${remote_name%%/*}"
  fi
  remote_name="${remote_name%/}"

  local avail ck
  mapfile -t avail < <(list_remote_ckpts "$remote_name")
  if [[ ${#avail[@]} -eq 0 ]]; then
    warn "BOS 上未找到模型或其下无 .pt/.ckpt: $remote_name"
    info "$BOS_ROOT 下可用条目:"
    list_remote_top | sed 's/^/    /' >&2
    die "远端模型不存在或无 ckpt: $remote_name"
  fi
  resolve_ckpt "$ckpt" "${avail[@]}"
  ck="$RES_CKPT"
  info "远端模型: $remote_name"
  info "选中 ckpt: $ck"

  local out="${OUT_DIR:-$MODEL_STORE/$remote_name}" remote_obj local_pt
  remote_obj="$BOS_ROOT$remote_name/$ck"      # ck 是相对路径
  local_pt="$out/$ck"                          # 保留远端相对路径
  mkdir -p "$(dirname "$local_pt")"
  info "下载 -> $local_pt"
  run_bce bos cp "$remote_obj" "$local_pt" -y
  if [[ "${NO_SIDECAR:-0}" != "1" ]]; then
    local s
    for s in "${SIDECARS[@]}"; do
      if $BCE bos ls "$BOS_ROOT$remote_name/$s" >/dev/null 2>&1; then
        info "附带 $s"
        run_bce bos cp "$BOS_ROOT$remote_name/$s" "$out/$s" -y
      fi
    done
  fi
  info "完成 -> $local_pt"
}

do_list() {
  local arg="${1:-}"
  if [[ -z "$arg" ]]; then
    info "训练输出 ($LOCAL_ROOT):";   list_local_models | sed 's/^/    /' >&2
    echo >&2
    info "本地模型库 ($MODEL_STORE):"; list_store_models | sed 's/^/    /' >&2
    echo >&2
    info "BOS 顶层条目 ($BOS_ROOT):";  list_remote_top  | sed 's/^/    /' >&2
    return
  fi
  # 本地: 遍历 训练输出 / 模型库
  local root label d runs shown=0
  for root in "$LOCAL_ROOT" "$MODEL_STORE"; do
    [[ -d "$root/$arg" ]] || continue
    shown=1
    [[ "$root" == "$LOCAL_ROOT" ]] && label="本地训练输出" || label="本地模型库"
    info "$label $arg:"
    runs="$(list_local_runs "$root/$arg")"
    if [[ -n "$runs" ]]; then
      while read -r d; do
        info "  时间戳 $d:"
        list_local_ckpts "$root/$arg/$d" | sed 's/^/      /' >&2
      done <<<"$runs"
    else
      list_local_ckpts "$root/$arg" | sed 's/^/      /' >&2
    fi
    echo >&2
  done
  [[ "$shown" == 1 ]] || warn "本地无此模型: $arg"
  # 远端
  local rck
  rck="$(list_remote_ckpts "$arg")"
  if [[ -n "$rck" ]]; then
    info "BOS 模型 $arg:"
    printf '    %s\n' "$rck" >&2
  else
    warn "BOS 无此模型或其下无 ckpt: $arg (BOS 顶层条目:)"
    list_remote_top | sed 's/^/    /' >&2
  fi
}

usage() { sed -n '3,/^# =\{10,\}$/p' "$0" | sed 's/^# \?//'; exit 0; }

# ============================================================================
# 参数解析
# ============================================================================
MODE="" MODEL="" CKPT=""
TIMESTAMP="" AS_NAME="" OUT_DIR="" DRY_RUN=0 NO_SIDECAR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    upload|download|list) MODE="$1" ;;
    -h|--help) usage ;;
    -n|--dry-run) DRY_RUN=1 ;;
    --no-sidecar) NO_SIDECAR=1 ;;
    --timestamp) TIMESTAMP="${2:?--timestamp 需要参数}"; shift ;;
    --as) AS_NAME="${2:?--as 需要参数}"; shift ;;
    --out) OUT_DIR="${2:?--out 需要参数}"; shift ;;
    --local-root) LOCAL_ROOT="${2:?--local-root 需要参数}"; shift ;;
    --store) MODEL_STORE="${2:?--store 需要参数}"; MODEL_STORE="${MODEL_STORE/#\~/$HOME}"; shift ;;
    --*) die "未知选项: $1" ;;
    *)
      if [[ -z "$MODEL" ]]; then MODEL="$1"
      elif [[ -z "$CKPT" ]]; then CKPT="$1"
      else die "多余参数: $1"; fi
      ;;
  esac
  shift
done

[[ -n "$MODE" ]] || { usage; }
command -v "$BCE" >/dev/null 2>&1 || die "未找到 bcecmd, 请先安装/配置 BCE CLI"

case "$MODE" in
  upload)
    [[ -n "$MODEL" ]] || die "缺少模型参数 (upload <model_path_or_name> [ckpt])"
    do_upload "$MODEL" "${CKPT:-latest}"
    ;;
  download)
    [[ -n "$MODEL" ]] || die "缺少模型参数 (download <model_name> [ckpt])"
    do_download "$MODEL" "${CKPT:-latest}"
    ;;
  list)
    do_list "$MODEL"
    ;;
esac
