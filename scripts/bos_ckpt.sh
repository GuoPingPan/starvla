#!/usr/bin/env bash
# ============================================================================
# bos_ckpt.sh — 上传/下载 model checkpoint 到对象存储 (阿里云 OSS / 百度云 BOS 双云)
#
# 远端根路径 : ${CKPT_USER_ROOT}ckpts/   (默认: OSS 环境 oss://algo-public0/orion/,
#               BOS 环境 bos://umi-gripper/Orion/; 可用 CKPT_USER_ROOT / BOS_USER_ROOT 覆盖)
# 底层命令   : python scripts/cloud_io.py (oss2 / boto3, 凭证走环境变量或 /root/.ossenv)
#               —— 不再依赖 bcecmd CLI
#
# 云自动选择 : CLOUD 环境变量可强制 oss|bos; 默认 auto —— OSS 凭证可得
#               (OSS_ACCESS_KEY_ID 或 PAI DSW 凭证服务 localhost:7002) 优先 OSS,
#               否则用 BOS (BOS_ACCESS_KEY_ID)。
#
# 用法:
#   ./bos_ckpt.sh upload   <model_or_path> [timestamp] [ckpt] [选项]
#   ./bos_ckpt.sh download <model_or_uri_path> [timestamp] [ckpt] [选项]
#   ./bos_ckpt.sh list     [model_name]
#
# download 输入形式 (任选其一):
#   <model> <timestamp>                       例: run_umi_xxx_20000step 0805_105158
#   <model>                                   默认取最新时间戳 (mtime)
#   (oss|bos)://.../ckpts/<model>/<ts>[/...]  完整云路径, 自动解析 model + timestamp
# ckpt 默认 final (= 最后权重); 也可传 20000 / latest / final / steps_..._pytorch_model.pt 等。
#
# 参数:
#   model_path_or_name  本地模型目录(完整路径) 或 模型名(位于 LOCAL_ROOT 下)
#                       例: /root/data/model_outputs/run_umi_xxx/0724_042505
#                           run_umi_shenzhen_relative_6d_full_20000step
#   ckpt                可选, 默认 latest。支持:
#                       latest       -> 最高步数的 steps_<N>_pytorch_model.pt
#                       final        -> final_model/pytorch_model.pt
#                       <数字>        -> 如 20000 => steps_20000_pytorch_model.pt
#                       <完整/相对名>  -> 如 steps_20000_pytorch_model.pt
#                       <子串>        -> 如 steps_20000 / latest.ckpt (模糊匹配)
#
# 选项:
#   --timestamp TS      显式指定时间戳 (也可直接作位置参数 <model> <ts> [ckpt]); 默认最新 mtime
#   --as NAME           upload 时指定远端模型名 (默认 = 本地模型名)
#   --out DIR           download 时本地目标根目录 (默认 = ~/.models/<model_name>)
#   --store DIR         本地模型库根目录, download 默认存放处 (默认 ~/.models)
#   --local-root DIR    训练输出根目录, upload 数据源 (默认 /root/data/model_outputs)
#   --no-sidecar        不附带 config.yaml / dataset_statistics.json
#   --cloud oss|bos     强制指定云端 (默认 auto)
#   -n, --dry-run       只打印将执行的 cloud_io 命令, 不真正传输
#   -h, --help          显示帮助
#
# 远端布局 (嵌套, 以模型名为 key, 其下按时间戳分子目录; 训练后整目录上传):
#   <root>ckpts/<model>/<timestamp>/final_model/pytorch_model.pt
#   <root>ckpts/<model>/<timestamp>/checkpoints/steps_<N>_pytorch_model.pt
#   <root>ckpts/<model>/<timestamp>/config.yaml
#   <root>ckpts/<model>/<timestamp>/dataset_statistics.json
#   download / list 按时间戳分组; 多时间戳默认取最新 (mtime), 可用 --timestamp <ts> 指定。
#
# 例:
#   ./bos_ckpt.sh download run_umi_shenzhen_relative_6d_v2_forceclose_20000step 0805_105158
#   ./bos_ckpt.sh download run_umi_shenzhen_relative_6d_v2_forceclose_20000step 0805_105158 20000
#   ./bos_ckpt.sh download oss://algo-public0/orion/ckpts/run_umi_xxx/0805_105158
#   ./bos_ckpt.sh download run_umi_shenzhen_relative_6d_v2_forceclose_20000step   # 默认最新 ts + final
#   ./bos_ckpt.sh upload   /root/data/model_outputs/run_umi_xxx/0805_105158
#   ./bos_ckpt.sh list     run_umi_shenzhen_relative_6d_v2_forceclose_20000step
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_IO="$SCRIPT_DIR/cloud_io.py"
PY="${CLOUD_IO_PYTHON:-python3}"

LOCAL_ROOT="/root/data/model_outputs"     # 训练输出根目录 (upload 的数据源)
MODEL_STORE="${HOME}/.models"            # 本地模型库 (download 默认存放处)
SIDECARS=(config.yaml dataset_statistics.json)

# ---- 输出工具 ----
info() { printf '\033[1;34m>>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 云选择 (auto: OSS 凭证可得优先, 否则 BOS) ----
detect_cloud() {
  if [[ -n "${CLOUD:-}" ]]; then
    case "$CLOUD" in oss|bos) echo "$CLOUD"; return ;; *) die "CLOUD 必须是 oss 或 bos" ;; esac
  fi
  if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" ]] \
     || curl -sf -m 1 -o /dev/null "${OSS_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}" 2>/dev/null; then
    echo oss; return
  fi
  if [[ -n "${BOS_ACCESS_KEY_ID:-}" && -n "${BOS_SECRET_ACCESS_KEY:-}" ]]; then
    echo bos; return
  fi
  die "无任何云端凭证: 请 export OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET (或 BOS_*), 或写 /root/.ossenv"
}

# 凭证文件 (不入库) 自动 source; 两者可并存 (同时有 OSS/BOS 凭证时 auto 选 OSS)
for _envf in "${CLOUD_ENV_FILE:-/root/.ossenv}" /root/.bosenv; do
  # shellcheck disable=SC1090
  [[ -f "$_envf" ]] && . "$_envf"
done

CLOUD="$(detect_cloud)"

# 远端根: 显式覆盖 > BOS_USER_ROOT (兼容旧名) > env 桶+命名空间 (必填, 无默认)
case "$CLOUD" in
  oss)
    : "${OSS_BUCKET:?ERROR: OSS_BUCKET 未设置 (请 export 或写入 /root/.ossenv)}"
    : "${OSS_NAMESPACE:?ERROR: OSS_NAMESPACE 未设置 (请 export 或写入 /root/.ossenv)}"
    DEFAULT_USER_ROOT="oss://${OSS_BUCKET}/${OSS_NAMESPACE}/"
    ;;
  bos)
    : "${BOS_BUCKET:?ERROR: BOS_BUCKET 未设置 (请 export 或写入 /root/.bosenv)}"
    : "${BOS_NAMESPACE:?ERROR: BOS_NAMESPACE 未设置 (请 export 或写入 /root/.bosenv)}"
    DEFAULT_USER_ROOT="bos://${BOS_BUCKET}/${BOS_NAMESPACE}/"
    ;;
esac
USER_ROOT="${CKPT_USER_ROOT:-${BOS_USER_ROOT:-$DEFAULT_USER_ROOT}}"
case "$USER_ROOT" in
  *://*) ;;  # 已带 scheme (用户显式给的), 保持
  *) die "USER_ROOT 必须是 oss:// 或 bos:// 开头的 URI: $USER_ROOT" ;;
esac
ROOT="${USER_ROOT}ckpts/"

# ---- 云操作封装 (替代原 bcecmd) ----
# ls 输出格式 (cloud_io.py ls): 目录 "  PRE  <prefix>"; 文件 "  {size}  {mtime}  {key}"
cloud_ls()        { "$PY" "$CLOUD_IO" ls "$@"; }              # 非递归
cloud_ls_r()      { "$PY" "$CLOUD_IO" ls -r "$@"; }           # 递归
run_cp_download() { # 远端对象 -> 本地
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '\033[1;32m[dry-run]\033[0m %s %s get %s %s\n' "$PY" "$CLOUD_IO" "$1" "$2"
  else
    "$PY" "$CLOUD_IO" get "$1" "$2"
  fi
}
run_cp_upload() { # 本地文件 -> 远端目录 (key = <prefix>/<文件名>)
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '\033[1;32m[dry-run]\033[0m %s %s put %s %s\n' "$PY" "$CLOUD_IO" "$1" "$2"
  else
    "$PY" "$CLOUD_IO" put "$1" "$2"
  fi
}

# 是否为时间戳形态 (如 0805_105158 / 0724_042505), 用于位置参数分流
_is_ts() { [[ "$1" =~ ^[0-9]{4}_[0-9]{6}$ ]]; }

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
  RES_MODEL_DIR=""; RES_MODEL_NAME=""; RES_RUN=""
  if [[ "$arg" == */* || "$arg" == /* ]]; then
    # 当作路径
    local d="${arg%/}"
    if [[ -d "$d/checkpoints" || -d "$d/final_model" ]]; then
      # .../<model>/<timestamp>  -> 嵌套
      RES_MODEL_DIR="$d"; RES_RUN="$(basename "$d")"; RES_MODEL_NAME="$(basename "$(dirname "$d")")"
      return
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
  RES_MODEL_DIR="$base/$run"; RES_MODEL_NAME="$mname"; RES_RUN="$run"
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

# ============================================================================
# 远端解析 (cloud_io ls; 输出: "  {size}  {mtime}  {key}" / "  PRE  <prefix>")
#   注: cloud_io ls 列出的 key 是桶内路径 (不含 bucket), awk 里剥掉 <bucket>/ 前缀后再解析
# ============================================================================

# URI "oss://bucket/a/b/" -> "a/b/" (桶内路径前缀)
_cloud_pfx() { local p="${1#*://}"; printf '%s' "${p#*/}"; }

# 打印 ROOT 下顶层条目 (模型/独立ckpt), 相对 ROOT 的名字
list_remote_top() {
  cloud_ls "$ROOT" 2>/dev/null \
    | awk -v pfx="$(_cloud_pfx "$ROOT")" '{ key=$NF; sub(pfx, "", key); print key }' \
    | grep -E '^[A-Za-z0-9._-]+/?$' \
    | sed 's:/$::' | sort -u
}

# 递归列出 <model>/ 下全部 ckpt, 输出 "<run>\t<rel>\t<mtime>"
#   run  = 路径第一段 (时间戳); 无嵌套 (扁平) 时为空
#   rel  = run 内相对路径 (如 checkpoints/steps_X_pytorch_model.pt / final_model/pytorch_model.pt)
#   mtime= 云端修改时间 "YYYY-MM-DD HH:MM:SS", 用于挑最新 run
_remote_ckpts_raw() {
  local model="$1"
  local prefix="${ROOT}${model}/"
  cloud_ls_r "$prefix" 2>/dev/null | awk -v pfx="$(_cloud_pfx "$prefix")" '
    NF>=4 && $NF ~ /\.(pt|ckpt)$/ {
      key=$NF; sub(pfx, "", key);
      if (key ~ /\/$/) next;                 # 目录标记
      run=""; rel=key;
      n=index(key, "/");
      if (n>0) { run=substr(key,1,n-1); rel=substr(key,n+1); }
      print run "\t" rel "\t" $2 " " $3;
    }'
}

# 某 model 下的时间戳 run 列表, 按"最新 ckpt mtime"升序 (最新在最后); 仅含嵌套 run
list_remote_runs() {
  _remote_ckpts_raw "$1" | awk -F'\t' '
    $1 != "" { if (!($1 in mx) || $3 > mx[$1]) mx[$1]=$3 }
    END { for (r in mx) print mx[r] "\t" r }
  ' | sort -t$'\t' -k1,1 | awk -F'\t' '{print $2}'
}

# 某 model + run 下的 ckpt 相对路径
list_remote_ckpts() {
  local model="$1" run="${2:-}"
  _remote_ckpts_raw "$model" | awk -F'\t' -v want="$run" '$1==want {print $2}' | sort -u
}

# 选定远端 run: --timestamp 指定则校验; 否则取最新 (mtime); 无 run -> RES_RUN="" (调用方据此判 not-found)
resolve_remote_run() {
  local model="$1" ts="${2:-}"
  RES_RUN=""
  local runs; runs="$(list_remote_runs "$model")"
  [[ -n "$runs" ]] || return 0   # 无 run -> 视为未找到
  if [[ -n "$ts" ]]; then
    grep -qxF "$ts" <<<"$runs" || {
      warn "远端时间戳 $ts 不存在"
      info "可用时间戳:"; printf '    %s\n' "$runs" >&2
      die "时间戳不存在: $ts"
    }
    RES_RUN="$ts"
  else
    RES_RUN="$(tail -n1 <<<"$runs")"
    local n; n="$(grep -c . <<<"$runs")"
    [[ "$n" -gt 1 ]] && info "远端有 $n 个时间戳, 默认最新: $RES_RUN (--timestamp 指定其它)"
  fi
}

# ============================================================================
# 子命令
# ============================================================================

do_upload() {
  local arg="$1" ckpt="${2:-latest}" ts="${3:-}"
  local model_dir model_name run remote_name="${AS_NAME:-}"
  resolve_local_model "$arg" "$ts"
  model_dir="$RES_MODEL_DIR"; model_name="$RES_MODEL_NAME"; run="$RES_RUN"
  [[ -n "$remote_name" ]] || remote_name="$model_name"
  info "本地模型目录: $model_dir"
  info "时间戳      : $run"
  info "远端模型名  : $remote_name"

  local avail ck
  mapfile -t avail < <(list_local_ckpts "$model_dir")
  [[ ${#avail[@]} -gt 0 ]] || die "运行目录下没有 ckpt (checkpoints/ 为空或无 steps_*_pytorch_model.pt): $model_dir"
  resolve_ckpt "$ckpt" "${avail[@]}"
  ck="$RES_CKPT"
  info "选中 ckpt  : $ck"

  # ck -> 本地文件 & 模型目录内相对路径
  local rel_local local_pt
  if [[ "$ck" == "final" ]]; then
    rel_local="final_model/pytorch_model.pt"; local_pt="$model_dir/$rel_local"
  else
    rel_local="checkpoints/$ck"; local_pt="$model_dir/$rel_local"
  fi
  [[ -f "$local_pt" ]] || die "ckpt 文件不存在: $local_pt"

  # 远端对象路径: 嵌套, 镜像本地结构 (<model>/<run>/<rel_local>)
  [[ -n "$run" ]] || die "缺少时间戳: $model_dir 不是 <model>/<timestamp> 布局"
  local remote_prefix="$ROOT$remote_name/$run/"

  info "上传 $(basename "$local_pt")"
  run_cp_upload "$local_pt" "$remote_prefix$(dirname "$rel_local")/"
  if [[ "${NO_SIDECAR:-0}" != "1" ]]; then
    local s
    for s in "${SIDECARS[@]}"; do
      if [[ -f "$model_dir/$s" ]]; then
        info "附带 $s"
        run_cp_upload "$model_dir/$s" "$remote_prefix"
      fi
    done
  fi
  info "完成 -> $remote_prefix$rel_local"
}

do_download() {
  local remote_name="$1" ckpt="${2:-final}" ts="${3:-}"
  # 允许传入完整云路径 ((oss|bos)://.../ckpts/<model>[/<timestamp>[/...]]), 解析出 model + timestamp
  if [[ "$remote_name" == "$ROOT"* ]]; then
    local stripped="${remote_name#"$ROOT"}"
    remote_name="${stripped%%/*}"                       # model = 第一段
    local rest="${stripped#*/}"                         # model 之后
    if [[ "$rest" =~ ^([0-9]{4}_[0-9]{6}) ]]; then ts="${ts:-${BASH_REMATCH[1]}}"; fi
  fi
  remote_name="${remote_name%/}"

  resolve_remote_run "$remote_name" "$ts"
  local run="$RES_RUN"

  local avail ck
  mapfile -t avail < <(list_remote_ckpts "$remote_name" "$run")
  if [[ ${#avail[@]} -eq 0 ]]; then
    warn "云端未找到模型或其下无 .pt/.ckpt: $remote_name${run:+/$run}"
    info "$ROOT 下可用条目:"
    list_remote_top | sed 's/^/    /' >&2
    die "远端模型不存在或无 ckpt: $remote_name"
  fi
  resolve_ckpt "$ckpt" "${avail[@]}"
  ck="$RES_CKPT"
  info "远端模型: $remote_name${run:+ (时间戳 $run)}"
  info "选中 ckpt: $ck"

  local out="${OUT_DIR:-$MODEL_STORE/$remote_name}" prefix local_prefix remote_obj local_pt
  prefix="$ROOT$remote_name/"; local_prefix="$out/"
  [[ -n "$run" ]] && { prefix+="$run/"; local_prefix+="$run/"; }
  remote_obj="$prefix$ck"                     # ck 为 run 内相对路径
  local_pt="$local_prefix$ck"                 # 本地保留嵌套结构
  mkdir -p "$(dirname "$local_pt")"
  info "下载 -> $local_pt"
  run_cp_download "$remote_obj" "$local_pt"
  if [[ "${NO_SIDECAR:-0}" != "1" ]]; then
    local s
    for s in "${SIDECARS[@]}"; do
      if cloud_ls "${prefix}$s" 2>/dev/null | grep -q .; then
        info "附带 $s"
        mkdir -p "$(dirname "$local_prefix$s")"
        run_cp_download "${prefix}$s" "$local_prefix$s"
      fi
    done
  fi
  info "完成 -> $local_pt"
}

do_list() {
  local arg="${1:-}"
  if [[ -z "$arg" ]]; then
    info "云端 ($CLOUD) ckpt 根: $ROOT"
    info "训练输出 ($LOCAL_ROOT):";   list_local_models | sed 's/^/    /' >&2
    echo >&2
    info "本地模型库 ($MODEL_STORE):"; list_store_models | sed 's/^/    /' >&2
    echo >&2
    info "云端顶层条目 ($ROOT):";  list_remote_top  | sed 's/^/    /' >&2
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
  # 远端: 按时间戳分组
  local runs r
  runs="$(list_remote_runs "$arg")"
  if [[ -n "$runs" ]]; then
    while read -r r; do
      info "云端模型 $arg / 时间戳 $r:"
      list_remote_ckpts "$arg" "$r" | sed 's/^/      /' >&2
    done <<<"$runs"
  else
    warn "云端无此模型或其下无 ckpt: $arg"
    info "$ROOT 顶层条目:"
    list_remote_top | sed 's/^/    /' >&2
  fi
}

usage() { sed -n '3,/^# =\{10,\}$/p' "$0" | sed 's/^# \?//'; exit 0; }

# ============================================================================
# 参数解析
# ============================================================================
MODE="" MODEL="" CKPT="" TIMESTAMP=""
AS_NAME="" OUT_DIR="" DRY_RUN=0 NO_SIDECAR=0
POS=()

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
    --cloud) CLOUD="${2:?--cloud 需要参数}"; shift ;;
    --*) die "未知选项: $1" ;;
    *) POS+=("$1") ;;
  esac
  shift
done

# 位置参数分类: [0]=model; 其余按形态判 时间戳(0805_105158) / ckpt(final|20000|latest|...)
[[ ${#POS[@]} -ge 1 ]] && MODEL="${POS[0]}"
for a in "${POS[@]:1}"; do
  if _is_ts "$a"; then
    [[ -z "$TIMESTAMP" ]] && TIMESTAMP="$a" || die "多余时间戳参数: $a"
  elif [[ -z "$CKPT" ]]; then CKPT="$a"
  else die "多余参数: $a"; fi
done

[[ -n "$MODE" ]] || { usage; }
[[ -f "$CLOUD_IO" ]] || die "未找到 $CLOUD_IO"
_SDK_MOD="oss2"; [[ "$CLOUD" == "bos" ]] && _SDK_MOD="boto3"
"$PY" -c "import $_SDK_MOD" 2>/dev/null \
  || die "python ($PY) 缺少 $CLOUD SDK ($_SDK_MOD), 请: pip install $_SDK_MOD"

case "$MODE" in
  upload)
    [[ -n "$MODEL" ]] || die "缺少模型参数 (upload <model_or_path> [timestamp] [ckpt])"
    do_upload "$MODEL" "${CKPT:-latest}" "$TIMESTAMP"
    ;;
  download)
    [[ -n "$MODEL" ]] || die "缺少模型参数 (download <model_or_uri_path> [timestamp] [ckpt])"
    do_download "$MODEL" "${CKPT:-final}" "$TIMESTAMP"
    ;;
  list)
    do_list "$MODEL"
    ;;
esac
