#!/usr/bin/env bash
# ship_code_to_node.sh -- ship CODE (not weights/data) from the wzc1 canonical
# checkout to another node, then verify byte-level parity by sha256.
#
# WHY THIS EXISTS (user directive 2026-08-17):
#   "wzc1 是主要的盘 ... 模型可以留下但是代码最好每次训练都新传。不然可能有我们本地
#    更新的代码那边没 sync（或者你检查一下代码是否一致也可以）"
#   => wzc1 is the single source of truth. Other disks (zwfy6, wzz) are DERIVED
#      copies. Models/ckpts may live remotely forever (big, immutable); code must
#      be re-shipped before every training run, or PROVEN identical first.
#   Real precedent in this repo: the zwfy6 checkout is independent and has been
#   measured at commit 2d98c5a which is *not even an ancestor* of local HEAD.
#   "the remote should be up to date" is an assumption that has already burned
#   multiple agents. Hence: parity is verified, never assumed.
#
# ---------------------------------------------------------------------------
# DESIGN NOTE 1 -- why "carry everything under the code roots, then EXCLUDE",
#                  rather than "list what to include":
#   A hand-enumerated include list is guaranteed to go stale: the moment someone
#   adds src/memory/newthing/ or scripts/foo/bar.py, an include list silently
#   drops it and the remote runs old code. This repo has the scar: a freeze step
#   hand-passed 2 --evidence files while 24 were on disk. So we walk the code
#   roots wholesale and subtract, in four layers:
#     (L1) build/junk         __pycache__, *.pyc/*.pyo, *.egg-info, .git
#     (L2) weights/data/blobs *.pt *.bin *.safetensors *.npy ... (by extension)
#     (L3) secrets by NAME    password*, *credential*, *deploy_key*, *.pem, ...
#     (L4) secrets by CONTENT grep for private-key headers / quoted api_key=...
#          + `git check-ignore` as a backstop, so a NEWLY added
#          configs/password_<newnode>.txt is refused even before anyone updates
#          this script.
#   Failure mode of an include list = silently ship too little (stale remote).
#   Failure mode of an exclude list = ship something you did not mean to; L3+L4
#   are content-verified and printed, so that failure is loud, not silent.
#
# DESIGN NOTE 2 -- one tar stream, not per-file scp:
#   Measured wzc1<->wzz bandwidth is ~7.9 MB/s, and the payload is ~1.3k files
#   averaging ~9 KB. Per-file scp would pay ~1.3k round trips. We build one
#   gzipped tar and push it over a single ssh session.
#
# DESIGN NOTE 3 -- exit codes are BINARY. 0 = remote code == wzc1 code. Anything
#   non-zero is a failure that must be acted on. There is deliberately NO
#   "informational non-zero rc" tier: this repo has already been bitten by a
#   gate documented as "rc=1 is informational", which then printed real defects
#   for five rounds with nobody reconciling them.
#     0  in parity / ship+verify succeeded
#     1  parity differences (missing, mismatched, or extra remote code files)
#     2  usage error
#     3  transport / remote-side failure (ssh, tar, mkdir)
#     4  local precondition failure (not wzc1 source, unsafe filename, no payload)
#
# USAGE
#   scripts/ship_code_to_node.sh --target wzz25 [--check-only]
#   scripts/ship_code_to_node.sh --host 1.2.3.4 --remote-root /disk/proj \
#                                --password-file configs/password_x.txt
#   scripts/ship_code_to_node.sh --list          # payload manifest, no network
#
# OPTIONS
#   --target NAME        preset node (see PRESETS below)
#   --host H             explicit ssh host (overrides preset)
#   --remote-root PATH   explicit remote repo root (overrides preset)
#   --password-file F    sshpass -f file (overrides preset)
#   --check-only         do not transfer; only answer "is remote == wzc1?"
#   --list               print payload manifest + exclusion summary; no network
#   --prune-extra        delete remote code files absent from the wzc1 manifest
#   --allow-extra        do not count remote-only code files as a difference
#   --roots "a b c"      override code roots (default: see CODE_ROOTS)
#   --local-root PATH    override source root (default: this script's repo)
#   --allow-foreign-source  permit running from a non-wzc1 checkout (NOT advised)
#   -h|--help

set -uo pipefail

# ---------------------------------------------------------------------------
# Presets. Remote roots are DIFFERENT PER DISK and are never hardcoded to
# /apdcephfs_wzc1/... for a remote: on .73 that path is a symlink to zwfy6, and
# on .82 / the wzz pod it does not exist at all.
# ---------------------------------------------------------------------------
ZWFY6_ROOT='/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory'
WZZ_ROOT='/apdcephfs_wzz/share_303419932/pighzliu_code/Mixture-of-Memory'

preset_lookup() {
  # echoes "HOST<TAB>REMOTE_ROOT<TAB>PASSWORD_FILE"
  case "$1" in
    wzz25|25|.25)  printf '28.197.251.25\t%s\tconfigs/password_taiji_wzz25.txt\n' "$WZZ_ROOT" ;;
    73|.73)        printf '28.85.35.73\t%s\tconfigs/password_h20_853573.txt\n'    "$ZWFY6_ROOT" ;;
    82|.82)        printf '28.82.250.82\t%s\tconfigs/password_h20_82250.txt\n'    "$ZWFY6_ROOT" ;;
    104|.104)      printf '28.83.24.104\t%s\tconfigs/password_h20_24104.txt\n'    "$ZWFY6_ROOT" ;;
    *) return 1 ;;
  esac
}

# Code roots. Directories AND top-level files are both fine (find accepts both).
CODE_ROOTS_DEFAULT='src scripts configs monitor tests pyproject.toml requirements.txt'

die()  { printf 'FATAL: %s\n' "$*" >&2; exit "${2:-2}"; }
info() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
TARGET='' HOST='' RROOT='' PWFILE=''
MODE='ship'            # ship | check | list
PRUNE_EXTRA=0 ALLOW_EXTRA=0 ALLOW_FOREIGN=0
CODE_ROOTS="$CODE_ROOTS_DEFAULT"
LOCAL_ROOT=''

while [ $# -gt 0 ]; do
  case "$1" in
    --target)          TARGET="${2:-}"; shift 2 ;;
    --host)            HOST="${2:-}";   shift 2 ;;
    --remote-root)     RROOT="${2:-}";  shift 2 ;;
    --password-file)   PWFILE="${2:-}"; shift 2 ;;
    --check-only)      MODE='check';    shift ;;
    --list)            MODE='list';     shift ;;
    --prune-extra)     PRUNE_EXTRA=1;   shift ;;
    --allow-extra)     ALLOW_EXTRA=1;   shift ;;
    --roots)           CODE_ROOTS="${2:-}"; shift 2 ;;
    --local-root)      LOCAL_ROOT="${2:-}"; shift 2 ;;
    --allow-foreign-source) ALLOW_FOREIGN=1; shift ;;
    -h|--help)         sed -n '1,80p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# ---------------------------------------------------------------------------
# resolve source root; enforce "wzc1 is the source of truth"
# ---------------------------------------------------------------------------
if [ -z "$LOCAL_ROOT" ]; then
  _self_dir=$(cd -- "$(dirname -- "$0")" && pwd -P) || die "cannot resolve script dir" 4
  LOCAL_ROOT=$(cd -- "$_self_dir/.." && pwd -P)     || die "cannot resolve repo root" 4
else
  LOCAL_ROOT=$(cd -- "$LOCAL_ROOT" && pwd -P)       || die "bad --local-root" 4
fi
case "$LOCAL_ROOT" in
  /apdcephfs_wzc1/*) : ;;
  *) if [ "$ALLOW_FOREIGN" -eq 0 ]; then
       die "source root '$LOCAL_ROOT' is not on wzc1. wzc1 is the single source of truth; shipping FROM a derived copy would propagate stale code. Pass --allow-foreign-source only if you know why." 4
     fi
     info "WARNING: shipping from non-wzc1 source '$LOCAL_ROOT' (--allow-foreign-source)" ;;
esac
cd -- "$LOCAL_ROOT" || die "cannot cd to $LOCAL_ROOT" 4

if [ "$MODE" != 'list' ]; then
  if [ -n "$TARGET" ]; then
    _p=$(preset_lookup "$TARGET") || die "unknown --target '$TARGET' (known: wzz25 .73 .82 .104)"
    [ -n "$HOST" ]   || HOST=$(printf '%s' "$_p" | cut -f1)
    [ -n "$RROOT" ]  || RROOT=$(printf '%s' "$_p" | cut -f2)
    [ -n "$PWFILE" ] || PWFILE=$(printf '%s' "$_p" | cut -f3)
  fi
  [ -n "$HOST" ]  || die "need --target or --host"
  [ -n "$RROOT" ] || die "need --target or --remote-root"
  [ -n "$PWFILE" ] || die "need --target or --password-file"
  [ -f "$PWFILE" ] || die "password file not found: $PWFILE" 4
  case "$RROOT" in /*) : ;; *) die "--remote-root must be absolute, got '$RROOT'" ;; esac
fi

# ssh: NEVER pass -p. Global /etc/ssh/ssh_config sets Port 36000 for Host *;
# writing "-p 22" lands on a different sshd and looks exactly like an expired
# password. Verified with `ssh -G <host> | grep ^port`.
ssh_run() {
  sshpass -f "$PWFILE" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
    -o PreferredAuthentications=password -o LogLevel=ERROR "root@$HOST" "$@"
}

WORK=$(mktemp -d) || die "mktemp failed" 4
trap 'rm -rf -- "$WORK"' EXIT

# ---------------------------------------------------------------------------
# The find expression. ONE definition, used verbatim on BOTH sides, so local and
# remote enumerate the same universe. If these ever drift, parity becomes a lie.
# ---------------------------------------------------------------------------
#
# These lists are built programmatically (single-line output) because the find
# expression is `eval`ed locally AND shipped as a string to the remote shell:
# a hand-written multi-line string silently truncates at the first newline under
# eval, which yields a *successful-looking* find over a half-expression.
PRUNE_DIRS='__pycache__ .git .ship_code *.egg-info node_modules .ipynb_checkpoints .mypy_cache .pytest_cache .venv'
# L1 build junk / L2 weights+data blobs / L3 secret-shaped filenames
DENY_GLOBS='*.pyc *.pyo *.so *.o *.a *.swp *~ .DS_Store
 *.pt *.pth *.ckpt *.bin *.safetensors *.gguf *.msgpack
 *.npy *.npz *.h5 *.hdf5 *.pkl *.arrow *.parquet *.feather *.mmap *.idx *.model
 *.tar *.tgz *.gz *.zip *.bz2 *.xz *.7z *.zst
 password* *credential* *credentials* *deploy_key* *.pem *.key id_rsa* id_ed25519* *hf_token* .env'

build_find_cmd() {  # $1 = space-separated roots (already existence-filtered)
  local p d out
  p=''
  for d in $PRUNE_DIRS; do
    [ -z "$p" ] && p="-name '$d'" || p="$p -o -name '$d'"
  done
  out="find $1 \\( $p \\) -prune -o \\( -type f"
  for d in $DENY_GLOBS; do out="$out ! -name '$d'"; done
  printf '%s \\) -print0' "$out"
}

# existing local roots only (find errors on a missing root and poisons rc)
LROOTS=''
for r in $CODE_ROOTS; do
  if [ -e "$r" ]; then LROOTS="$LROOTS $r"; else info "note: local code root absent, skipped: $r"; fi
done
[ -n "${LROOTS// /}" ] || die "no code roots exist under $LOCAL_ROOT" 4

# ---------------------------------------------------------------------------
# enumerate local candidates
# ---------------------------------------------------------------------------
eval "$(build_find_cmd "$LROOTS")" > "$WORK/cand.z" 2>"$WORK/find.err"
frc=$?
[ "$frc" -eq 0 ] || { cat "$WORK/find.err" >&2; die "local find failed rc=$frc" 4; }

# Reject paths containing tab/newline/backslash. sha256sum escapes those and the
# manifest is TAB-separated; rather than silently mangle a path, stop.
if LC_ALL=C tr '\0' '\n' < "$WORK/cand.z" | grep -qP '[\t\\]'; then
  info "unsafe characters (tab/backslash) in these paths:"
  LC_ALL=C tr '\0' '\n' < "$WORK/cand.z" | grep -nP '[\t\\]' >&2
  die "refusing to build a TAB-separated manifest over unsafe paths" 4
fi

# ---------------------------------------------------------------------------
# L4a: git check-ignore backstop. Catches secrets/artifacts added AFTER this
# script was written (e.g. a new configs/password_<node>.txt) without anyone
# having to remember to edit DENY_EXPR.
# ---------------------------------------------------------------------------
: > "$WORK/excluded.txt"
if git -C "$LOCAL_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$LOCAL_ROOT" check-ignore -z --stdin < "$WORK/cand.z" > "$WORK/gi.z" 2>/dev/null
  # rc 0 = some ignored, 1 = none ignored, >1 = real error. Both 0 and 1 are normal.
  LC_ALL=C tr '\0' '\n' < "$WORK/gi.z" | sed 's/$/\tgitignored/' >> "$WORK/excluded.txt"
else
  info "note: not a git repo, skipping git check-ignore backstop"
  : > "$WORK/gi.z"
fi

# ---------------------------------------------------------------------------
# L4b: content-level secret scan. Name rules alone are not enough:
# configs/litellm_proxy.yaml is git-TRACKED (not ignored) and holds live API keys.
#
# TWO TIERS, deliberately. A blanket content-block is wrong here and would make
# this tool break training: ~130 scripts/launch_*.sh legitimately export the
# project's WANDB_API_KEY (it is also in committed CLAUDE.md), and they are
# exactly the files a training node needs. A tool that silently withholds them
# gets bypassed, which is worse than a tool that ships them with a loud warning.
#   BLOCK : private-key blocks (never legitimate in source), and credential
#           material in *config/data* files (.yaml/.json/.ini/.toml/.txt/.env/...)
#           -- those files exist to carry configuration, so a secret in them is
#           the payload, not an incidental.
#   WARN  : credential-shaped strings inside executable source (.sh/.py/.md).
#           Shipped, but printed, so the hygiene debt is visible.
# grep note: the regex begins with '-' so it MUST be passed via `-e`, else grep
# parses it as an option and exits 2. First draft of this script did exactly
# that and the rc was swallowed -- hence the explicit rc!=0,1 check below.
# ---------------------------------------------------------------------------
PRIVKEY_RE='-----BEGIN [A-Z ]*PRIVATE KEY-----'
SECRET_RE='sk-[A-Za-z0-9]{24,}|sk-ant-[A-Za-z0-9_-]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{34,}|AKIA[0-9A-Z]{16}|(api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|passwd|password)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][A-Za-z0-9+/._-]{16,}'
: > "$WORK/secret_blocked.txt"
: > "$WORK/secret_warned.txt"
grep_has() { # $1=regex $2=file ; 0 hit, 1 miss, dies on grep error
  LC_ALL=C grep -I -q -a -E -e "$1" -- "$2"
  local g=$?
  [ "$g" -eq 0 ] && return 0
  [ "$g" -eq 1 ] && return 1
  die "grep failed rc=$g on '$2' (regex misparsed?)" 4
}
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s -- "$f" 2>/dev/null || echo 0)
  [ "$sz" -le 2000000 ] || continue          # skip big blobs; not source anyway
  if grep_has "$PRIVKEY_RE" "$f"; then
    printf '%s\tprivate-key\n' "$f" >> "$WORK/secret_blocked.txt"
    printf '%s\tsecret-content\n' "$f" >> "$WORK/excluded.txt"
    continue
  fi
  if grep_has "$SECRET_RE" "$f"; then
    case "$f" in
      *.yaml|*.yml|*.json|*.ini|*.cfg|*.conf|*.toml|*.txt|*.env|*.pem|*.key|*.properties)
        printf '%s\tcredential-in-config\n' "$f" >> "$WORK/secret_blocked.txt"
        printf '%s\tsecret-content\n' "$f" >> "$WORK/excluded.txt" ;;
      *)
        printf '%s\n' "$f" >> "$WORK/secret_warned.txt" ;;
    esac
  fi
done < "$WORK/cand.z"

# ---------------------------------------------------------------------------
# payload = candidates - excluded
# ---------------------------------------------------------------------------
LC_ALL=C cut -f1 "$WORK/excluded.txt" | LC_ALL=C sort -u > "$WORK/excl_paths.txt"
LC_ALL=C tr '\0' '\n' < "$WORK/cand.z" | LC_ALL=C sort -u > "$WORK/cand.txt"
LC_ALL=C comm -23 "$WORK/cand.txt" "$WORK/excl_paths.txt" > "$WORK/payload.txt"

N_CAND=$(wc -l < "$WORK/cand.txt")
N_EXCL=$(wc -l < "$WORK/excl_paths.txt")
N_PAY=$(wc -l < "$WORK/payload.txt")
[ "$N_PAY" -gt 0 ] || die "payload is empty after exclusions -- refusing to ship nothing" 4

PAY_BYTES=$(LC_ALL=C tr '\n' '\0' < "$WORK/payload.txt" | xargs -0 -r stat -c%s -- \
            | awk '{s+=$1} END{printf "%d", s+0}')

GIT_HEAD=$(git -C "$LOCAL_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'NO_GIT')
GIT_DIRTY=$(git -C "$LOCAL_ROOT" status --porcelain -- $LROOTS 2>/dev/null | wc -l)

info "=============================================================="
info " ship_code_to_node.sh   mode=$MODE"
info " source (wzc1)  : $LOCAL_ROOT"
info " git HEAD       : $GIT_HEAD   (uncommitted changes in code roots: $GIT_DIRTY)"
[ "$MODE" = 'list' ] || info " target         : root@$HOST:$RROOT"
info " code roots     :$LROOTS"
info " candidates=$N_CAND  excluded=$N_EXCL  payload=$N_PAY files, $PAY_BYTES bytes"
info "=============================================================="

if [ -s "$WORK/secret_blocked.txt" ]; then
  info ""
  info "*** SECRET-BLOCKED by content scan (NOT shipped) ***"
  nl -ba "$WORK/secret_blocked.txt" | sed 's/^/    /'
fi
if [ -s "$WORK/secret_warned.txt" ]; then
  info ""
  info "*** WARNING: $(wc -l < "$WORK/secret_warned.txt") executable source file(s) embed credential-shaped strings."
  info "    These ARE shipped (they are the launch scripts a training node needs)."
  info "    Sample:"
  head -n 5 "$WORK/secret_warned.txt" | sed 's/^/      /'
  [ "$(wc -l < "$WORK/secret_warned.txt")" -gt 5 ] && info "      ... ($(( $(wc -l < "$WORK/secret_warned.txt") - 5 )) more; run --list for all)"
fi
NGI=$(LC_ALL=C grep -c $'\tgitignored$' "$WORK/excluded.txt" || true)
info ""
info "exclusion tally: gitignored=$NGI  secret-blocked=$(wc -l < "$WORK/secret_blocked.txt")  secret-warned-but-shipped=$(wc -l < "$WORK/secret_warned.txt")"

if [ "$MODE" = 'list' ]; then
  info ""
  info "--- payload manifest (relative to repo root) ---"
  cat "$WORK/payload.txt"
  info ""
  info "--- excluded (path<TAB>reason) ---"
  LC_ALL=C sort -u "$WORK/excluded.txt"
  info ""
  info "--- shipped-with-warning (credential-shaped string in executable source) ---"
  cat "$WORK/secret_warned.txt"
  exit 0
fi

# ---------------------------------------------------------------------------
# local manifest: sha256 <TAB> relpath, sorted by path
# ---------------------------------------------------------------------------
LC_ALL=C tr '\n' '\0' < "$WORK/payload.txt" | xargs -0 -r sha256sum -- > "$WORK/local.raw" 2>"$WORK/sha.err"
src=$?
[ "$src" -eq 0 ] || { cat "$WORK/sha.err" >&2; die "local sha256sum failed rc=$src" 4; }
LC_ALL=C awk '{h=$1; sub(/^[^ ]+ [ *]/,""); printf "%s\t%s\n", $0, h}' "$WORK/local.raw" \
  | LC_ALL=C sort -k1,1 > "$WORK/local.tsv"
[ "$(wc -l < "$WORK/local.tsv")" -eq "$N_PAY" ] || die "manifest size $(wc -l < "$WORK/local.tsv") != payload $N_PAY" 4

# ---------------------------------------------------------------------------
# ship
# ---------------------------------------------------------------------------
T0=$(date +%s)
TAR_BYTES=0
if [ "$MODE" = 'ship' ]; then
  info ""
  info "[1/3] building tar ..."
  LC_ALL=C tr '\n' '\0' < "$WORK/payload.txt" \
    | tar --null -T - -czf "$WORK/payload.tgz" --owner=0 --group=0 2>"$WORK/tar.err"
  # pipeline: tr is [0], tar is [1]. Check BOTH -- a bare $? here would be tar's
  # only, and a bare `| tail; echo $?` pattern has fabricated success in this
  # repo twice.
  trc=("${PIPESTATUS[@]}")
  [ "${trc[0]}" -eq 0 ] && [ "${trc[1]}" -eq 0 ] || {
    cat "$WORK/tar.err" >&2; die "tar build failed (tr=${trc[0]} tar=${trc[1]})" 4; }
  TAR_BYTES=$(stat -c%s -- "$WORK/payload.tgz")
  info "      tar = $TAR_BYTES bytes gz (from $PAY_BYTES raw, $N_PAY files)"

  info "[2/3] pushing to root@$HOST:$RROOT ..."
  ssh_run "mkdir -p -- '$RROOT' && tar -xzf - -C '$RROOT'" < "$WORK/payload.tgz" \
      > "$WORK/push.log" 2>&1
  prc=$?
  cat "$WORK/push.log"
  [ "$prc" -eq 0 ] || die "remote mkdir/untar failed rc=$prc (see output above)" 3

  # provenance stamp, inside a dir that both sides prune from enumeration
  STAMP="shipped_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_root=$LOCAL_ROOT
source_host=$(hostname)
git_head=$GIT_HEAD
git_dirty_files_in_code_roots=$GIT_DIRTY
files=$N_PAY
raw_bytes=$PAY_BYTES
tar_gz_bytes=$TAR_BYTES
code_roots=$LROOTS"
  printf '%s\n' "$STAMP" | ssh_run "mkdir -p -- '$RROOT/.ship_code' && cat > '$RROOT/.ship_code/SHIP_INFO.txt'" >/dev/null 2>&1
  cp -- "$WORK/local.tsv" "$WORK/MANIFEST.tsv"
  ssh_run "cat > '$RROOT/.ship_code/MANIFEST.tsv'" < "$WORK/MANIFEST.tsv" >/dev/null 2>&1
fi
T1=$(date +%s)

# ---------------------------------------------------------------------------
# verify: enumerate + hash on the remote with the SAME expression
# ---------------------------------------------------------------------------
info ""
info "[$([ "$MODE" = ship ] && echo 3/3 || echo 1/1)] verifying sha256 parity on $HOST ..."
REMOTE_SCRIPT="set -u
cd -- '$RROOT' 2>/dev/null || { echo '__ROOT_MISSING__'; exit 0; }
R=''
for r in $LROOTS; do [ -e \"\$r\" ] && R=\"\$R \$r\"; done
if [ -z \"\${R# }\" ]; then echo '__NO_ROOTS__'; exit 0; fi
$(build_find_cmd '$R') | xargs -0 -r sha256sum --"
# NOTE: $(build_find_cmd '$R') expands build_find_cmd's template locally but
# leaves the literal string \$R for the remote shell -- the roots list is
# computed remotely so a root missing there is handled there.
ssh_run "$REMOTE_SCRIPT" > "$WORK/remote.raw" 2>"$WORK/remote.err"
rrc=$?
if [ "$rrc" -ne 0 ]; then
  cat "$WORK/remote.err" >&2
  die "remote enumeration failed rc=$rrc" 3
fi

ROOT_MISSING=0
if LC_ALL=C grep -qx '__ROOT_MISSING__' "$WORK/remote.raw"; then
  ROOT_MISSING=1
  info "      remote root does not exist: $RROOT"
  : > "$WORK/remote.tsv"
elif LC_ALL=C grep -qx '__NO_ROOTS__' "$WORK/remote.raw"; then
  info "      remote root exists but contains none of the code roots"
  : > "$WORK/remote.tsv"
else
  LC_ALL=C awk '{h=$1; sub(/^[^ ]+ [ *]/,""); printf "%s\t%s\n", $0, h}' "$WORK/remote.raw" \
    | LC_ALL=C sort -k1,1 > "$WORK/remote.tsv"
fi
N_REMOTE=$(wc -l < "$WORK/remote.tsv")

# classify. Remote files that we deliberately excluded locally (gitignored /
# secret-blocked) are NOT "extra": e.g. a password file placed on the node
# out-of-band is legitimate and is none of our business.
#
# Pass separation uses an FNR==1 counter, NOT FILENAME==ARGV[n]. The ARGV form
# is a trap: `awk '...' VAR=x f1 f2` puts the *assignment* in ARGV[1], so
# ARGV[1..3] were the -v-style assignments and every block was skipped -- n
# stayed 0 and the script printed "remote code == wzc1 code (rc=0)" against a
# node holding a partial checkout. Caught by the negative control.
LC_ALL=C awk -F'\t' -v OUT_MISS="$WORK/miss.txt" -v OUT_BAD="$WORK/bad.txt" \
                    -v OUT_EXTRA="$WORK/extra.txt" '
  FNR==1 { pass++ }
  pass==1 { skip[$1]=1; next }
  pass==2 { L[$1]=$2; order[++n]=$1; next }
  pass==3 { R[$1]=$2; next }
  END {
    for (i=1;i<=n;i++) { p=order[i]
      if (!(p in R))            { print "MISSING\t" p  > OUT_MISS;  nmiss++ }
      else if (R[p]!=L[p])      { print "MISMATCH\t" p > OUT_BAD;   nbad++  }
      else                      { nok++ }
    }
    for (p in R) if (!(p in L) && !(p in skip)) { print "EXTRA\t" p > OUT_EXTRA; nextra++ }
    printf "%d %d %d %d %d\n", nok+0, nmiss+0, nbad+0, nextra+0, n+0
  }' "$WORK/excl_paths.txt" "$WORK/local.tsv" "$WORK/remote.tsv" > "$WORK/counts.txt"
arc=$?
[ "$arc" -eq 0 ] || die "classification awk failed rc=$arc" 4
touch "$WORK/miss.txt" "$WORK/bad.txt" "$WORK/extra.txt"
read -r N_OK N_MISS N_BAD N_EXTRA N_SEEN < "$WORK/counts.txt"
# Self-check: every payload file must land in exactly one bucket. An empty
# classification that still reports parity is the failure mode above; assert it
# away rather than trusting the loop ran.
[ "$N_SEEN" -eq "$N_PAY" ] || die "classifier saw $N_SEEN of $N_PAY payload files -- pass separation is broken, parity result is NOT trustworthy" 4
[ $(( N_OK + N_MISS + N_BAD )) -eq "$N_PAY" ] || die "bucket sum $(( N_OK + N_MISS + N_BAD )) != payload $N_PAY" 4

# ---------------------------------------------------------------------------
# optional prune of remote-only code files
# ---------------------------------------------------------------------------
if [ "$PRUNE_EXTRA" -eq 1 ] && [ "$N_EXTRA" -gt 0 ]; then
  info ""
  info "--prune-extra: deleting $N_EXTRA remote-only code file(s) under $RROOT"
  LC_ALL=C cut -f2 "$WORK/extra.txt" | sed 's/^/    rm /'
  LC_ALL=C cut -f2 "$WORK/extra.txt" | tr '\n' '\0' \
    | ssh_run "cd -- '$RROOT' && xargs -0 -r rm -f --" > "$WORK/prune.log" 2>&1
  krc=$?
  cat "$WORK/prune.log"
  [ "$krc" -eq 0 ] || die "remote prune failed rc=$krc" 3
  N_EXTRA=0
  : > "$WORK/extra.txt"
fi

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
show() { # $1=file $2=label $3=cap
  [ -s "$1" ] || return 0
  info ""
  info "--- $2 ---"
  head -n "$3" "$1" | sed 's/^/    /'
  t=$(wc -l < "$1"); [ "$t" -gt "$3" ] && info "    ... and $((t-$3)) more"
  return 0
}
show "$WORK/bad.txt"   "MISMATCHED (remote content differs from wzc1)" 40
show "$WORK/miss.txt"  "MISSING on remote"                            40
show "$WORK/extra.txt" "EXTRA on remote (present remotely, not in wzc1 payload)" 40

DIFF=$(( N_MISS + N_BAD ))
[ "$ALLOW_EXTRA" -eq 1 ] || DIFF=$(( DIFF + N_EXTRA ))

info ""
info "=============================================================="
info " $N_PAY files shipped, $N_OK verified, $N_BAD mismatched, $N_MISS missing, $N_EXTRA extra"
info " remote enumerated: $N_REMOTE files under $RROOT"
if [ "$MODE" = 'ship' ]; then
  DT=$(( T1 - T0 )); [ "$DT" -gt 0 ] || DT=1
  info " transfer: $TAR_BYTES bytes gz in ${DT}s = $(awk -v b="$TAR_BYTES" -v t="$DT" 'BEGIN{printf "%.2f", b/t/1048576}') MiB/s (tar+push wall)"
fi
if [ "$DIFF" -eq 0 ]; then
  info " RESULT: remote code == wzc1 code  (rc=0)"
  info "=============================================================="
  exit 0
fi
info " RESULT: NOT IN PARITY -- $DIFF differing file(s)  (rc=1)"
[ "$MODE" = 'check' ] && info " remedy: re-run without --check-only to ship$([ "$N_EXTRA" -gt 0 ] && echo ', add --prune-extra to drop remote-only files')"
info "=============================================================="
exit 1
