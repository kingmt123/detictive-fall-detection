#!/usr/bin/env bash
# URFD 下载脚本 v2：cd 到目标目录后用相对路径写文件（curl -o 绝对路径在本环境会 exit 23）
set -u
BASE="https://fenix.ur.edu.pl/~mkepski/ds/data"
DST="/d/HermesWorkspace/Detictive/data/sources/urfd"
mkdir -p "$DST"
cd "$DST" || exit 1
LOG="_download.log"
: > "$LOG"
rm -f t1.txt t2.txt test-fall-01-cam1.mp4
fail=0; ok=0
dl() {
  local f="$1"
  if curl -sL -C - --max-time 1200 --retry 3 --retry-delay 5 -o "$f" "$BASE/$f"; then
    ok=$((ok+1)); echo "OK  $f" >> "$LOG"
  else
    fail=$((fail+1)); echo "FAIL $f" >> "$LOG"
  fi
}
for i in $(seq -w 1 30); do
  dl "fall-$i-cam0.mp4"; dl "fall-$i-cam1.mp4"; dl "fall-$i-acc.csv"
done
for i in $(seq -w 1 40); do
  dl "adl-$i-cam0.mp4"; dl "adl-$i-acc.csv"
done
echo "DONE ok=$ok fail=$fail" >> "$LOG"
echo "DONE ok=$ok fail=$fail"
