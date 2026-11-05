#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/docs/agent_versions/ARTIFACT_MANIFEST.tsv"
failures=0

while IFS=$'\t' read -r version expected relative_path role; do
  [[ -z "${version:-}" || "$version" == \#* ]] && continue
  artifact="$repo_root/$relative_path"
  if [[ ! -f "$artifact" ]]; then
    printf 'MISSING\t%s\t%s\t%s\n' "$version" "$relative_path" "$role"
    failures=$((failures + 1))
    continue
  fi
  actual=$(sha256sum "$artifact" | awk '{print $1}')
  if [[ "$actual" == "$expected" ]]; then
    printf 'PRESENT\t%s\t%s\n' "$version" "$relative_path"
  else
    printf 'MISMATCH\t%s\t%s\texpected=%s\tactual=%s\n' \
      "$version" "$relative_path" "$expected" "$actual"
    failures=$((failures + 1))
  fi
done < "$manifest"

exit "$failures"
