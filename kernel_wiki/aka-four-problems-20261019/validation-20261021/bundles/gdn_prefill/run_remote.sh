#!/bin/bash
set -u
op="$1"
python3 -B harness.py --op "$op" --phase validate > validate.log 2>&1
validate_status=$?
echo "validate_status=$validate_status"
tail -8 validate.log
if [ "$validate_status" -eq 0 ]; then
  ncu --profile-from-start off --target-processes all --clock-control none --cache-control all --replay-mode kernel --metrics gpu__time_duration.sum,gpc__cycles_elapsed.avg.per_second --csv --log-file ncu.csv python3 -B harness.py --op "$op" --phase profile > profile.log 2>&1
  profile_status=$?
  echo "profile_status=$profile_status"
  tail -8 profile.log
else
  profile_status=99
fi
tar -czf results.tar.gz validate.json validate.log profile.json profile.log ncu.csv 2>/dev/null || true
exit "$((validate_status || profile_status))"
