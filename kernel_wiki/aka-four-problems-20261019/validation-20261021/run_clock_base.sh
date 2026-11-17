#!/bin/bash
set -u
ncu --profile-from-start off --target-processes all --clock-control base --cache-control all --replay-mode kernel --metrics gpu__time_duration.sum,gpc__cycles_elapsed.avg.per_second --csv --log-file ncu.csv python3 -B harness.py --op gdn_prefill --phase profile > profile.log 2>&1
status=$?
echo "profile_status=$status"
tail -6 profile.log
tar -czf results.tar.gz profile.json profile.log ncu.csv
exit "$status"
