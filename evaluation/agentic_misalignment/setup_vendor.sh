#!/bin/bash
# Fetch MSM's agentic-misalignment eval code into vendor/ (gitignored: upstream
# has no license, so it is never committed) at the pinned commit every number
# in the post was run on.
set -e
cd "$(dirname "$0")"
PIN=e8288a84912ba32af68ad15f2e52a7c1b4e81891
rm -rf vendor && mkdir vendor && cd vendor
git init -q
git remote add origin https://github.com/chloeli-15/model_spec_midtraining
git fetch -q --depth 1 origin $PIN
git checkout -q FETCH_HEAD
cd ..
# Our grader-retry patch goes on top of every fresh fetch.
./apply_template_overrides.sh
echo "vendor ready at $PIN"
