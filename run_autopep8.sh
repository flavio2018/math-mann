#!/bin/sh

# from: https://stackoverflow.com/questions/29738206/autopep8-in-a-git-pre-commit-how-to-automatically-recommit
# runs autopep8 on the files included in the commit

status=0
touched_python_files=$(git diff --cached --name-only | grep -E '\.py$')
if [ -n "$touched_python_files" ]; then
    options="\
      --ignore=E26 \
      --max-line-length=150 \
    "

    output=$(autopep8 -d $options "${touched_python_files[@]}")
    # status=$?  # when autopep8 does nothing this blocks commits
    autopep8_status=$?
    if [ $autopep8_status -ne 0 ]; then
	echo ">>> autopep8 did not edit any file <<<"
    fi

    if [ -n "$output" ]; then
        autopep8 -i -j 0 $options "${touched_python_files[@]}"
        echo ">>> autopep8 edited some files <<<"
        exit 1
    fi
fi

exit $status

