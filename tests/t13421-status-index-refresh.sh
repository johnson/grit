#!/bin/sh

test_description='the git CLI keeps its index refresh (the shim path is tested in Rust)'

. ./test-lib.sh

#
# NOTE: this pins the *CLI's* behaviour, which already refreshed the index
# before that was moved into porcelain::status. The app does not go through
# the CLI -- it calls ops::status, which is porcelain::status -- so the
# change this describes is tested in NeuraScribeGrit/tests/index_ops.rs,
# where a control run without the fix fails. This test stays because the CLI
# must not regress either.
# Status hashes every entry whose cached stat data does not match the file on
# disk, which on a repository whose files were written by another device means
# nearly all of them. The stat tuple it just read makes the next pass a stat
# comparison instead of another full rehash -- but only if it is written back.
# That is what made status on the iPad cost minutes on every call rather than
# only the first.
#
# `touch` is how an index goes stale with content unchanged, and
# `git ls-files --debug` prints the cached stat data as text: the test reads a
# real git's view of the index instead of parsing the binary format. The
# mtimes are set to a fixed future date so the comparison cannot pass on a
# timestamp that happened to coincide.

# Seconds since the epoch for $1, spelled out because `stat` takes different
# flags on Linux and macOS and the harness runs on both.
mtime_of () {
	perl -e 'print((stat($ARGV[0]))[9])' "$1"
}

test_expect_success 'setup: the index predates the files' '
	git init repo &&
	(
	cd repo &&
	git config user.name "Test" &&
	git config user.email "test@example.com" &&
	echo "one" >one.txt &&
	echo "two" >two.txt &&
	git add -A &&
	git commit -m base >/dev/null &&
	touch -t 203001010101 one.txt two.txt
	git ls-files --debug one.txt >cached.txt &&
	! grep -q "$(mtime_of one.txt)" cached.txt
	)
'

test_expect_success 'status reports the tree as clean' '
	(
	cd repo &&
	git status --porcelain --untracked-files=no >actual &&
	test_cmp /dev/null actual
	)
'

test_expect_success 'status leaves the mtime it read in the index' '
	(
	cd repo &&
	git ls-files --debug one.txt | grep -q "mtime: $(mtime_of one.txt)"
	)
'

test_expect_success 'a real change is still reported afterwards' '
	(
	cd repo &&
	echo changed >one.txt &&
	git status --porcelain --untracked-files=no >actual &&
	echo " M one.txt" >expect &&
	test_cmp expect actual
	)
'

test_expect_success 'a second status is clean again once the change is undone' '
	(
	cd repo &&
	git checkout -- one.txt &&
	git status --porcelain --untracked-files=no >actual &&
	test_cmp /dev/null actual
	)
'

test_done
