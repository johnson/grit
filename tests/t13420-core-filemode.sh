#!/bin/sh

test_description='core.filemode=false hides the execute bit from status and diff'

. ./test-lib.sh

# git trusts the execute bit only while core.filemode is true, and init turns it
# off by itself when the bits on disk disagree with the index -- which is what a
# filesystem that hands out 0700 for everything produces. That was the iPad
# checkout behind a 9763-file mode-only commit. With the bit untrusted, a
# mode-only difference is not a change.
#
# Two environment notes, both learned by having this test fail first:
#  - .git/info/exclude keeps the AppleDouble `._*` companions out. On an ExFAT
#    volume the harness writes one next to every file, and they would otherwise
#    be committed alongside the fixtures.
#  - every status assertion passes --untracked-files=no, because the actual
#    files land inside the repo and would otherwise be what is asserted about.
test_expect_success 'setup: tracked files carry an execute bit the trees do not' '
	git init repo &&
	(
	cd repo &&
	git config user.name "Test" &&
	git config user.email "test@example.com" &&
	echo "._*" >.git/info/exclude &&
	git config core.filemode false &&
	echo one >one.txt &&
	echo two >two.txt &&
	mkdir sub &&
	echo three >sub/three.txt &&
	git add -A &&
	git commit -m "initial" &&
	chmod 700 one.txt two.txt sub/three.txt
	)
'

test_expect_success 'status reports nothing after only the bit changed' '
	(
	cd repo &&
	git status --porcelain --untracked-files=no >actual &&
	test_must_be_empty actual
	)
'

test_expect_success 'diff-files reports nothing after only the bit changed' '
	(
	cd repo &&
	git diff-files >actual &&
	test_must_be_empty actual
	)
'

test_expect_success 'the whole tree is not offered as a mode change' '
	(
	cd repo &&
	git diff --summary >actual &&
	test_must_be_empty actual
	)
'

test_expect_success 'a real content change is still reported' '
	(
	cd repo &&
	echo edited >two.txt &&
	git status --porcelain --untracked-files=no >actual &&
	grep "^ M two.txt" actual
	)
'

test_expect_success 'add keeps the mode the tree recorded' '
	(
	cd repo &&
	git add two.txt &&
	git ls-files --stage two.txt >actual &&
	grep "^100644 " actual
	)
'

test_expect_success 'committing the edit carries no mode change' '
	(
	cd repo &&
	git commit -qam "edit only" &&
	git diff --summary HEAD~1 HEAD >actual &&
	! grep "mode change" actual
	)
'

test_expect_success 'core.filemode=true makes the same bit a change again' '
	(
	cd repo &&
	git config core.filemode true &&
	git status --porcelain --untracked-files=no >actual &&
	grep "^ M one.txt" actual
	)
'

test_expect_success 'add records 100755 once the bit is trusted' '
	(
	cd repo &&
	echo executable >script.sh &&
	chmod 755 script.sh &&
	git add script.sh &&
	git ls-files --stage script.sh >actual &&
	grep "^100755 " actual
	)
'

test_expect_success 'a new file is 100644 while the bit is untrusted' '
	(
	cd repo &&
	git config core.filemode false &&
	echo plain >plain.sh &&
	chmod 755 plain.sh &&
	git add plain.sh &&
	git ls-files --stage plain.sh >actual &&
	grep "^100644 " actual
	)
'

test_done
