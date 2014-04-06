#!/usr/bin/env python

# Write a directory to the Git index.
# Prints the directory's SHA-1 to stdout.
#
# Copyright 2013 Lars Buitinck / University of Amsterdam.
# License: MIT (http://opensource.org/licenses/MIT)

import os
from os.path import split
from posixpath import join
from subprocess import check_output, Popen, PIPE
import sys


def hash_file(path):
    """Write file at path to Git index, return its SHA1 as a string."""
    return check_output(["git", "hash-object", "-w", "--", path]).strip()


def _lstree(files, dirs):
    """Make git ls-tree like output."""
    for f, sha1 in files:
        yield "100644 blob {}\t{}\0".format(sha1, f)

    for d, sha1 in dirs:
        yield "040000 tree {}\t{}\0".format(sha1, d)


def _mktree(files, dirs):
    mkt = Popen(["git", "mktree", "-z"], stdin=PIPE, stdout=PIPE)
    return mkt.communicate("".join(_lstree(files, dirs)))[0].strip()


def hash_dir(path):
    """Write directory at path to Git index, return its SHA1 as a string."""
    dir_hash = {}

    for root, dirs, files in os.walk(path, topdown=False):
        f_hash = ((f, hash_file(join(root, f))) for f in files)
        d_hash = ((d, dir_hash[join(root, d)]) for d in dirs)
        # split+join normalizes paths on Windows (note the imports)
        dir_hash[join(*split(root))] = _mktree(f_hash, d_hash)

    return dir_hash[path]


if __name__ == "__main__":
    print(hash_dir(sys.argv[1]))
