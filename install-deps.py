#!/usr/bin/env python

if __name__ == '__main__':
    import sys
    import os
    import subprocess

    for package in sys.argv[1:]:
        if os.path.isfile(package):
            cmd = "pip install " + package
        elif package == 'conda':
            cmd = "pip install conda; conda init"
        else:
            cmd = "conda install --yes --quiet " + package
        print(cmd)
        subprocess.call(cmd, shell=True)

