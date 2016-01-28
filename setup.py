from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os.path
import re
import sys


dist_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dist_dir, 'seqlearn/_version.py'), 'r') as f:
    version = re.search("^ *__version__ = '(.*)'$", f.read()).group(1)


def readme():
    try:
        with open(os.path.join(dist_dir, 'README.rst')) as f:
            return f.read()
    except IOError:
        return "seqlearn: sequence classification library for Python"


setup_options = dict(
    name="seqlearn",
    version=version,
    description="Sequence learning toolkit",
    maintainer="Lars Buitinck",
    maintainer_email="larsmans@gmail.com",
    license="MIT",
    url="https://github.com/larsmans/seqlearn",
    packages=["seqlearn", "seqlearn._utils", "seqlearn._decode",
              "seqlearn/tests"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
    ],
    ext_modules=["seqlearn/_decode/bestfirst.pyx",
                 "seqlearn/_decode/viterbi.pyx",
                 "seqlearn/_utils/ctrans.pyx",
                 "seqlearn/_utils/safeadd.pyx"],
    requires=["sklearn", "Cython"],
)

# NOTE: See https://github.com/hmmlearn/hmmlearn/issues/43. However,
# cythonize doesn't pass include_path to Extension either, so we're
# hacking it directly.
for em in setup_options["ext_modules"]:
    em.include_dirs = [np.get_include()]
    
# FIXME: Cython doesn't exist on Heroku before pip runs. And this depends
# on Cython. But we can't declare that we depend on Cython because we try
# to import Cython before it exists and promptly exception out. So, we declare
# the dependency above (when pip presumably first loads it to check dependencies)
# and then backtrack and add back the cythonize call after Cython is installed by
# pip and it calls us again to actually install us. This is a glorious, GLORIOUS
# hack but fuck it. It's 01-27 and we're launching in 4 days. JFDI, bitch.
#
# -@kushalc (2016-01-27)
#
# P.S. I hope some interns get a laugh out of this someday.
try:
    from Cython.Build import cythonize
    setup_options["ext_modules"] = cythonize(setup_options["ext_modules"])
except ImportError:
    pass

setup(**setup_options)
