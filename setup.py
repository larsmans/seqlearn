from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os.path
import sys


dist_dir = os.path.dirname(os.path.abspath(__file__))
execfile(os.path.join(dist_dir, 'seqlearn/_version.py'))


def readme():
    try:
        with open(os.path.join(dist_dir, 'README.rst')) as f:
            return f.read()
    except IOError:
        return "seqlearn: sequence classification library for Python"


setup_options = dict(
    name="seqlearn",
    version=__version__,
    description="Sequence learning toolkit",
    maintainer="Lars Buitinck",
    maintainer_email="larsmans@gmail.com",
    license="MIT",
    url="https://github.com/larsmans/seqlearn",
    packages=["seqlearn", "seqlearn._utils", "seqlearn._decode",
              "seqlearn/tests"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing",
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("seqlearn._decode.bestfirst",
                  ["seqlearn/_decode/bestfirst.pyx"]),
        Extension("seqlearn._decode.viterbi", ["seqlearn/_decode/viterbi.pyx"]),
        Extension("seqlearn._utils.ctrans", ["seqlearn/_utils/ctrans.pyx"]),
        Extension("seqlearn._utils.safeadd", ["seqlearn/_utils/safeadd.pyx"]),
    ],
)

# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('--help-commands', 'egg_info', '--version', 'clean')
if not ('--help' in sys.argv[1:]
        or len(sys.argv) > 1 and sys.argv[1] in NO_NUMPY_ACTIONS):
    import numpy
    setup_options['include_dirs'] = [numpy.get_include()]

setup(**setup_options)
