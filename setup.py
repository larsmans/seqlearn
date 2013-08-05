from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys

setup_options = dict(
    name="seqlearn",
    version="0.0.0",
    description="Sequence learning toolkit",
    maintainer="Lars Buitinck",
    maintainer_email="L.J.Buitinck@uva.nl",
    license="MIT",
    url="https://github.com/larsmans/seqlearn",
    packages=["seqlearn"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing",
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("seqlearn._decode.viterbi",
                           ["seqlearn/_decode/viterbi.pyx"])],
)

# For these actions, NumPy is not required. We want them to succeed without,
# for example when pip is used to install seqlearn without NumPy present.
NO_NUMPY_ACTIONS = ('egg_info', '--version', 'clean')
if len(sys.argv) < 2 or (not sys.argv[1:].startswith('--help')
                         and sys.argv[1] not in NO_NUMPY_ACTIONS):
    import numpy
    setup_options['include_dirs'] = [numpy.get_include()]

setup(**setup_options)
