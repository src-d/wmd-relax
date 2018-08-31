from setuptools import setup, Extension
import platform
try:
    import numpy
except ImportError as e:
    print("=" * 80)
    print("You need to install numpy *separately* and *before* installing this package(s).")
    print("=" * 80)
    raise e from None


PACKAGE = "wmd"

CXX_FLAGS = {
   "Darwin": ["-std=c++11", "-march=native", "-ftree-vectorize", "-DNDEBUG",
              "-Wno-sign-compare", "-fPIC", "-flto"],
   "Linux": ["-fopenmp", "-std=c++11", "-march=native", "-ftree-vectorize",
             "-DNDEBUG", "-Wno-sign-compare", "-fPIC", "-flto"],
   "Windows": ["/openmp", "/std:c++latest", "/arch:AVX2", "/DNDEBUG", "/LTCG",
               "/GL"]
}

LD_FLAGS = {
    "Darwin": ["-fPIC", "-flto"],
    "Linux": ["-fPIC", "-flto"],
    "Windows": ["/LTCG", "/GL"]
}

setup(
    name=PACKAGE,
    description="Accelerated functions to calculate Word Mover's Distance",
    version="1.2.11",
    license="Apache Software License",
    author="source{d}",
    author_email="vadim@sourced.tech",
    url="https://github.com/src-d/wmd-relax",
    download_url="https://github.com/src-d/wmd-relax",
    ext_modules=[Extension("libwmdrelax", sources=[
        "python.cc", "or-tools/src/graph/min_cost_flow.cc",
        "or-tools/src/graph/max_flow.cc", "or-tools/src/base/stringprintf.cc",
        "or-tools/src/base/logging.cc", "or-tools/src/base/sysinfo.cc",
        "or-tools/src/util/stats.cc"],
        extra_compile_args=CXX_FLAGS[platform.system()],
        extra_link_args=LD_FLAGS[platform.system()],
        include_dirs=[numpy.get_include(), "or-tools/src"])],
    packages=[PACKAGE],
    setup_requires=["numpy"],  # does not really help - we need it to get_include()
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ]
)

# python3 setup.py bdist_wheel
# auditwheel repair -w dist dist/*
# twine upload dist/*manylinux*
