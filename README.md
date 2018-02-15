Fast Word Mover's Distance [![Build Status](https://travis-ci.org/src-d/wmd-relax.svg?branch=master)](https://travis-ci.org/src-d/wmd-relax) [![PyPI](https://img.shields.io/pypi/v/wmd.svg)](https://pypi.python.org/pypi/wmd) [![codecov](https://codecov.io/github/src-d/wmd-relax/coverage.svg)](https://codecov.io/gh/src-d/wmd-relax)
==========================

Calculates Word Mover's Distance as described in
[From Word Embeddings To Document Distances](http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf)
by Matt Kusner, Yu Sun, Nicholas Kolkin and Kilian Weinberger.

<img src="https://blog.sourced.tech/post/lapjv/wmd.png" alt="Word Mover's Distance" width="200"/>

The high level logic is written in Python, the low level functions related to
linear programming are offloaded to the bundled native extension. The native
extension can be built as a generic shared library not related to Python at all.
**Python 2.7 and older are not supported.** The heavy-lifting is done by
[google/or-tools](https://github.com/google/or-tools).


### Installation

```
pip3 install wmd
```
Tested on Linux and macOS.

### Usage

You should have the embeddings numpy array and the nbow model - that is,
every sample is a weighted set of items, and every item is embedded.

```python
import numpy
from wmd import WMD

embeddings = numpy.array([[0.1, 1], [1, 0.1]], dtype=numpy.float32)
nbow = {"first":  ("#1", [0, 1], numpy.array([1.5, 0.5], dtype=numpy.float32)),
        "second": ("#2", [0, 1], numpy.array([0.75, 0.15], dtype=numpy.float32))}
calc = WMD(embeddings, nbow, vocabulary_min=2)
print(calc.nearest_neighbors("first"))
```
```
[('second', 0.10606599599123001)]
```

`embeddings` must support `__getitem__` which returns an item by it's
identifier; particularly, `numpy.ndarray` matches that interface.
`nbow` must be iterable - returns sample identifiers - and support
`__getitem__` by those identifiers which returns tuples of length 3.
The first element is the human-readable name of the sample, the
second is an iterable with item identifiers and the third is `numpy.ndarray`
with the corresponding weights. All numpy arrays must be float32. The return
format is the list of tuples with sample identifiers and relevancy
indices (lower the better).

It is possible to use this package with [spaCy](https://github.com/explosion/spaCy):

```python
import spacy
import wmd

nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
doc1 = nlp("Politician speaks to the media in Illinois.")
doc2 = nlp("The president greets the press in Chicago.")
print(doc1.similarity(doc2))
```

Besides, see another [example](spacy_example.py) which finds similar Wikipedia
pages.

### Building from source

Either build it as a Python package:

```
pip3 install git+https://github.com/src-d/wmd-relax
```

or use CMake:

```
git clone --recursive https://github.com/src-d/wmd-relax
cmake -D CMAKE_BUILD_TYPE=Release .
make -j
```

Please note the `--recursive` flag for `git clone`. This project uses source{d}'s
fork of [google/or-tools](https://github.com/google/or-tools) as the git submodule.

### Tests

Tests are in `test.py` and use the stock `unittest` package.

### Documentation

```
cd doc
make html
```

The files are in `doc/doxyhtml` and `doc/html` directories.

### Contributions

...are welcome! See [CONTRIBUTING](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

### License
[Apache 2.0](LICENSE.md)

#### README {#ignore_this_doxygen_anchor}
