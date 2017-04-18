# Fast Word Mover's Distance

Calculates Word Mover's Distance as described in
[From Word Embeddings To Document Distances](http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf)
by Matt Kusner, Yu Sun, Nicholas Kolkin and Kilian Weinberger.

The high level logic is written in Python, the low level functions related to
linear programming are offloaded to the bundled native extension. The native
extension can be built as a generic shared library not related to Python at all.
**Python 2.7 and older are not supported.** The heavy-lifting is done by
[google/or-tools](https://github.com/google/or-tools).


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

### License

MIT.