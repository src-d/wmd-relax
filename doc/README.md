For installing Sphinx:

```bash
pip install Sphinx
```

You'll also need the "doxygen" command installed. This is usually
installed with:

```bash
(apt or yum or whatever your distro uses) install doxygen
```

For generating/updating the API doc: 

```bash
cd doc
sphinx-apidoc -o wmd ../wmd/
```

For generating Doxygen XML files:
```bash
doxygen # Should use doc/Doxyfile, generates to doc/doxyhtml
```

For generating/updating the HTML files:

```
make html
```

These should include a link to the Doxygen documentation in Sphinx'
doc/_build/html/index.html.
