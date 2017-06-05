For installing Sphinx and Breathe (Doxygen & Sphinx bridge):

```bash
pip install Sphinx
pip install breathe
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
