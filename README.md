# Clinamen2 documentation

The local compilation of the documentation requires the packages:

- [sphinx](https://pypi.org/project/Sphinx/)
- [sphinx-autodoc-typehints](https://pypi.org/project/sphinx-autodoc-typehints/)
- [sphinx-rtd-theme](https://pypi.org/project/sphinx-rtd-theme/)

that can be installed manually or by installing Clinamen2 with
```
    pip install -e .[docs]
```

Then you can run
```
    make html
```

in the docs directory.

Please be aware that for compiling the complete docs, [NeuralIL](https://github.com/Madsen-s-research-group/neuralil-public-releases/tree/clinamen2)
needs to be installed with all dependencies.
