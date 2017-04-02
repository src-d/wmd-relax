#include <memory>
#include <string>
#include <unordered_map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cuda_runtime_api.h>
#include "emd.h"
#include "emd_relaxed.h"

static char module_docstring[] =
    "This module provides functions which accelerate Word Mover's Distance calculation.";
static char approximate_relaxed_docstring[] =
    "Approximates WMD by relaxing one of the two conditions and taking the max.";

static PyObject *py_approximate_relaxed(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"approximate_relaxed", reinterpret_cast<PyCFunction>(py_approximate_relaxed),
   METH_VARARGS | METH_KEYWORDS, approximate_relaxed_docstring},
  {NULL, NULL, 0, NULL}
};

extern "C" {
PyMODINIT_FUNC PyInit_libwmdrelax(void) {
  static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "libwmdrelax",         /* m_name */
      module_docstring,    /* m_doc */
      -1,                  /* m_size */
      module_functions,    /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
  };
  PyObject *m = PyModule_Create(&moduledef);
  if (m == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "PyModule_Create() failed");
    return NULL;
  }
  // numpy
  import_array();
  return m;
}
}

template <typename O>
using pyobj_parent = std::unique_ptr<O, std::function<void(O*)>>;

template <typename O>
class _pyobj : public pyobj_parent<O> {
 public:
  _pyobj() : pyobj_parent<O>(
      nullptr, [](O *p){ if (p) Py_DECREF(p); }) {}
  explicit _pyobj(PyObject *ptr) : pyobj_parent<O>(
      reinterpret_cast<O *>(ptr), [](O *p){ if(p) Py_DECREF(p); }) {}
  void reset(PyObject *p) noexcept {
    pyobj_parent<O>::reset(reinterpret_cast<O*>(p));
  }
};

using pyobj = _pyobj<PyObject>;
using pyarray = _pyobj<PyArrayObject>;

static PyObject *py_approximate_relaxed(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *w1_obj, *w2_obj, *dist_obj, *cache_obj = Py_None;
  static const char *kwlist[] = {"w1", "w2", "dist", "cache", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "OOO|O", const_cast<char**>(kwlist),
      &w1_obj, &w2_obj, &dist_obj, &cache_obj)) {
    return NULL;
  }

  pyarray w1_array(PyArray_FROM_OTF(w1_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY)),
          w2_array(PyArray_FROM_OTF(w2_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY)),
          dist_array(PyArray_FROM_OTF(dist_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY)),
          cache_array;
  if (!w1_array || !w2_array || !dist_array) {
    PyErr_SetString(PyExc_TypeError,
                    "\"w1\", \"w2\" and \"dist\" must be float32 numpy arrays");
    return NULL;
  }
  int size;
  {
    auto ndims = PyArray_NDIM(dist_array.get());
    if (ndims != 2) {
      PyErr_SetString(PyExc_ValueError, "\"dist\" must be a 2D float32 numpy array");
      return NULL;
    }
    auto dims = PyArray_DIMS(dist_array.get());
    if (dims[0] != dims[1]) {
      PyErr_SetString(PyExc_ValueError, "\"dist\" must be square");
      return NULL;
    }
    size = dims[0];
  }
  pyarray* w1ptr = &w1_array;
  pyarray* w2ptr = &w2_array;
  for (auto arr : {w1ptr, w2ptr}) {
    auto ndims = PyArray_NDIM(arr->get());
    if (ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "weights must be 1D float32 numpy arrays");
      return NULL;
    }
    auto dims = PyArray_DIMS(arr->get());
    if (dims[0] != size) {
      PyErr_SetString(PyExc_ValueError, "weights size does not match \"dist\"");
      return NULL;
    }
  }
  std::unique_ptr<int32_t[]> cache_ptr;
  int32_t *cache;
  if (cache_obj == Py_None) {
    cache_ptr.reset(new int32_t[size]);
    cache = cache_ptr.get();
  } else {
    cache_array.reset(PyArray_FROM_OTF(cache_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    if (!cache_array) {
      PyErr_SetString(PyExc_TypeError, "\"cache\" must be an int32 numpy array");
      return NULL;
    }
    auto ndims = PyArray_NDIM(cache_array.get());
    if (ndims != 1) {
      PyErr_SetString(PyExc_ValueError, "\"cache\" must be a 1D int32 numpy array");
      return NULL;
    }
    auto dims = PyArray_DIMS(cache_array.get());
    if (dims[0] < size) {
      PyErr_SetString(PyExc_ValueError, "\"cache\" size is too small");
      return NULL;
    }
    cache = reinterpret_cast<int32_t *>(PyArray_DATA(cache_array.get()));
  }
  auto w1 = reinterpret_cast<float *>(PyArray_DATA(w1_array.get()));
  auto w2 = reinterpret_cast<float *>(PyArray_DATA(w2_array.get()));
  auto dist = reinterpret_cast<float *>(PyArray_DATA(dist_array.get()));
  float result;
  Py_BEGIN_ALLOW_THREADS
  result = emd_relaxed(w1, w2, dist, size, cache);
  Py_END_ALLOW_THREADS
  return Py_BuildValue("f", result);
}
