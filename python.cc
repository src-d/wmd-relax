// fix inttypes for GCC
#define __STDC_FORMAT_MACROS
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include <Python.h>
// fix for the fix - it conflicts with numpy
#undef __STDC_FORMAT_MACROS
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "emd.h"
#include "emd_relaxed.h"

static char module_docstring[] =
    "This module provides functions which accelerate Word Mover's Distance calculation.";
static char emd_relaxed_docstring[] =
    "Approximates WMD by relaxing one of the two conditions and taking the max.";
static char emd_relaxed_cache_init_docstring[] =
    "Allocates the cache for emd_relaxed().";
static char emd_relaxed_cache_fini_docstring[] =
    "Deallocates the cache for emd_relaxed().";
static char emd_docstring[] = "Calculates the exact WMD.";
static char emd_cache_init_docstring[] = "Allocates the cache for emd().";
static char emd_cache_fini_docstring[] = "Deallocates the cache for emd().";

static PyObject *py_emd_relaxed(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_emd(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_emd_relaxed_cache_init(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_emd_relaxed_cache_fini(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_emd_cache_init(PyObject *self, PyObject *args, PyObject *kwargs);
static PyObject *py_emd_cache_fini(PyObject *self, PyObject *args, PyObject *kwargs);

static PyMethodDef module_functions[] = {
  {"emd_relaxed", reinterpret_cast<PyCFunction>(py_emd_relaxed),
   METH_VARARGS | METH_KEYWORDS, emd_relaxed_docstring},
  {"emd_relaxed_cache_init", reinterpret_cast<PyCFunction>(py_emd_relaxed_cache_init),
   METH_VARARGS, emd_relaxed_cache_init_docstring},
  {"emd_relaxed_cache_fini", reinterpret_cast<PyCFunction>(py_emd_relaxed_cache_fini),
   METH_VARARGS, emd_relaxed_cache_fini_docstring},
  {"emd", reinterpret_cast<PyCFunction>(py_emd),
   METH_VARARGS | METH_KEYWORDS, emd_docstring},
  {"emd_cache_init", reinterpret_cast<PyCFunction>(py_emd_cache_init),
   METH_VARARGS, emd_cache_init_docstring},
  {"emd_cache_fini", reinterpret_cast<PyCFunction>(py_emd_cache_fini),
   METH_VARARGS, emd_cache_fini_docstring},
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

static std::mutex emd_lock;

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

template <class C>
PyObject* emd_entry(
    PyObject *self, PyObject *args, PyObject *kwargs,
    float (*payload)(const float*, const float*, const float*, uint32_t, const C&)) {
  PyObject *w1_obj, *w2_obj, *dist_obj, *cache_obj = Py_None;
  static const char *kwlist[] = {"w1", "w2", "dist", "cache", NULL};
  if (!PyArg_ParseTupleAndKeywords(
      args, kwargs, "OOO|O", const_cast<char**>(kwlist),
      &w1_obj, &w2_obj, &dist_obj, &cache_obj)) {
    return NULL;
  }

  pyarray w1_array(PyArray_FROM_OTF(w1_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY)),
          w2_array(PyArray_FROM_OTF(w2_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY)),
          dist_array(PyArray_FROM_OTF(dist_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
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
  if (size == 0) {
    return Py_BuildValue("f", 1.0f);
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
  auto w1 = reinterpret_cast<float *>(PyArray_DATA(w1_array.get()));
  auto w2 = reinterpret_cast<float *>(PyArray_DATA(w2_array.get()));
  auto dist = reinterpret_cast<float *>(PyArray_DATA(dist_array.get()));

  C *cache = nullptr;
  std::unique_ptr<C> cache_ptr;
  if (cache_obj != Py_None) {
	long  l=PyLong_AsLong(cache_obj);
	cache = reinterpret_cast<C *>(reinterpret_cast<intptr_t>( &l ));
    if (PyErr_Occurred()) {
      return NULL;
    }
  }
  if (cache == nullptr) {
    cache_ptr.reset(new C());
    auto alloc_result = cache_ptr->allocate(size);
#ifndef NDEBUG
    assert(alloc_result == wmd::Cache::kAllocationErrorSuccess);
#else
    if (alloc_result != wmd::Cache::kAllocationErrorSuccess) {
      PyErr_SetString(PyExc_MemoryError, "failed to allocate the cache");
      return NULL;
    }
#endif
    cache = cache_ptr.get();
  }
  float result;
  Py_BEGIN_ALLOW_THREADS
  result = payload(w1, w2, dist, size, *cache);
  Py_END_ALLOW_THREADS
  if (result < 0) {
    PyErr_SetString(PyExc_RuntimeError, "negative cost was returned");
    return NULL;
  }
  return Py_BuildValue("f", result);
}

template <class C>
PyObject *cache_init(PyObject *args) {
  uint32_t size = 0;
  if (!PyArg_ParseTuple(args, "I", &size)) {
    return NULL;
  }
  auto cache = new C();
  cache->allocate(size);
  return Py_BuildValue("l", cache);
}

template <class C>
PyObject *cache_fini(PyObject *args) {
  intptr_t cache = 0;
  if (!PyArg_ParseTuple(args, "l", &cache)) {
    return NULL;
  }
  delete reinterpret_cast<C*>(cache);
  return Py_None;
}

static PyObject *py_emd_relaxed(PyObject *self, PyObject *args, PyObject *kwargs) {
  return emd_entry(self, args, kwargs, emd_relaxed<float>);
}

static PyObject *py_emd(PyObject *self, PyObject *args, PyObject *kwargs) {
  return emd_entry(self, args, kwargs, emd<float>);
}

static PyObject *py_emd_relaxed_cache_init(PyObject *self, PyObject *args, PyObject *kwargs) {
  return cache_init<EMDRelaxedCache>(args);
}

static PyObject *py_emd_relaxed_cache_fini(PyObject *self, PyObject *args, PyObject *kwargs) {
  return cache_fini<EMDRelaxedCache>(args);
}

static PyObject *py_emd_cache_init(PyObject *self, PyObject *args, PyObject *kwargs) {
  return cache_init<EMDCache>(args);
}

static PyObject *py_emd_cache_fini(PyObject *self, PyObject *args, PyObject *kwargs) {
  return cache_fini<EMDCache>(args);
}
