#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *chisq(PyObject *self, PyObject *args);

static PyObject *chisq(PyObject *self, PyObject *args)
{
  PyArrayObject *mod, *data, *errors;
  int i,lim;
  double chi;

  if(!PyArg_ParseTuple(args,"OOO", &mod, &data, &errors))
    {
      return NULL;
    }
  
  lim = mod->dimensions[0];
  chi = 0;

  //CANNOT PERFORM PARALLEL CALCULATION HERE
  for(i=0;i<lim;i++)
    {
      chi += pow((IND(mod,i)-IND(data,i))/IND(errors,i),2);
    }
  return Py_BuildValue("d",chi);
}

static char module_docstring[]="\
   This function creates the chi squared statistic given input parameters.\n\
\n\
   Parameters\n\
   ----------\n\
   mod:   1D NPY ARRAY - contains the model to be tested\n\
   data:  1D NPY ARRAY - contains the actual measurements\n\
   errors 1D NPY ARRAY - errors made on the meaurements (not weights)\n\
\n\
   Returns\n\
   -------\n\
   Float - the chi squared value given the model and weights\n\
\n\
   Revisions\n\
   ---------\n\
    2010-06-11   Kevin Stevenson, UCF\n\
                 kevin218@knights.ucf.edu\n\
                 Original version\n\n\
   2011-01-08    Nate Lust, UCF\n\
                 natelust at linux dot com\n\
                 Initial version, as c extension\n\n\
    2018-11-22   Jonathan Fraine, SSI\n\
                 jfraine at spacescience.org\n\
                 Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"chisq", chisq,METH_VARARGS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_chisq(void)
#else
    initchisq(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "chisq",             /* m_name */
            module_docstring,    /* m_doc */
            -1,                  /* m_size */
            module_methods,      /* m_methods */
            NULL,                /* m_reload */
            NULL,                /* m_traverse */
            NULL,                /* m_clear */
            NULL,                /* m_free */
        };
    #endif

    #if PY_MAJOR_VERSION >= 3
        module = PyModule_Create(&moduledef);
        if (!module)
            return NULL;
        /* Load `numpy` functionality. */
        import_array();
        return module;
    #else
        PyObject *m = Py_InitModule3("chisq", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
