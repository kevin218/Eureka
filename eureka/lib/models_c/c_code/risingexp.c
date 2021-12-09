#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *risingexp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *risingexp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,m,x0;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  m    = IND(rampparams,1);
  x0   = IND(rampparams,2);

  dims[0] = x->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);

  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal*(1-exp(-1*m*(IND(x,i)-x0)));
    }
  return PyArray_Return(y);
}

static char module_docstring[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m:	   rise exp\n\
    x0:	   time offset\n\
    x:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2008-06-24	Kevin Stevenson, UCF  \n\
			    kevin218@knights.ucf.edu\n\n\
                Original version\n\
  2010-12-24    Nate Lust, UCF \n\
                natelust at linux dot com\n\n\
  2018-11-22    Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"risingexp",(PyCFunction)risingexp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_risingexp(void)
#else
    initrisingexp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "risingexp",             /* m_name */
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
        PyObject *m = Py_InitModule3("risingexp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
