#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *seqramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *seqramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double goal,r0,r1,r2,r3,x0,pm;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  r0   = IND(rampparams,1);
  r1   = IND(rampparams,2);
  r2   = IND(rampparams,3);
  r3   = IND(rampparams,4);
  x0   = IND(rampparams,5);
  pm   = IND(rampparams,6);

  dims[0] = x->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal + pm*exp(-r0*IND(x,i)+r1) + r2*(IND(x,i)-x0) + r3*pow((IND(x,i)-x0),2);
    }
  return PyArray_Return(y);
}

static char module_docstring[]="\
    This function NEEDS A DOC_STRING.\n\
    \n\
    Parameters\n\
    ----------\n\
    \n\
    Returns\n\
    -------\n\
    \n\
    Revisions\n\
    ---------\n\
    2010-07-30    Kevin Stevenson, UCF  \n\
                  kevin218@knights.ucf.edu\n\
                  Original version\n\n\
    2010-12-24    Nate Lust, UCF\n\
                  natelust at linux dot com\n\
                  Converted to C\n\n\
    2018-11-22    Jonathan Fraine, SSI\n\
                  jfraine at spacescience.org\n\
                  Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"seqramp",(PyCFunction)seqramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_seqramp(void)
#else
    initseqramp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "seqramp",             /* m_name */
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
        PyObject *m = Py_InitModule3("seqramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
