#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *re2ramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *re2ramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t,*y,*rampparams;
  double goal,a,b,m1,m2,t1,t2;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","t","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"		\
				  ,kwlist,&rampparams,&t,&etc))
    {
      return NULL;
    }
  
  goal = IND(rampparams,0);
  a    = IND(rampparams,1);
  m1   = IND(rampparams,2);
  t1   = IND(rampparams,3);
  b    = IND(rampparams,4);
  m2   = IND(rampparams,5);
  t2   = IND(rampparams,6);

  dims[0] = t->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);

  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal-a*exp(-m1*(IND(t,i)-t1))-b*exp(-m2*(IND(t,i)-t2));
    }
  return PyArray_Return(y);
}

static char module_docstring[]="\
  This function creates a model that fits a ramp using a rising exponential.\n\
\n\
  Parameters\n\
  ----------\n\
    goal:  goal as x -> inf\n\
    m1,m2: rise exp\n\
    t1,t2: time offset\n\
    t:	   Array of time/phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns an array of y values by combining an eclipse and a rising exponential\n\
\n\
  Revisions\n\
  ---------\n\
  2010-07-30    Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
  2010-12-24    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Converted to C\n\
  2018-11-22    Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\
";

static PyMethodDef module_methods[] = {
  {"re2ramp",(PyCFunction)re2ramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_re2ramp(void)
#else
    initre2ramp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "re2ramp",             /* m_name */
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
        PyObject *m = Py_InitModule3("re2ramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
