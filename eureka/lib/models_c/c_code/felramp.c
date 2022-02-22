#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *felramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *felramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t,*y,*rampparams;
  double goal,m,t0,a,t1;
  int i;
  npy_intp dims[1];

  //etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","t","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"		\
				  ,kwlist,&rampparams,&t,&etc))
    {
      return NULL;
    }

  goal = IND(rampparams,0);
  m    = IND(rampparams,1);
  t0   = IND(rampparams,2);
  a    = IND(rampparams,3);
  t1   = IND(rampparams,4);

  dims[0] = t->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);

  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = goal*(1+exp(-1*m*(IND(t,i)-t0)))+a*(IND(t,i)-t1);
    }

  return PyArray_Return(y);
}


static char module_docstring[] ="\
    FINDME: This function needs a docstring -- the following is a copy/paste from fallingexp.c.\n\
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
    2008-06-16	  Kevin Stevenson, UCF  \n\
                  kevin218@knights.ucf.edu\n\
                  Original version\n\n\
    2018-11-22    Jonathan Fraine, SSI\n\
                  jfraine at spacescience.org\n\
                  Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"felramp",(PyCFunction)felramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_felramp(void)
#else
    initfelramp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "felramp",             /* m_name */
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
            PyObject *m = Py_InitModule3("felramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
