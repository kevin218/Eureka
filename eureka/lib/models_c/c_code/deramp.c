#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *deramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *deramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double g,r0,r1,th0,th1,pm,goal,a,b,gb,r0b,r1b;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  g    = IND(rampparams,0);
  r0   = IND(rampparams,1);
  r1   = IND(rampparams,2);
  th0  = IND(rampparams,3);     //Angle b/w r0 & r1
  th1  = IND(rampparams,4);     //Angle b/w r0 & g
  pm   = IND(rampparams,5);
  gb   = IND(rampparams,6);     //Best-fit value
  r0b  = IND(rampparams,7);     //Best-fit value
  r1b  = IND(rampparams,8);     //Best-fit value
  
  a    =  r0*cos(th1)*cos(th0) - r1*cos(th1)+sin(th0) + g*sin(th1) + r0b;
  b    =  r0*sin(th0)          + r1*cos(th0)                       + r1b;
  goal = -r0*sin(th1)*cos(th0) + r1*sin(th1)*sin(th0) + g*cos(th1) + gb;

  dims[0] = x->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);

  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      // IND(y,i) = goal+pm*exp(-(IND(x,i)*cost-sint)/r0 + (IND(x,i)*sint+cost)*r1);
      IND(y,i) = goal+pm*exp(-a*IND(x,i) + b);
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
			    kevin218@knights.ucf.edu\n\
		        Original version\n\n\
  2010-12-24    Nate Lust, UCF \n\
                natelust at linux dot com\n\n\
  2018-11-22    Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated C extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"deramp",(PyCFunction)deramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

// static char module_docstring[] =
// "This module is used to calcuate the deramp";

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_deramp(void)
#else
    initderamp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "deramp",             /* m_name */
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
        PyObject *m = Py_InitModule3("deramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
