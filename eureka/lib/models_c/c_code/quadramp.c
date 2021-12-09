#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *quadramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *quadramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double a,b,c,x0;
  int i;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  a  = IND(rampparams,0);
  b  = IND(rampparams,1);
  c  = IND(rampparams,2);
  x0 = IND(rampparams,3);

  dims[0] = x->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = a*pow((IND(x,i)-x0),2)+b*(IND(x,i)-x0)+c;
    }
  return PyArray_Return(y);
}

static char module_docstring[] = "\
 NAME:\n\
	QUADRAMP\n\
\n\
 PURPOSE:\n\
	This function creates a model that fits a quadratically ramped eclipse\n\
\n\
 CATEGORY:\n\
	Astronomy.\n\
\n\
 CALLING SEQUENCE:\n\
\n\
	Result = QUADRAMP([midpt,width,depth,a,b,c],x)\n\
\n\
 INPUTS:\n\
    	midpt:	Midpoint of eclipse\n\
	width:	Eclipse durations\n\
	depth:	Depth of eclipse\n\
	a:	x^2 constant\n\
	b:	x constant\n\
	c:	x=0 offset\n\
	x0: time/phase offset (constant)\n\
	x:	Array of time/phase points\n\
\n\
 OUTPUTS:\n\
	This function returns an array of y values by combining an eclipse and a quadratic\n\
\n\
 PROCEDURE:\n\
\n\
 EXAMPLE:\n\
\n\
\n\
\n\
 MODIFICATION HISTORY:\n\
 	Written by:	Kevin Stevenson, UCF  	\n\n\
    2008-06-22   Kevin Stevenson, UCF  	\n\
			     kevin218@knights.ucf.edu\n\n\
    2010-12-25   Nate Lust, UCF          \n\
                 natelust at linux dot com\n\
                 converted to c\n\n\
    2018-11-22   Jonathan Fraine, SSI\n\
                 jfraine at spacescience.org\n\
                 Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"quadramp",(PyCFunction)quadramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_quadramp(void)
#else
    initquadramp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "quadramp",             /* m_name */
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
        PyObject *m = Py_InitModule3("quadramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
