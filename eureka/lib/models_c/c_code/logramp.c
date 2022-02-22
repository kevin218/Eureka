#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *logramp(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *logramp(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *x,*y, *rampparams;
  double x0,a,b,c,d,e;
  int i;
  npy_intp dims[1];

  static char *kwlist[] = {"rampparams","x","etc",NULL};

  //etc = PyList_New(0);

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&rampparams,&x,&etc))
    {
      return NULL;
    }

  x0 = IND(rampparams,0);
  a  = IND(rampparams,1);
  b  = IND(rampparams,2);
  c  = IND(rampparams,3);
  d  = IND(rampparams,4);
  e  = IND(rampparams,5);

  dims[0] = x->dimensions[0];

  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  #pragma omp parallel for 
  for(i=0;i<dims[0];i++)
    {
      if(IND(x,i)<=x0)
        IND(y,i) = e;
      else
	    IND(y,i) = a*pow(log(IND(x,i)-x0),4)+b*pow(log(IND(x,i)-x0),3) \
        +c*pow(log(IND(x,i)-x0),2)+d*log(IND(x,i)-x0)+e;
    }
  return PyArray_Return(y);
}

static char module_docstring[]="\
 NAME:\n\
	LOGRAMP\n\
\n\
 PURPOSE:\n\
	This function creates a model that fits a natural log + linear ramped eclipse\n\
\n\
 CATEGORY:\n\
	Astronomy.\n\
\n\
 CALLING SEQUENCE:\n\
\n\
	Result = LOGRAMP([midpt,width,depth,x12,x34,x0,b,c],x)\n\
\n\
 INPUTS:\n\
    	midpt:	Midpoint of eclipse\n\
	width:	Eclipse durations\n\
	depth:	Depth of eclipse\n\
	x12:	Ingress time\n\
	x34:	Egress time\n\
	x0:	time offset\n\
	b:	x constant\n\
	c:	x=0 offset\n\
	x:	Array of time/phase points\n\
\n\
 OUTPUTS:\n\
	This function returns an array of y values by combining an eclipse and the ramp model\n\
\n\
 PROCEDURE:\n\
\n\
 EXAMPLE:\n\
\n\
\n\
\n\
 MODIFICATION HISTORY:\n\
	Written by:	Kevin Stevenson, UCF\n\n\
                kevin218@knights.ucf.edu\n\
    \n\
    2008-06-26  Original creation \n\
	\n\
	2018-11-22  Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\
";

static PyMethodDef module_methods[] = {
  {"logramp",(PyCFunction)logramp,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_logramp(void)
#else
    initlogramp(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "logramp",             /* m_name */
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
        PyObject *m = Py_InitModule3("logramp", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
