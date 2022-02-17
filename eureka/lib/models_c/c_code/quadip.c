#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))
//#define IND_arr(a,i) (PyArrayObject *)(a->data+i*a->strides[0])
#define IND2(a,i,j) *((double *)(a->data+i*a->strides[0]+j*a->strides[1]))

static PyObject *quadip(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *quadip(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *out, *ipparams, *position;
  double a,b,c,d,e,f;
  int i;
  npy_intp dims[1];

  static char *kwlist[] = {"ipparams","position","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&ipparams,&position,&etc))
    {
      return NULL;
    }

  a = IND(ipparams,0);
  b = IND(ipparams,1);
  c = IND(ipparams,2);
  d = IND(ipparams,3);
  e = IND(ipparams,4);
  f = IND(ipparams,5);

  dims[0] = PyArray_DIM(position, 1);

  out = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  
  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(out,i) = a*pow(IND2(position,0,i),2)+b*pow(IND2(position,1,i),2)+  \
                   c*IND2(position,0,i)*IND2(position,1,i)+d*IND2(position,0,i)+e*IND2(position,1,i)+f;
    }
  return PyArray_Return(out);
}

static char module_docstring[]="\
  This function fits the intra-pixel sensitivity effect using a 2D quadratic.\n\
\n\
  Parameters\n\
  ----------\n\
    a: quadratic coefficient in y\n\
    b: quadratic coefficient in x\n\
    c: coefficient for cross-term\n\
    d: linear coefficient in y\n\
    e: linear coefficient in x\n\
    f: constant\n\
\n\
  Returns\n\
  -------\n\
    returns the flux values for the intra-pixel model\n\
\n\
  Revisions\n\
  ---------\n\
  2008-07-05	Kevin Stevenson, UCF  \n\
			    kevin218@knights.ucf.edu\n\
		        Original version\n\
  2011-01-05    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Converted to c extention function\n\
  2018-11-22    Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\
\n\
";

static PyMethodDef module_methods[] = {
  {"quadip",(PyCFunction)quadip,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_quadip(void)
#else
    initquadip(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "quadip",             /* m_name */
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
        PyObject *m = Py_InitModule3("quadip", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
