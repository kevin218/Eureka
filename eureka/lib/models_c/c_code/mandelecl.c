#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *mandelecl(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *mandelecl(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *etc;
  PyArrayObject *t, *y, *eclparams;
  npy_intp dims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"eclparams","t","etc",NULL};
  
  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O"\
				  ,kwlist,&eclparams,&t,&etc))
    {
      return NULL;
    }
 
  double midpt,width,depth,t12,t34,flux;
  double t1,t2,t3,t4,p,z,k0,k1;

  midpt = IND(eclparams,0);
  width = IND(eclparams,1);
  depth = IND(eclparams,2);
  t12   = IND(eclparams,3);
  t34   = IND(eclparams,4);
  flux  = IND(eclparams,5);

  if(depth == 0)
    {
      dims[0] = t->dimensions[0];
      y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
      int i;
      for(i=0;i<dims[0];i++)
	{
	  IND(y,i) = flux;
	}
      return PyArray_Return(y);
    }
  
  //Compute Time of Contact Points
  t1    = midpt -width/2;

  if((t1+t12)<midpt)
    t2 = t1+t12;
  else
    t2 = midpt;

  t4    = midpt +width/2;
  if((t4-t34) > midpt)
    t3  = t4-t34;
  else
    t3  = midpt;

  p = sqrt(fabs(depth))*(depth/fabs(depth));
  dims[0] = t->dimensions[0];
 
  y = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  
  int i;
  #pragma omp parallel for private(z,k0,k1)
  for(i=0;i<dims[0];i++)
    {
      IND(y,i) = 1;
      if(IND(t,i)>=t2&&IND(t,i)<=t3)
	{
	  IND(y,i) = 1 - depth;
	}
      else if(p != 0)
	{
	  // Use Mandel & agol (2002) for ingress of eclipse
	  if(IND(t,i)>t1&&IND(t,i)<t2)
	    {
	      z  = -2*p*(IND(t,i)-t1)/t12 +1+p;
	      k0 = acos((p*p+z*z-1)/2/p/z);
	      k1 = acos((1-p*p+z*z)/2/z);
	      IND(y,i) = 1-depth/fabs(depth)/M_PI*(p*p*k0+k1
			  -sqrt((4*z*z-pow((1+z*z-p*p),2))/4));
	    }
	  else if(IND(t,i)>t3&&IND(t,i)<t4)
	    {
	      z  = 2*p*(IND(t,i)-t3)/t34+1-p;
	      k0 = acos((p*p+z*z-1)/2/p/z);
	      k1 = acos((1-p*p+z*z)/2/z);
	      IND(y,i) = 1-depth/fabs(depth)/M_PI*(p*p*k0+k1
			  - sqrt((4*z*z-pow((1+z*z-p*p),2))/4));
	    }
	  
	}
      IND(y,i) *= flux;

    }
  Py_XDECREF(dims);
  return PyArray_Return(y);
}

static char module_docstring[] ="\
  This function computes the secondary eclipse shape using equations provided by Mandel & Agol (2002)\n\
\n\
  Parameters\n\
  ----------\n\
    midpt:  Center of eclipse\n\
    width:  Eclipse duration from contacts 1 to 4\n\
    depth:  Eclipse depth\n\
    t12:    Eclipse duration from contacts 1 to 2\n\
    t34:    Eclipse duration from contacts 3 to 4\n\
    flux:   Flux offset from 0\n\
    t:	    Array of phase points\n\
\n\
  Returns\n\
  -------\n\
    This function returns the flux for each point in t.\n\
\n\
  Revisions\n\
  ---------\n\
  2008-05-08	Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
  2010-12-19    Nate Lust, UCF\n\
                natelust at linux dot com\n\
                Changed to c extention function\n\
  2018-11-22    Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\
";

static PyMethodDef module_methods[]={
  {"mandelecl",(PyCFunction)mandelecl,METH_VARARGS|METH_KEYWORDS,module_docstring}, {NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_mandelecl(void)
#else
    initmandelecl(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "mandelecl",             /* m_name */
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
        PyObject *m = Py_InitModule3("mandelecl", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
