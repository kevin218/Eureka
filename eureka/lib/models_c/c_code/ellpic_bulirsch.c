#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>
#include<stdio.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))

static PyObject *ellpic_bulirsch(PyObject *self, PyObject *args);

static PyObject *ellpic_bulirsch(PyObject *self, PyObject *args)
{
  PyArrayObject *k,*n,*output;
  int i,isdone;
  double foo,max;
  npy_intp dims[1];

  if(!PyArg_ParseTuple(args,"OO", &n, &k))
    {
      return NULL;
    }
  
  dims[0] = k->dimensions[0];
  output  = (PyArrayObject *) PyArray_SimpleNew(1,dims,PyArray_DOUBLE);
  double kc[dims[0]],p[dims[0]],c[dims[0]],d[dims[0]],e[dims[0]],f[dims[0]],g[dims[0]],m0[dims[0]];
  
  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      kc[i] = sqrt(1.-pow(IND(k,i),2)); 
      e[i]  = kc[i];
      p[i]  = sqrt(IND(n,i)+1.);
      d[i]  = 1./p[i];
      c[i]  = 1.;
      m0[i] = 1.;
    }
  
  isdone = 0;
  while (isdone == 0)
    {
      #pragma omp parallel for
      for(i=0;i<dims[0];i++)
        {
          f[i]  = c[i];
          c[i]  = d[i]/p[i]+c[i];
          g[i]  = e[i]/p[i]; 
          d[i]  = 2.*(f[i]*g[i]+d[i]);
          p[i]  = g[i] + p[i]; 
          g[i]  = m0[i]; 
          m0[i] = kc[i] + m0[i];
        }
      max = fabs(1.-kc[0]/g[0]);
      #pragma omp parallel for private(foo)
      for(i=1;i<dims[0];i++)
        {
          foo = fabs(1.-kc[i]/g[i]);
          if (foo > max)
              max = foo;
        }
      if (max > 1.e-8)
        {
          #pragma omp parallel for
          for(i=0;i<dims[0];i++)
            {
              kc[i] = 2*sqrt(e[i]);
              e[i]  = kc[i]*m0[i];
            }
        }
      else
        {
          #pragma omp parallel for
          for(i=0;i<dims[0];i++)
              IND(output,i) = 0.5*M_PI*(c[i]*m0[i]+d[i])/(m0[i]*(m0[i]+p[i]));
          isdone = 1;
        }
    }
  
  return PyArray_Return(output);
}

static char module_docstring[]="\
   Computes the complete elliptical integral of the third kind using\n\
   the algorithm of Bulirsch (1965).\n\
\n\
   Parameters\n\
   ----------\n\
   n:       1D NPY ARRAY - contains values from trquad.py\n\
   k:       1D NPY ARRAY - contains values from trquad.py\n\
\n\
   Returns\n\
   -------\n\
   output:  1D NPY ARRAY - \n\
\n\
   Revisions\n\
   ---------\n\
   Original version by Jason Eastman\n\
   2012-08-25   Kevin Stevenson, UChicago \n\
                kbs@uchicago.edu\n\
                Converted from Python\n\n\
   2018-11-22   Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\n\
";

static PyMethodDef module_methods[] = {
  {"ellpic_bulirsch", ellpic_bulirsch,METH_VARARGS,module_docstring},{NULL}};

// static char module_docstring[] =
// "This module is used to calcuate the ellpic_bulirsch";

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_ellpic_bulirsch(void)
#else
    initellpic_bulirsch(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "ellpic_bulirsch",             /* m_name */
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
        PyObject *m = Py_InitModule3("ellpic_bulirsch", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
