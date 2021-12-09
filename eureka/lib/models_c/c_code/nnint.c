#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))
#define IND_int(a,i) *((int *)(a->data+i*a->strides[0]))
#define IND2(a,i,j) *((double *)(a->data+i*a->strides[0]+j*a->strides[1]))
#define IND2_int(a,i,j) *((int *)(a->data+i*a->strides[0]+j*a->strides[1]))

static PyObject *nnint(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *nnint(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *posflux, *retbinflux, *retbinstd, *issmoothing;
  PyObject *wbfipmask;
  PyArrayObject *x, *y, *flux, *binfluxmask, *kernel, *binloc, *dydx, *etc, *ipparams;
  //need to make some temp tuples to read in from argument list, then parse
  // a,a,a,a,a dtype=int,a,tuple[d,d,d,d],array[a,a]dtyp=int,array[a,a,a,a],tuple[int,int],bool
  PyObject *tup1, *tup2;
  
  //initialize the keywords
  retbinflux = Py_False;
  retbinstd  = Py_False;

  //make the keywords list
  static char *kwlist[] = {"ipparams","posflux","etc","retbinflux","retbinstd",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|OOO",kwlist,&ipparams,&posflux\
                                  ,&etc,&retbinflux,&retbinstd))
    {
      return NULL;
    }


  //now we must break appart the posflux tuple
  y           = (PyArrayObject *) PyList_GetItem(posflux,0);
  x           = (PyArrayObject *) PyList_GetItem(posflux,1);
  flux        = (PyArrayObject *) PyList_GetItem(posflux,2);
  wbfipmask   =                   PyList_GetItem(posflux,3);
  binfluxmask = (PyArrayObject *) PyList_GetItem(posflux,4);
  kernel      = (PyArrayObject *) PyList_GetItem(posflux,5);
  tup1        =                   PyList_GetItem(posflux,6);
  binloc      = (PyArrayObject *) PyList_GetItem(posflux,7);
  dydx        = (PyArrayObject *) PyList_GetItem(posflux,8);
  tup2        =                   PyList_GetItem(posflux,9);
  issmoothing =                   PyList_GetItem(posflux,10);

  //create the arrays the will be returned, under various conditions
  PyArrayObject *output, *binflux, *binstd, *tempwbfip;
  npy_intp dims[1];
  
  dims[0] = flux->dimensions[0];
  output  = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  dims[0] = PyList_Size(wbfipmask);
  binflux = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);
  binstd  = (PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_DOUBLE);

  int dis = binfluxmask->dimensions[0];
  int i,j,arsize,temp_int,counter;
  double temp_mean,temp_std,meanbinflux;

  //need to make a lock to deal with the menbinflux variable
  omp_lock_t lck;
  omp_init_lock(&lck);

  counter = 0;
  meanbinflux = 0;
  // remind kevin to make all wbfipmask things arrays
  // shared(lck,meanbinflux,counter) 
  #pragma omp parallel for private(j,tempwbfip, arsize,temp_mean,temp_std,temp_int)
  for(i = 0; i<dis;i++)
    {
      if(IND_int(binfluxmask,i) == 1)
        {
          if(PyObject_IsTrue(retbinstd) == 1)
            {
              tempwbfip = (PyArrayObject *) PyList_GetItem(wbfipmask,i);
              arsize = tempwbfip->dimensions[0];
              temp_mean = 0;
              temp_std  = 0;
              for(j=0;j<arsize;j++)
                {
                  temp_int   = IND_int(tempwbfip,j); 
                  temp_mean += (IND(flux,temp_int)/IND(etc,temp_int));
                }
              temp_mean /= (double) arsize;
              
              for(j=0;j<arsize;j++)
                {
                  temp_int  = IND_int(tempwbfip,j);
                  temp_std += pow(((IND(flux,temp_int)/IND(etc,temp_int))\
                                   -temp_mean),2);
                }
              temp_std /= (double) arsize;
              temp_std = sqrt(temp_std);

              IND(binflux,i) = temp_mean;
              IND(binstd,i)  = temp_std;
              
              omp_set_lock(&lck);
              meanbinflux += temp_mean;
              counter += 1;
              omp_unset_lock(&lck);
            }
          else
            {
              tempwbfip = (PyArrayObject *) PyList_GetItem(wbfipmask,i);
              arsize = tempwbfip->dimensions[0];
              temp_mean = 0;
              for(j=0;j<arsize;j++)
                {
                  temp_int   = IND_int(tempwbfip,j);
                  temp_mean += (IND(flux,temp_int)/IND(etc,temp_int));
                }
              temp_mean /= (double) arsize;
              IND(binflux,i) = temp_mean;

              omp_set_lock(&lck);
              meanbinflux += temp_mean;
              counter     += 1;
              omp_unset_lock(&lck);
            }
        }
      else
        {
          IND(binflux,i) = 0;
          IND(binstd, i) = 0;
        }
    }
  meanbinflux /= (double) counter;

  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
      IND(binflux,i) /= meanbinflux;
      IND(binstd, i) /= meanbinflux;
    }
  
  
  dims[0] = flux->dimensions[0];
  #pragma omp parallel for
  for(i=0;i<dims[0];i++)
    {
			temp_int = IND2_int(binloc,0,i);
		  IND(output,i) = IND(binflux,temp_int);
    }
  
  if(PyObject_IsTrue(retbinflux) == 0 && PyObject_IsTrue(retbinstd) == 0)
    {
      Py_XDECREF(binflux);
      Py_XDECREF(binstd);
      return PyArray_Return(output);
    }
  else if (PyObject_IsTrue(retbinflux) == 1 && PyObject_IsTrue(retbinstd)==1)
    {
      return Py_BuildValue("NNN",output,binflux,binstd);
    }
  else if (PyObject_IsTrue(retbinflux) == 1)
    {
      Py_XDECREF(binstd);
      return Py_BuildValue("NN",output,binflux);
    }
  else
    {
      Py_XDECREF(binflux);
      return Py_BuildValue("NN",output,binstd);
    }
}

static char module_docstring[]="\
  This function fits the intra-pixel sensitivity effect using the mean \n\
   within a given binned position (nearest-neighbor interpolation).\n\
\n\
  Parameters\n\
  ----------\n\
    ipparams :  tuple\n\
                unused\n\
    y :         1D array, size = # of measurements\n\
                Pixel position along y\n\
    x :         1D array, size = # of measurements\n\
                Pixel position along x\n\
    flux :      1D array, size = # of measurements\n\
                Observed flux at each position\n\
    wherebinflux :  1D array, size = # of bins\n\
                    Measurement number assigned to each bin\n\
    gridpt :    1D array, size = # of measurements        \n\
        \n\
  Returns\n\
  -------\n\
    1D array, size = # of measurements\n\
    Normalized intrapixel-corrected flux multiplier        \n\
\n\
  Revisions\n\
  ---------\n\
    2010-06-07  Kevin Stevenson, UCF  \n\
                kevin218@knights.ucf.edu\n\
                Original version\n\
    2010-07-07  Kevin\n\
                Added wbfipmask\n\
    2011-01-06  nate lust, ucf\n\
                natelust at linux dot com\n\
                converted to c extension function\n\
    2018-11-27  Jonathan Fraine, SSI\n\
                jfraine at spacescience.org\n\
                Updated c extensions to python3, with support for python2.7\n\
";

static PyMethodDef module_methods[] = {
  {"nnint",(PyCFunction)nnint,METH_VARARGS|METH_KEYWORDS,module_docstring},{NULL}};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
    PyInit_nnint(void)
#else
    initnnint(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module;
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "nnint",             /* m_name */
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
        PyObject *m = Py_InitModule3("nnint", module_methods, module_docstring);
        if (m == NULL)
            return;
        /* Load `numpy` functionality. */
        import_array();
    #endif
}
