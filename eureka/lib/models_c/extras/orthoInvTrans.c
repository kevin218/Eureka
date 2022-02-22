#include<Python.h>
#include<numpy/arrayobject.h>
#include<math.h>
#include<omp.h>

#define IND(a,i) *((double *)(a->data+i*a->strides[0]))
#define IND2(a,i,j) *((double *)(a->data+i*a->strides[0]+j*a->strides[1]))

static PyObject *orthoInvTrans(PyObject *self, PyObject *args, PyObject *keywds);

static PyObject *orthoInvTrans(PyObject *self, PyObject *args, PyObject *keywds)
{
  // PyObject *etc;
  PyArrayObject *params, *newparams, *invtrans, *etc;
  //double goal,r0,r1,sint,cost,pm;
  int i,j;
  npy_intp tdims[2], pdims[1];

  //  etc = PyList_New(0);

  static char *kwlist[] = {"params","invtrans","etc",NULL};

  if(!PyArg_ParseTupleAndKeywords(args,keywds,"OO|O",kwlist,&params,&invtrans,&etc))
    {
      return NULL;
    }

  //goal = IND(rampparams,0);

  tdims[0] = invtrans->dimensions[0];
  tdims[1] = invtrans->dimensions[1];
  //printf("%i, %i\n",tdim0,tdim1);
  pdims[0] = params->dimensions[0];
  //pdims[1] = params->dimensions[1];

  newparams = (PyArrayObject *) PyArray_SimpleNew(1,pdims,PyArray_DOUBLE);

  //#pragma omp parallel for
  for(i=0;i<tdims[0];i++)
  {
    IND(newparams,i)=0;
    //IND2(newparams,i,j)=0;
    for(j=0;j<tdims[1];j++)
    {
        IND(newparams,i) += IND2(invtrans,i,j)*IND(params,j);
        //IND2(newparams,i,k)+= IND2(invtrans,i,j)*IND2(params,j,k);
    }
    IND(newparams,i) += IND(etc,i);
  }
  
  return PyArray_Return(newparams);
}

static char orthoInvTrans_doc[]="\
  This function uses principal component analysis to modify parameter values.\n\
\n\
  Parameters\n\
  ----------\n\
    params:     Array of parameters to be modified\n\
    invtrans:   Inverse transformation matrix, np.matrix() type\n\
    origin:	    Array of len(params) indicating the reference frame origin\n\
\n\
  Returns\n\
  -------\n\
    This function returns the modified parameter values\n\
\n\
  Revisions\n\
  ---------\n\
  2011-07-25	Kevin Stevenson, UCF  \n\
			kevin218@knights.ucf.edu\n\
		Original version\n\
";

static PyMethodDef orthoInvTrans_methods[] = {
  {"orthoInvTrans",(PyCFunction)orthoInvTrans,METH_VARARGS|METH_KEYWORDS,orthoInvTrans_doc},{NULL}};

void initorthoInvTrans(void)
{
  Py_InitModule("orthoInvTrans",orthoInvTrans_methods);
  import_array();
}
