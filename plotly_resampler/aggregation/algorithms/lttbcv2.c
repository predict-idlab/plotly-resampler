#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>  // pull the python API into the current namespace
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>


// method below assumes the x-delta's are equidistant
static PyObject* downsample_return_index(PyObject *self, PyObject *args) {
    int threshold;
    PyObject *x_obj = NULL, *y_obj = NULL;
    PyArrayObject *x_array = NULL, *y_array = NULL;

    if (!PyArg_ParseTuple(args, "OOi", &x_obj, &y_obj, &threshold))
        return NULL;

    if ((!PyArray_Check(x_obj) && !PyList_Check(y_obj))) {
        PyErr_SetString(PyExc_TypeError, "Function requires x and y input to be of type list or ndarray ...");
        goto fail;
    }


    // Interpret the input objects as numpy arrays, with reqs (contiguous, aligned, and writeable ...)
    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (x_array == NULL || y_array == NULL) {
        goto fail;
    }

    if (PyArray_NDIM(x_array) != 1 || PyArray_NDIM(y_array) != 1) {;
        PyErr_SetString(PyExc_ValueError, "Y must have a single dimension ...");
        goto fail;
    }

    if (!PyArray_SAMESHAPE(x_array, y_array)) {
        PyErr_SetString(PyExc_ValueError, "Input x and y must have the same shape ...");
        goto fail;
    }

    // Declare data length and check if we actually have to downsample!
    const Py_ssize_t data_length = (Py_ssize_t)PyArray_DIM(y_array, 0);
    if (threshold >= data_length || threshold <= 0) {
        // Nothing to do!
        PyObject *value = Py_BuildValue("O", x_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return value;
    }

    // Access the data in the NDArray!
    double *y = (double*)PyArray_DATA(y_array);

    // Create an empty output array with shape and dim for the output!
    npy_intp dims[1];
    dims[0] = threshold;
    PyArrayObject *sampled_x = (PyArrayObject *)PyArray_Empty(1, dims,
        PyArray_DescrFromType(NPY_INT64), 0);

    // Get a pointer to its data
    long long *sampled_x_data = (long long*)PyArray_DATA(sampled_x);

    // The main loop here!
    Py_ssize_t sampled_index = 0;
    const double every = (double)(data_length - 2) / (threshold - 2);

    Py_ssize_t a = 0;
    Py_ssize_t next_a = 0;

    // Always add the first point!
    sampled_x_data[sampled_index] = 0;

    sampled_index++;
    Py_ssize_t i;
    for (i = 0; i < threshold - 2; ++i) {
        // Calculate point average for next bucket (containing c)
        double avg_x = threshold;
        double avg_y = 0;
        Py_ssize_t avg_range_start = (Py_ssize_t)(floor((i + 1)* every) + 1);
        Py_ssize_t avg_range_end = (Py_ssize_t)(floor((i + 2) * every) + 1);
        if (avg_range_end >= data_length){
            avg_range_end = data_length;
        }
        Py_ssize_t avg_range_length = avg_range_end - avg_range_start;

        for (;avg_range_start < avg_range_end; avg_range_start++){
            avg_y += y[avg_range_start];
        }
        avg_y /= avg_range_length;
        avg_x = avg_range_start + every / 2;

        // Get the range for this bucket
        Py_ssize_t range_offs = (Py_ssize_t)(floor((i + 0) * every) + 1);
        Py_ssize_t range_to = (Py_ssize_t)(floor((i + 1) * every) + 1);


        // Point a
        double point_a_y = y[a];

        double max_area = -1.0;
        for (; range_offs < range_to; range_offs++){
            // Calculate triangle area over three buckets
            double area = fabs((a - avg_x) * (y[range_offs] - point_a_y) - (a - range_offs) * (avg_y - point_a_y)) * 0.5;
            if (area > max_area){
                max_area = area;
                next_a = range_offs; // Next a is this b
            }
        }
        // Pick this point from the bucket
        sampled_x_data[sampled_index] = next_a;

        sampled_index++;

        // Current a becomes the next_a (chosen b)
        a = next_a;
    }

    // Always add last! Check for finite values!
    sampled_x_data[sampled_index] = data_length - 1;

    // Provide our return value
    PyObject *value = Py_BuildValue("O", sampled_x);

    // And remove the references!
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_XDECREF(sampled_x);

    return value;

fail:
    Py_DECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
}


// Method definition object
static PyMethodDef lttbcv2Methods[] = {
    {
        "downsample_return_index",  // The name of the method
        downsample_return_index,    // Function pointer to the method implementation
        METH_VARARGS,
        "Compute the largest triangle three buckets (LTTB) algorithm in a C extension."
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef lttbc_module_definition = {
    PyModuleDef_HEAD_INIT,
    "lttbcv2",  /* name of module */
    "A Python module that computes the largest triangle three buckets algorithm (LTTB) using C code.",
    -1,         /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
    lttbcv2Methods
};

// Module initialization
PyMODINIT_FUNC PyInit_lttbcv2(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&lttbc_module_definition);
}
