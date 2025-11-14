#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gaClique.h"

namespace py = pybind11;
PYBIND11_MODULE(gaclique, m) {
    m.def("run_max_clique", &run_max_clique);
}
