#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <sample.h>
#include <data_generator.h>

namespace py = pybind11;

PYBIND11_MODULE(seer_train, m){
  m.def("half_feature_numel", train::half_feature_numel);
  m.def("max_active_half_features", train::max_active_half_features);

  py::class_<train::feature_set>(m, "FeatureSet")
    .def_readonly("white", &train::feature_set::white)
    .def_readonly("black", &train::feature_set::black);

  py::class_<train::state_type>(m, "StateType")
    .def_static("parse_fen", train::state_type::parse_fen)
    .def_static("start_pos", train::state_type::start_pos)
    .def("fen", &train::state_type::fen);

  py::class_<train::sample>(m, "Sample")
    .def(py::init<const train::state_type&, const train::score_type&>())
    .def("features", &train::sample::features)
    .def("mirrored", &train::sample::mirrored)
    .def("pov", &train::sample::pov)
    .def("score", &train::sample::score)
    .def("to_string", &train::sample::to_string);

  py::class_<train::sample_writer>(m, "SampleWriter")
    .def(py::init<const std::string&>())
    .def("append_sample", &train::sample_writer::append_sample);

  py::class_<train::sample_reader>(m, "SampleReader")
    .def(py::init<const std::string&>())
    .def("size", py::overload_cast<>(&train::sample_reader::size))
    .def("__iter__", [](const train::sample_reader& r) { return py::make_iterator(r.begin(), r.end()); }, py::keep_alive<0, 1>());

  py::class_<train::data_generator>(m, "DataGenerator")
    .def(py::init<const std::string&, const size_t&, const size_t&>())
    .def("set_concurrency", &train::data_generator::set_concurrency)
    .def("set_fixed_depth", &train::data_generator::set_fixed_depth)
    .def("set_ply_limit", &train::data_generator::set_ply_limit)
    .def("set_random_ply", &train::data_generator::set_random_ply)
    .def("generate_data", &train::data_generator::generate_data);

    


}