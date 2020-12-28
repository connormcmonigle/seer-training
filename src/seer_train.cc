#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sample.h>
#include <session.h>

namespace py = pybind11;

PYBIND11_MODULE(seer_train, m){
  m.def("half_feature_numel", train::half_feature_numel);

  m.def("known_win_value", train::known_win_value);
  m.def("known_draw_value", train::known_draw_value);
  m.def("known_loss_value", train::known_loss_value);

  m.def("raw_n_man_path", train::raw_n_man_path);
  m.def("train_n_man_path", train::train_n_man_path);

  py::class_<train::feature_set>(m, "FeatureSet")
    .def_readonly("white", &train::feature_set::white)
    .def_readonly("black", &train::feature_set::black);

  py::class_<train::state_type>(m, "StateType")
    .def_static("parse_fen", train::state_type::parse_fen)
    .def_static("start_pos", train::state_type::start_pos)
    .def("fen", &train::state_type::fen);

  py::class_<train::sample>(m, "Sample")
    .def(py::init<const train::state_type&, const train::wdl_type&>())
    .def("features", &train::sample::features)
    .def("pov", &train::sample::pov)
    .def("win", &train::sample::win)
    .def("draw", &train::sample::draw)
    .def("loss", &train::sample::loss)
    .def("to_string", &train::sample::to_string);

  py::class_<train::sample_writer>(m, "SampleWriter")
    .def(py::init<const std::string&>())
    .def("append_sample", &train::sample_writer::append_sample);

  py::class_<train::raw_fen_reader>(m, "RawFenReader")
    .def(py::init<const std::string&>())
    .def("size", &train::raw_fen_reader::size)
    .def("__iter__", [](const train::raw_fen_reader& r) { return py::make_iterator(r.begin(), r.end()); }, py::keep_alive<0, 1>());


  py::class_<train::sample_reader>(m, "SampleReader")
    .def(py::init<const std::string&>())
    .def("size", &train::sample_reader::size)
    .def("__iter__", [](const train::sample_reader& r) { return py::make_iterator(r.begin(), r.end()); }, py::keep_alive<0, 1>());

  py::class_<train::session>(m, "Session")
    .def(py::init<const std::string&>())
    .def("load_weights", &train::session::load_weights)
    .def("get_n_man_train_writer", &train::session::get_n_man_train_writer)
    .def("get_n_man_raw_reader", &train::session::get_n_man_raw_reader)
    .def("get_n_man_train_reader", &train::session::get_n_man_train_reader);


}