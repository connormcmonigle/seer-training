#pragma once

#include <optional>
#include <string>

#include <line_count_size.h>
#include <file_reader_iterator.h>
#include <sample.h>

namespace train{

struct sample_reader : line_count_size<sample_reader> {
  using iterator = file_reader_iterator<sample>;
  std::string path_;

  file_reader_iterator<sample> begin() const { return file_reader_iterator<sample>(to_line_reader<sample>(sample::from_string), path_); }
  file_reader_iterator<sample> end() const { return file_reader_iterator<sample>(); }

  sample_reader(const std::string& path) : path_{path} {}
};

}