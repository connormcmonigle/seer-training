#pragma once

#include <optional>
#include <string>

#include <line_count_size.h>
#include <file_reader_iterator.h>
#include <training.h>


namespace train{

struct raw_fen_reader : line_count_size<raw_fen_reader> {
  using iterator = file_reader_iterator<state_type>;

  std::string path_;

  file_reader_iterator<state_type> begin() const { return file_reader_iterator<state_type>(to_line_reader<state_type>(state_type::parse_fen), path_); }
  file_reader_iterator<state_type> end() const { return file_reader_iterator<state_type>(); }

  raw_fen_reader(const std::string& path): path_{path} {}
};

}