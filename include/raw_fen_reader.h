#pragma once

#include <optional>
#include <string>

#include <file_reader_iterator.h>
#include <training.h>

namespace train{

struct raw_fen_reader{
  using iterator = file_reader_iterator<state_type>;

  std::optional<size_t> memoi_size_{std::nullopt};
  std::string path_;

  file_reader_iterator<state_type> begin() const { return file_reader_iterator<state_type>(to_line_reader<state_type>(state_type::parse_fen), path_); }
  file_reader_iterator<state_type> end() const { return file_reader_iterator<state_type>(); }


  size_t const_size() const {
    size_t size_{};
    for(const auto& _ : *this){ (void)_; ++size_; }
    return size_;
  }  

  size_t size(){
    if(memoi_size_.has_value()){ return memoi_size_.value(); }
    size_t size_ = const_size();
    memoi_size_ = size_;
    return size_;
  }

  raw_fen_reader(const std::string& path): path_{path} {}
};

}