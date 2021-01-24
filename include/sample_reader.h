#pragma once

#include <optional>
#include <string>

#include <file_reader_iterator.h>
#include <sample.h>

namespace train{

struct sample_reader{
  std::optional<size_t> memoi_size_{std::nullopt};
  std::string path_;

  file_reader_iterator<sample> begin() const { return file_reader_iterator<sample>(to_line_reader<sample>(sample::from_string), path_); }
  file_reader_iterator<sample> end() const { return file_reader_iterator<sample>(); }

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

  sample_reader(const std::string& path) : path_{path} {}
};

}