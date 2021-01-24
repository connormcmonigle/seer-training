#pragma once

#include <string>
#include <fstream>

#include <sample.h>

namespace train{

struct sample_writer{
  std::string path_;
  std::ofstream file_;

  sample_writer& append_sample(const sample& datum){
    file_ << datum.to_string() << '\n';
    return *this;
  }

  sample_writer(const std::string& path) : path_{path}, file_{path} {}
};

}