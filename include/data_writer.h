#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <optional>

#include <sample_writer.h>

namespace train{

struct data_writer{
  std::mutex writer_mutex_;
  sample_writer writer_;

  std::uint64_t total_;
  std::atomic_uint64_t completed_;

  bool is_complete() {
    std::lock_guard<std::mutex> lck(writer_mutex_);
    return completed_ >= total_;
  }

  std::tuple<std::uint64_t, std::uint64_t> progress() const {
    return std::tuple(completed_.load(), total_);
  }

  data_writer& write_block(const std::vector<sample>& data){
    std::lock_guard<std::mutex> lck(writer_mutex_);
    for(const auto& elem : data){
      if (completed_ >= total_) { break; }
      writer_.append_sample(elem);
      ++completed_;
    }
    return *this;
  }

  data_writer(const std::string& write_path, const size_t& total) : writer_{write_path}, total_{total}, completed_{0} {}
};

}