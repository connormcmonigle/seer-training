#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <optional>

#include <raw_fen_reader.h>
#include <sample_writer.h>

namespace train{

template<size_t work_block_size>
struct data_controller{

  std::mutex reader_mutex_;
  std::mutex writer_mutex_;

  raw_fen_reader reader_;
  sample_writer writer_;
  std::uint64_t total_;

  std::atomic_uint64_t completed_;
  raw_fen_reader::iterator iter_;

  std::tuple<std::uint64_t, std::uint64_t> progress() const {
    return std::tuple(completed_.load(), total_);
  }

  std::optional<std::vector<state_type>> read_block(){
    std::lock_guard<std::mutex> lck(reader_mutex_);
    if(iter_ == reader_.end()){ return std::nullopt; }
    std::vector<state_type> result{};
    for(size_t i(0); i < work_block_size && iter_ != reader_.end(); ++i, ++iter_){
      ++completed_;
      result.push_back(*iter_);
    }
    return result;
  }

  data_controller<work_block_size>& write_block(const std::vector<sample>& data){
    std::lock_guard<std::mutex> lck(writer_mutex_);
    for(const auto& elem : data){
      writer_.append_sample(elem);
    }
    return *this;
  }

  data_controller(const raw_fen_reader& reader, const std::string& write_path) : reader_{reader}, writer_{write_path} {
    total_ = reader_.size();
    iter_ = reader_.begin();
  }
};

}