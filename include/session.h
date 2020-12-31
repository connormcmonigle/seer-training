#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <filesystem>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <tuple>

#include <training.h>
#include <sample.h>


namespace train{

constexpr size_t min_base_cardinality = 4;
constexpr std::string_view path_delimiter = "/";
constexpr std::string_view extension = ".txt";
constexpr std::string_view train_dir = "train";
constexpr std::string_view raw_dir = "raw";

constexpr size_t half_feature_numel(){ return half_feature_numel_; }
constexpr wdl_type known_win_value(){ return win; }
constexpr wdl_type known_draw_value(){ return draw; }
constexpr wdl_type known_loss_value(){ return loss; }


std::string raw_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(raw_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

std::string train_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(train_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

struct raw_fen_reader{
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


struct session{
  size_t concurrency_{1};
  std::string root_;
  train_interface<real_type> interface_{};

  size_t concurrency() const { return concurrency_; }
  
  session& set_concurrency(const size_t& val){
    assert(val >= 1);
    concurrency_ = val;
    return *this;
  }

  void load_weights(const std::string& path){ interface_.load_weights(path); }


  sample_writer get_n_man_train_writer(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    return sample_writer(train_path);
  }

  raw_fen_reader get_n_man_raw_reader(const size_t& n) const {
    return raw_fen_reader(raw_n_man_path(root_, n));
  }

  std::tuple<size_t, size_t> calc_range_(const size_t& raw_size, const size_t& thread_idx) const {
    const size_t thread_count = concurrency();
    const size_t segment_size = raw_size / thread_count;
    assert(thread_idx < thread_count);
    const bool is_last = thread_idx == thread_count - 1;

    return is_last ? 
      std::tuple(thread_idx * segment_size, raw_size) : 
      std::tuple(thread_idx * segment_size, thread_idx * segment_size + segment_size);
  }

  void generate_links_for_(const size_t& n){
    constexpr size_t max_buffer_size = 512;

    const auto reader = get_n_man_raw_reader(n);
    const size_t raw_size = reader.const_size();

    std::mutex writer_mutex_;
    auto writer = get_n_man_train_writer(n);

    auto generate_link_range = [&, this](const size_t& first, const size_t& last){
      std::vector<sample> buffer{};
      
      auto flush_buffer = [&, this]{
        std::lock_guard<std::mutex> writer_lock(writer_mutex_);
        for(const auto& train_sample : buffer){ writer.append_sample(train_sample); }
        buffer.clear();
      };

      auto it = reader.begin();
      for(size_t i{0}; i < first; ++i){ ++it; }

      for(size_t idx{first}; idx < last; ++idx, ++it){
        const state_type elem = *it;
        const std::optional<continuation_type> continuation = interface_.get_continuation(elem);
        if(continuation.has_value()){
          const wdl_type wdl = interface_.get_wdl(continuation.value());
          const bool same_pov = ((continuation -> state()).turn() == elem.turn());
          buffer.push_back(sample{elem, same_pov ? wdl : mirror_wdl(wdl)});
        }

        if(buffer.size() >= max_buffer_size){ flush_buffer(); }
      }

      flush_buffer();
    };

    std::vector<std::thread> threads{};
    for(size_t i(0); i < concurrency(); ++i){
      const auto [first, last] = calc_range_(raw_size, i);
      threads.emplace_back(generate_link_range, first, last);
    }

    for(auto& th : threads){ th.join(); }
  }

  sample_reader get_n_man_train_reader(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    if(std::filesystem::exists(std::filesystem::path{}.assign(train_path))){ return sample_reader(train_path); }
    assert((n > min_base_cardinality));
    generate_links_for_(n);
    return sample_reader(train_path);
  }

  session(const std::string& root) : root_{root} {}
};

}
