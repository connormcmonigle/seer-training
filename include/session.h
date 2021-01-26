#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <functional>
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
#include <raw_fen_reader.h>
#include <sample_reader.h>
#include <sample_writer.h>
#include <data_controller.h>


namespace train{

constexpr size_t min_base_cardinality = 4;
constexpr std::string_view path_delimiter = "/";
constexpr std::string_view extension = ".txt";
constexpr std::string_view train_dir = "train";
constexpr std::string_view raw_dir = "raw";

constexpr size_t half_feature_numel(){ return half_feature_numel_; }
constexpr size_t max_active_half_features(){ return max_active_half_features_; }
constexpr wdl_type known_win_value(){ return win; }
constexpr wdl_type known_draw_value(){ return draw; }
constexpr wdl_type known_loss_value(){ return loss; }


std::string raw_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(raw_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

std::string train_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(train_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

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


  std::string get_n_man_train_path(const size_t& n) const {
    return train_n_man_path(root_, n);
  }

  std::string get_n_man_raw_path(const size_t& n) const {
    return raw_n_man_path(root_, n);
  }

  void generate_links_for_(const size_t& n){
    constexpr size_t work_block_size = 512;

    auto controller = data_controller<work_block_size>(
      get_n_man_raw_path(n),
      get_n_man_train_path(n)
    );

    auto generate_link_part = [&, this]{      
      for(auto block = controller.read_block(); block.has_value(); block = controller.read_block()){
        std::vector<sample> buffer{};
        for(const auto& elem : block.value()){
          const std::optional<continuation_type> continuation = interface_.get_continuation(elem);
          if(continuation.has_value()){
            const wdl_type wdl = interface_.get_wdl(continuation.value());
            const bool same_pov = ((continuation -> state()).turn() == elem.turn());
            buffer.push_back(sample{elem, same_pov ? wdl : mirror_wdl(wdl)});
          }
        }
        controller.write_block(buffer);
      }
    };

    std::vector<std::thread> threads{};
    for(size_t i(0); i < concurrency(); ++i){ threads.emplace_back(generate_link_part); }
    for(auto& th : threads){ th.join(); }
  }

  void maybe_generate_links_for(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    if(!std::filesystem::exists(std::filesystem::path{}.assign(train_path))){
      assert((n > min_base_cardinality));
      generate_links_for_(n);
    }
  }

  session(const std::string& root) : root_{root} {}
};

}
