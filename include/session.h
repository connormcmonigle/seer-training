#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <filesystem>

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

  size_t size(){
    if(memoi_size_.has_value()){ return memoi_size_.value(); }
    size_t size_{};
    for(const auto& _ : *this){ (void)_; ++size_; }
    memoi_size_ = size_;
    return size_;
  }

  raw_fen_reader(const std::string& path): path_{path} {}
};


struct session{
  std::string root_;
  train_interface<real_type> interface_{};

  void load_weights(const std::string& path){ interface_.load_weights(path); }

  sample_writer get_n_man_train_writer(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    return sample_writer(train_path);
  }

  raw_fen_reader get_n_man_raw_reader(const size_t& n) const {
    return raw_fen_reader(raw_n_man_path(root_, n));
  }

  sample_reader get_n_man_train_reader(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    if(std::filesystem::exists(std::filesystem::path{}.assign(train_path))){ return sample_reader(train_path); }

    assert((n > min_base_cardinality));

    const std::string raw_path = raw_n_man_path(root_, n);

    sample_writer writer(train_path);
    for(const state_type& elem : raw_fen_reader(raw_path)){
      const std::optional<state_type> link_position = interface_.get_continuation(elem);
      if(link_position.has_value()){
        const wdl_type wdl = interface_.get_wdl(link_position.value());
        const bool same_pov = (link_position -> turn() == elem.turn());
        writer.append_sample(sample{elem, same_pov ? wdl : mirror_wdl(wdl)});
      }
    }

    return sample_reader(train_path);
  }

  session(const std::string& root) : root_{root} {}
};

}