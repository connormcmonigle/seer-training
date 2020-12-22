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

std::string raw_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(raw_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

std::string train_n_man_path(const std::string& root, const size_t& n){
  return root + std::string(path_delimiter) + std::string(train_dir) + std::string(path_delimiter) + std::to_string(n) + std::string(extension);
}

struct raw_fen_reader_iterator{
  using difference_type = long;
  using value_type = state_type;
  using pointer = const state_type*;
  using reference = const state_type&;
  using iterator_category = std::output_iterator_tag;

  bool is_end_;
  std::ifstream file_;
  state_type current_{};

  raw_fen_reader_iterator& operator++(){
    std::string fen{}; std::getline(file_, fen);
    is_end_ = !(file_);
    if(!is_end_){ current_ = state_type::parse_fen(fen); }
    return *this;
  }
  
  bool operator==(const raw_fen_reader_iterator& other) const {
    return is_end_ && other.is_end_;
  }
  
  bool operator!=(const raw_fen_reader_iterator& other) const {
    return !(*this == other);
  }

  state_type operator*() const { return current_; }

  raw_fen_reader_iterator(const std::string& path) : is_end_{false}, file_{path} { ++(*this); }

  raw_fen_reader_iterator() : is_end_{true} {}
};

struct raw_fen_reader{

  std::string path_;

  raw_fen_reader_iterator begin() const { return raw_fen_reader_iterator(path_); }
  raw_fen_reader_iterator end() const { return raw_fen_reader_iterator(); }

  raw_fen_reader(const std::string& path): path_{path} {}
};


struct session{
  std::string root_;
  train_interface<real_type> interface_{};

  void load_weights(const std::string& path){ interface_.load_weights(path); }


  sample_reader get_n_man_training_set(const size_t& n){
    const std::string train_path = train_n_man_path(root_, n);
    if(std::filesystem::exists(std::filesystem::path{}.assign(train_path))){ return sample_reader(train_path); }

    assert((n > min_base_cardinality));

    const std::string raw_path = raw_n_man_path(root_, n);

    sample_writer writer(train_path);
    for(const state_type& elem : raw_fen_reader(raw_path)){
      const std::optional<state_type> link_position = interface_.get_continuation(elem);
      if(link_position.has_value()){
        const wdl_type wdl = interface_.get_wdl(link_position.value());
        writer.append_sample(sample{elem, wdl});
      }
    }

    return sample_reader(train_path);
  }

  session(const std::string& root) : root_{root} {}
};

}