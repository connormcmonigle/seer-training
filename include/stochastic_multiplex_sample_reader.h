#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <optional>
#include <sample.h>
#include <sample_reader.h>


namespace train {

struct stochastic_multiplex_sample_reader_iterator {
  using difference_type = long;
  using value_type = sample;
  using pointer = const sample*;
  using reference = const sample&;
  using iterator_category = std::input_iterator_tag;

  using distribution_type = std::discrete_distribution<size_t>;

  size_t advance_idx_{};
  size_t total_remaining_{};
  
  std::mt19937 random_number_generator_;
  distribution_type distribution_;
  std::vector<size_t> remaining_;
  std::vector<sample_reader::iterator> reader_begins_;

  constexpr bool empty() const { return total_remaining_ == 0; }

  constexpr bool operator==(const stochastic_multiplex_sample_reader_iterator& other) const { return empty() && other.empty(); }
  constexpr bool operator!=(const stochastic_multiplex_sample_reader_iterator& other) const { return !(*this == other); }

  stochastic_multiplex_sample_reader_iterator& operator++() {
    ++reader_begins_[advance_idx_];
    --remaining_[advance_idx_];
    --total_remaining_;

    if (!empty()){
      const auto params = distribution_type::param_type(remaining_.begin(), remaining_.end());
      advance_idx_ = distribution_(random_number_generator_, params);
    }

    return *this;
  }

  sample operator*() const { return *reader_begins_[advance_idx_]; }


  stochastic_multiplex_sample_reader_iterator() {}

  stochastic_multiplex_sample_reader_iterator(
    const std::vector<size_t>& sizes,
    const std::vector<sample_reader::iterator>& reader_begins
  ) :
    random_number_generator_(),
    distribution_(sizes.begin(), sizes.end()),
    remaining_{sizes},
    reader_begins_{reader_begins}
  {
    total_remaining_ = std::reduce(remaining_.begin(), remaining_.end());
    if (!empty()) { advance_idx_ = distribution_(random_number_generator_); }
  } 

};


struct stochastic_multiplex_sample_reader {
  std::vector<size_t> sizes_;
  std::vector<sample_reader> readers_;

  stochastic_multiplex_sample_reader_iterator begin() const {
    std::vector<sample_reader::iterator> reader_begins;
    std::transform(readers_.begin(), readers_.end(), std::back_inserter(reader_begins), [](const sample_reader& reader) { return reader.begin(); });
    return stochastic_multiplex_sample_reader_iterator(sizes_, reader_begins);
  }

  stochastic_multiplex_sample_reader_iterator end() const { return stochastic_multiplex_sample_reader_iterator(); }

  stochastic_multiplex_sample_reader(const std::vector<size_t>& sizes, const std::vector<sample_reader>& readers) :
    sizes_{sizes}, readers_{readers} {} 
};

}