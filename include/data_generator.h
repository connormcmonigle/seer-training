#pragma once

#include <optional>
#include <string>
#include <thread>
#include <memory>
#include <random>

#include <chess/board_history.h>
#include <nnue/eval.h>
#include <search/search_constants.h>
#include <search/search_worker.h>
#include <nnue/embedded_weights.h>
#include <nnue/weights_streamer.h>

#include <sample.h>
#include <sample_reader.h>
#include <sample_writer.h>
#include <data_writer.h>


namespace train{


struct data_generator{
  size_t concurrency_{1};

  search::depth_type ply_limit_{256};
  search::depth_type random_ply_{10};
  search::depth_type fixed_depth_{6};
  size_t fixed_nodes_{5120};

  score_type eval_limit_{6144};

  std::shared_ptr<search::transposition_table> tt_{nullptr};
  std::shared_ptr<search::search_constants> constants_ = std::make_shared<search::search_constants>(1);

  data_writer writer_;

  data_generator& set_concurrency(const size_t& concurrency){
    concurrency_ = concurrency;
    return *this;
  }

  data_generator& set_fixed_depth(const search::depth_type& fixed_depth){
    fixed_depth_ = fixed_depth;
    return *this;
  }

  data_generator& set_fixed_nodes(const size_t& fixed_nodes) {
    fixed_nodes_ = fixed_nodes;
    return *this;
  }

  data_generator& set_ply_limit(const search::depth_type& ply_limit){
    ply_limit_ = ply_limit;
    return *this;
  }

  data_generator& set_random_ply(const search::depth_type& random_ply){
    random_ply_ = random_ply;
    return *this;
  }

  data_generator& set_eval_limit(const score_type& eval_limit){
    eval_limit_ = std::abs(eval_limit);
    return *this;
  }

  data_generator& generate_data() {
    const nnue::weights weights = [&, this]{
      nnue::weights result{};
      nnue::embedded_weight_streamer embedded(nnue::embed::weights_file_data);
      result.load(embedded);
      return result;
    }();
    
    auto generate = [&, this]{
      using worker_type = search::search_worker;
      auto gen = std::mt19937(std::random_device()());


      std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(
        &weights, tt_, constants_,
        [&, this](const auto& w) { if (w.depth() >= fixed_depth_) { worker->stop(); } },
        [&, this](const auto& w) { if (w.nodes() >= fixed_nodes_) { worker->stop(); } }
      );

      while (!writer_.is_complete()) {
        (worker->internal).reset();
        
        std::vector<sample> block{};

        chess::board_history hist{};
        state_type state = state_type::start_pos();

        const result_type game_result = [&]{ 
          for (search::depth_type ply = 0; ply <= ply_limit_ && !is_terminal(hist, state); ++ply) {
            if (ply < random_ply_) {
              const auto mv_list = state.generate_moves();
              const size_t idx = std::uniform_int_distribution<size_t>(0, mv_list.size()-1)(gen);

              hist.push(state.hash());
              state = state.forward(mv_list[idx]);
            } else {
              worker->go(hist, state, 1);
              worker->iterative_deepening_loop();
              worker->stop();

              const auto best_move = worker->best_move();
              const auto best_score = worker->score();

              if (best_score >= eval_limit_) { return result_type::win; }
              if (best_score <= -eval_limit_) { return result_type::loss; }

              const auto view = search::stack_view::root((worker->internal).stack);
              const auto evaluator = [&] {
                nnue::eval result(&weights, &worker->internal.scratchpad, 0, 0);
                state.feature_full_reset(result);
                return result;
              }();
            
              const search::score_type static_eval = evaluator.evaluate(state.turn(), state.phase<real_type>());

              worker->go(hist, state, 1);
              nnue::eval_node node = nnue::eval_node::clean_node(evaluator);
              const search::score_type q_eval = worker->q_search<true, false>(view, node, state, search::mate_score, -search::mate_score, 0);
              worker->stop();

              if (!state.is_check() && static_eval == q_eval) { block.emplace_back(state, best_score); }

              hist.push(state.hash());
              state = state.forward(best_move);
            }
          }

          return get_result(hist, state);
        }();

        for (auto& elem : block) { elem.set_result(relative_result(state.turn(), elem.pov(), game_result)); }

        writer_.write_block(block);
      }
    };

    std::vector<std::thread> threads{};
    for (size_t i(0); i < concurrency_; ++i) { threads.emplace_back(generate); }
    for (auto& thread : threads) { thread.join(); }
    return *this;
  }

  data_generator(const std::string& path, const size_t& total, const size_t& tt_mb_size) : writer_{path, total} {
    tt_ = std::make_shared<search::transposition_table>(tt_mb_size);
  }
};

}
