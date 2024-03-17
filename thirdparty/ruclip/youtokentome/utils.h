#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "third_party/flat_hash_map.h"

namespace vkcom {
const uint32_t SPACE_TOKEN = 9601;

struct BPE_Rule {
  // x + y -> z
  uint32_t x{0};
  uint32_t y{0};
  uint32_t z{0};

  BPE_Rule() = default;

  BPE_Rule(uint32_t x, uint32_t y, uint32_t z);

  bool operator==(const BPE_Rule &other) const;
};

struct SpecialTokens {
  int pad_id = -1;
  int unk_id = -1;
  int bos_id = -1;
  int eos_id = -1;

  SpecialTokens() = default;

  SpecialTokens(int pad_id, int unk_id, int bos_id, int eos_id);

  void dump(std::ofstream &fout);

  void load(std::ifstream &fin);

  uint32_t max_id() const;

  bool taken_id(int id) const;

  uint64_t n_special_tokens() const;
};

struct BpeConfig {
  double character_coverage = 1;
  int n_threads = 0;
  SpecialTokens special_tokens;

  BpeConfig() = default;

  BpeConfig(double character_coverage, int n_threads,
            const SpecialTokens &special_tokens);
};

struct Status {
  int code{0};
  std::string message;
  Status() = default;
  Status(int code, std::string message);

  const std::string &error_message() const;
  bool ok() const;
};

struct BPEState {
  flat_hash_map<uint32_t, uint32_t> char2id;
  std::vector<BPE_Rule> rules;
  SpecialTokens special_tokens;

  void dump(const std::string &file_name);

  Status load(const std::string &file_name);
};

struct DecodeResult {
  std::vector<int> ids;
  std::vector<std::string> pieces;
};

struct EncodingConfig {
  bool bos;
  bool eos;
  bool reverse;
  double dropout_prob;
};

bool is_space(uint32_t ch);

std::vector<std::string> read_lines_from_stdin(uint64_t batch_limit, uint64_t *processed);

template<typename T>
void write_to_stdout(const std::vector<std::vector<T>> &sentences, bool flush) {
  for (const auto &sentence : sentences) {
    for (const auto &token : sentence) {
      std::cout << token << " ";
    }
    std::cout << "\n";
  }
  if (flush) {
    std::cout << std::flush;
  }
}

}  // namespace vkcom
