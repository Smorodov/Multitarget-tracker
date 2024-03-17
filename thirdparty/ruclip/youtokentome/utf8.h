#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <iterator>

namespace vkcom {

constexpr static uint32_t INVALID_UNICODE = 0x0fffffff;

uint32_t chars_to_utf8(const char* begin, uint64_t size, uint64_t* utf8_len);

void utf8_to_chars(uint32_t x, std::back_insert_iterator<std::string> it);

std::string encode_utf8(const std::vector<uint32_t> &utext);

std::vector<uint32_t> decode_utf8(const char *begin, const char *end);

std::vector<uint32_t> decode_utf8(const std::string &utf8_text);

struct UTF8Iterator {
  UTF8Iterator(char* begin, char* end): begin(begin), end(end) {}

  UTF8Iterator operator++() {
    if (!state) {
      parse();
    }
    begin += utf8_len;
    state = false;
    return *this;
  }

  uint32_t operator*() {
    if (!state) {
      parse();
    }
    return code_point;
  }

  char* get_ptr() {
    return begin;
  }
  uint64_t get_utf8_len() {
    return utf8_len;
  }

  bool empty() {
    assert(begin <= end);
    return begin == end;
  }
private:
  char *begin, *end;
  uint32_t code_point = 0;
  uint64_t utf8_len = 0;
  bool state = false;
  void parse() {
    if (state) {
      return;
    }
    assert(!empty());
    code_point = chars_to_utf8(begin, end - begin, &utf8_len);
    state = true;
  }
};

} // namespace vkcom
