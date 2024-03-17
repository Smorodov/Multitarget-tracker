#include "utf8.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

namespace vkcom {

using std::string;
using std::vector;


bool check_byte(char x) { return (static_cast<uint8_t>(x) & 0xc0u) == 0x80u; }

bool check_codepoint(uint32_t x) {
  return (x < 0xd800) || (0xdfff < x && x < 0x110000);
}

uint64_t utf_length(char ch) {
  if ((static_cast<uint8_t>(ch) & 0x80u) == 0) {
    return 1;
  }
  if ((static_cast<uint8_t>(ch) & 0xe0u) == 0xc0) {
    return 2;
  }
  if ((static_cast<uint8_t>(ch) & 0xf0u) == 0xe0) {
    return 3;
  }
  if ((static_cast<uint8_t>(ch) & 0xf8u) == 0xf0) {
    return 4;
  }
  // Invalid utf-8
  return 0;
}

uint32_t chars_to_utf8(const char* begin, uint64_t size, uint64_t* utf8_len) {
  uint64_t length = utf_length(begin[0]);
  if (length == 1) {
    *utf8_len = 1;
    return static_cast<uint8_t>(begin[0]);
  }
  uint32_t code_point = 0;
  if (size >= 2 && length == 2 && check_byte(begin[1])) {
    code_point += (static_cast<uint8_t>(begin[0]) & 0x1fu) << 6u;
    code_point += (static_cast<uint8_t>(begin[1]) & 0x3fu);
    if (code_point >= 0x0080 && check_codepoint(code_point)) {
      *utf8_len = 2;
      return code_point;
    }
  } else if (size >= 3 && length == 3 && check_byte(begin[1]) &&
             check_byte(begin[2])) {
    code_point += (static_cast<uint8_t>(begin[0]) & 0x0fu) << 12u;
    code_point += (static_cast<uint8_t>(begin[1]) & 0x3fu) << 6u;
    code_point += (static_cast<uint8_t>(begin[2]) & 0x3fu);
    if (code_point >= 0x0800 && check_codepoint(code_point)) {
      *utf8_len = 3;
      return code_point;
    }
  } else if (size >= 4 && length == 4 && check_byte(begin[1]) &&
             check_byte(begin[2]) && check_byte(begin[3])) {
    code_point += (static_cast<uint8_t>(begin[0]) & 0x07u) << 18u;
    code_point += (static_cast<uint8_t>(begin[1]) & 0x3fu) << 12u;
    code_point += (static_cast<uint8_t>(begin[2]) & 0x3fu) << 6u;
    code_point += (static_cast<uint8_t>(begin[3]) & 0x3fu);
    if (code_point >= 0x10000 && check_codepoint(code_point)) {
      *utf8_len = 4;
      return code_point;
    }
  }
  // Invalid utf-8
  *utf8_len = 1;
  return INVALID_UNICODE;
}

void utf8_to_chars(uint32_t x, std::back_insert_iterator<string> it) {
  assert(check_codepoint(x));

  if (x <= 0x7f) {
    *(it++) = x;
    return;
  }

  if (x <= 0x7ff) {
    *(it++) = 0xc0u | (x >> 6u);
    *(it++) = 0x80u | (x & 0x3fu);
    return;
  }

  if (x <= 0xffff) {
    *(it++) = 0xe0u | (x >> 12u);
    *(it++) = 0x80u | ((x >> 6u) & 0x3fu);
    *(it++) = 0x80u | (x & 0x3fu);
    return;
  }

  *(it++) = 0xf0u | (x >> 18u);
  *(it++) = 0x80u | ((x >> 12u) & 0x3fu);
  *(it++) = 0x80u | ((x >> 6u) & 0x3fu);
  *(it++) = 0x80u | (x & 0x3fu);
}

string encode_utf8(const vector<uint32_t>& text) {
  string utf8_text;
  for (const uint32_t c : text) {
    utf8_to_chars(c, std::back_inserter(utf8_text));
  }
  return utf8_text;
}

vector<uint32_t> decode_utf8(const char* begin, const char* end) {
  vector<uint32_t> decoded_text;
  uint64_t utf8_len = 0;
  bool invalid_input = false;
  for (; begin < end; begin += utf8_len) {
    uint32_t code_point = chars_to_utf8(begin, end - begin, &utf8_len);
    if (code_point != INVALID_UNICODE) {
      decoded_text.push_back(code_point);
    } else {
      invalid_input = true;
    }
  }
  if (invalid_input) {
    std::cerr << "WARNING Input contains invalid unicode characters."
              << std::endl;
  }
  return decoded_text;
}

vector<uint32_t> decode_utf8(const string& utf8_text) {
  return decode_utf8(utf8_text.data(), utf8_text.data() + utf8_text.size());
}

}  // namespace vkcom
