#pragma once

#include <cassert>
#include <iostream>
#include <string>

/// Very simple replacement for std::format introduced in C++20. Only supports
/// replacing `{}` in the template string with whatever `operator<<` for that
/// type turns it into.
/// Courtesy Jelmer: https://github.com/browsermt/bergamot-translator/blob/df5db525132fb24b02f80ac07dc98ba02f536e92/src/translator/html.cpp#L62
inline std::string format(std::string const &formatTemplate) {
  return formatTemplate;
}

template <typename Arg>
std::string format(std::string const &formatTemplate, Arg arg) {
  std::ostringstream os;
  auto index = formatTemplate.find("{}");
  assert(index != std::string::npos);
  os << formatTemplate.substr(0, index) << arg << formatTemplate.substr(index + 2);
  return os.str();
}

template <typename Arg, typename... Args>
std::string format(std::string const &formatTemplate, Arg arg, Args... args) {
  std::ostringstream os;
  auto index = formatTemplate.find("{}");
  assert(index != std::string::npos);
  os << formatTemplate.substr(0, index) << arg
     << format(formatTemplate.substr(index + 2), std::forward<Args>(args)...);
  return os.str();
}

#define ABORT(...)                                                            \
  do {                                                                        \
    std::cerr << format(__VA_ARGS__) << std::endl;                            \
    std::cerr << format("Aborted at {}:{}", __FILE__, __LINE__) << std::endl; \
    std::abort();                                                             \
  } while(0)

#define ABORT_IF(condition, ...) \
  do {                           \
    if((condition)) {            \
      ABORT(__VA_ARGS__);        \
    }                            \
  } while(0)
