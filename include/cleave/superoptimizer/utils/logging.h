#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cleave_superoptimizer {
namespace logging {

enum class LogSeverity {
  DEBUG = 0,
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  FATAL = 4,
};

struct VLogConfig {
  int default_level = -1;
  std::vector<std::pair<std::string, int>> file_levels;
};

inline char const *SeverityName(LogSeverity severity) {
  switch (severity) {
    case LogSeverity::DEBUG:
      return "DEBUG";
    case LogSeverity::INFO:
      return "INFO";
    case LogSeverity::WARNING:
      return "WARNING";
    case LogSeverity::ERROR:
      return "ERROR";
    case LogSeverity::FATAL:
      return "FATAL";
  }
  return "UNKNOWN";
}

inline std::mutex &LogMutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::string NormalizePath(std::string path) {
  std::replace(path.begin(), path.end(), '\\', '/');
  return path;
}

inline bool EndsWith(std::string const &value, std::string const &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

inline bool ParseInt(std::string const &value, int *out) {
  if (value.empty()) {
    return false;
  }
  char *end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (*end != '\0') {
    return false;
  }
  *out = static_cast<int>(parsed);
  return true;
}

inline void ApplyLogDebugSpec(std::string const &spec, VLogConfig *config) {
  if (spec == "1" || spec == "true" || spec == "TRUE") {
    config->default_level = 1;
    return;
  }
  if (spec == "0" || spec == "false" || spec == "FALSE") {
    config->default_level = -1;
    return;
  }

  size_t begin = 0;
  while (begin <= spec.size()) {
    size_t end = spec.find(',', begin);
    std::string token =
        spec.substr(begin, end == std::string::npos ? std::string::npos
                                                    : end - begin);
    if (!token.empty()) {
      size_t equals = token.find('=');
      if (equals == std::string::npos) {
        int level = -1;
        if (ParseInt(token, &level)) {
          config->default_level = level;
        }
      } else {
        std::string key = token.substr(0, equals);
        std::string value = token.substr(equals + 1);
        int level = -1;
        if (ParseInt(value, &level)) {
          if (key == "DEFAULT") {
            config->default_level = level;
          } else {
            config->file_levels.emplace_back(NormalizePath(key), level);
          }
        }
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
}

inline VLogConfig const &GetVLogConfig() {
  static VLogConfig config = [] {
    VLogConfig parsed;
    char const *env = std::getenv("CLEAVE_SUPEROPTIMIZER_LOG_DEBUG");
    if (env != nullptr && env[0] != '\0') {
      ApplyLogDebugSpec(env, &parsed);
    }
    return parsed;
  }();
  return config;
}

inline bool FileMatches(std::string const &file, std::string const &pattern) {
  return file == pattern || EndsWith(file, pattern);
}

inline int VLogLevelForFile(char const *file) {
  std::string normalized_file = NormalizePath(file == nullptr ? "" : file);
  VLogConfig const &config = GetVLogConfig();
  int level = config.default_level;
  size_t best_match_len = 0;
  for (auto const &entry : config.file_levels) {
    if (entry.first.size() >= best_match_len &&
        FileMatches(normalized_file, entry.first)) {
      best_match_len = entry.first.size();
      level = entry.second;
    }
  }
  return level;
}

inline bool IsVLogEnabled(char const *file, int level) {
  return level <= VLogLevelForFile(file);
}

class LogMessage {
 public:
  LogMessage(LogSeverity severity, char const *file, int line,
             int vlog_level = -1, bool plain_info = false)
      : severity_(severity),
        file_(file == nullptr ? "" : file),
        line_(line),
        vlog_level_(vlog_level),
        plain_info_(plain_info) {}

  ~LogMessage() {
    {
      std::lock_guard<std::mutex> lock(LogMutex());
      if (plain_info_ && severity_ == LogSeverity::INFO) {
        std::cerr << stream_.str() << std::endl;
      } else {
        std::cerr << "[" << SeverityName(severity_) << "] " << file_ << ":"
                  << line_;
        if (vlog_level_ >= 0) {
          std::cerr << " VLOG(" << vlog_level_ << ")";
        }
        std::cerr << ": " << stream_.str() << std::endl;
      }
    }
    if (severity_ == LogSeverity::FATAL) {
      std::abort();
    }
  }

  std::ostream &stream() { return stream_; }

 private:
  LogSeverity severity_;
  char const *file_;
  int line_;
  int vlog_level_;
  bool plain_info_;
  std::ostringstream stream_;
};

class LogMessageVoidify {
 public:
  void operator&(std::ostream &) {}
};

} // namespace logging
} // namespace cleave_superoptimizer

#define CSO_LOG(severity)                                                    \
  ::cleave_superoptimizer::logging::LogMessageVoidify() &                    \
      ::cleave_superoptimizer::logging::LogMessage(                          \
          ::cleave_superoptimizer::logging::LogSeverity::severity, __FILE__,  \
          __LINE__, -1, true)                                                 \
          .stream()

#define CSO_LOG_IF(severity, condition)                                      \
  !(condition) ? (void)0                                                     \
               : ::cleave_superoptimizer::logging::LogMessageVoidify() &     \
                     ::cleave_superoptimizer::logging::LogMessage(           \
                         ::cleave_superoptimizer::logging::LogSeverity::      \
                             severity,                                       \
                         __FILE__, __LINE__)                                 \
                         .stream()

#if defined(CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING) &&                  \
    CLEAVE_SUPEROPTIMIZER_ENABLE_DEBUG_LOGGING
#define CSO_VLOG_IS_ON(level)                                                \
  (::cleave_superoptimizer::logging::IsVLogEnabled(__FILE__, (level)))
#define CSO_DLOG(severity) CSO_LOG_IF(severity, CSO_VLOG_IS_ON(1))
#define CSO_VLOG(level)                                                      \
  !CSO_VLOG_IS_ON(level)                                                     \
      ? (void)0                                                              \
      : ::cleave_superoptimizer::logging::LogMessageVoidify() &              \
            ::cleave_superoptimizer::logging::LogMessage(                    \
                ::cleave_superoptimizer::logging::LogSeverity::DEBUG,         \
                __FILE__, __LINE__, (level))                                 \
                .stream()
#else
#define CSO_VLOG_IS_ON(level) false
#define CSO_DLOG(severity)                                                   \
  true ? (void)0                                                             \
       : ::cleave_superoptimizer::logging::LogMessageVoidify() &             \
             ::cleave_superoptimizer::logging::LogMessage(                   \
                 ::cleave_superoptimizer::logging::LogSeverity::severity,     \
                 __FILE__, __LINE__)                                         \
                 .stream()
#define CSO_VLOG(level)                                                      \
  true ? (void)0                                                             \
       : ::cleave_superoptimizer::logging::LogMessageVoidify() &             \
             ::cleave_superoptimizer::logging::LogMessage(                   \
                 ::cleave_superoptimizer::logging::LogSeverity::DEBUG,        \
                 __FILE__, __LINE__, (level))                                \
                 .stream()
#endif

#define CSO_CHECK(condition)                                                  \
  CSO_LOG_IF(FATAL, !(condition)) << "Check failed: " #condition ". "
