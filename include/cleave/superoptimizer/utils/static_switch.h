#pragma once

#define INT_SWITCH(VARNAME, CONSTNAME, ...)                                    \
  [&] {                                                                        \
    if (VARNAME <= 4) {                                                        \
      constexpr static int CONSTNAME = 4;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 8) {                                                 \
      constexpr static int CONSTNAME = 8;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 16) {                                                \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                               \
      constexpr static int CONSTNAME = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 256) {                                               \
      constexpr static int CONSTNAME = 256;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()

#define WARP_SHAPE_M_SWITCH(VARNAME, CONSTNAME, ...)                           \
  [&] {                                                                        \
    if (VARNAME <= 16) {                                                       \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                               \
      constexpr static int CONSTNAME = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()

#define WARP_SHAPE_N_SWITCH(VARNAME, CONSTNAME, ...)                           \
  [&] {                                                                        \
    if (VARNAME <= 8) {                                                        \
      constexpr static int CONSTNAME = 8;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 16) {                                                \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                               \
      constexpr static int CONSTNAME = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()
