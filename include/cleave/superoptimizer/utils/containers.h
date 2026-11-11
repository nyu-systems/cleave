#pragma once

#include <algorithm>
#include <vector>

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k) {
  return m.count(k) > 0;
}

template <typename T>
bool contains(std::vector<T> const &v, T const &e) {
  for (auto const &x : v) {
    if (x == e) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::vector<T> to_vector(int n, T *arr) {
  std::vector<T> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(arr[i]);
  }
  return v;
}
