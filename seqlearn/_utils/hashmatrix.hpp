/* Copyright 2013 Lars Buitinck */

#include <unordered_map>
#include <utility>

namespace seqlearn {
    /*
     * Hash table backing a sparse matrix of doubles.
     */
    class HashMatrix {
        struct hash_pair {
            size_t operator()(std::pair<size_t, size_t> const &ij) const
            {
                // Borrowed from boost::hash_combine;
                // http://stackoverflow.com/q/4948780/166749
                size_t i = ij.first, j = ij.second;
                return i ^ (j + 0x9e3779b9 + (i << 6) + (i >> 2));
            }
        };
        std::unordered_map<std::pair<size_t, size_t>, double,
                           hash_pair> h;

      public:
        // Add factor * B to this.
        void add(HashMatrix const *B, double factor)
        {
            for (auto it(B->h.begin()), end(B->h.end()); it != end; ++it)
                h[it->first] += factor * it->second;
        }

        // self[i, j] += v
        void add(size_t i, size_t j, double v)
        {
            h[std::make_pair(i, j)] += v;
        }

        // Add B to unraveled, C-contiguous array A.
        void add_to_dense(double *A, size_t ncols)
        {
            for (auto it(h.begin()), end(h.end()); it != end; ++it) {
                size_t i = it->first.first, j = it->first.second;
                A[i * ncols + j] += it->second;
            }
        }

        double get(size_t i, size_t j) const
        {
            auto it = h.find(std::make_pair(i, j));
            if (it == h.end())
                return 0.;
            else
                return it->second;
        }

        void mul(double v)
        {
            for (auto it(h.begin()), end(h.end()); it != end; ++it)
                it->second *= v;
        }

        void put(size_t i, size_t j, double v)
        {
            h[std::make_pair(i, j)] = v;
        }

        size_t size() const
        {
            return h.size();
        }
    };
}
