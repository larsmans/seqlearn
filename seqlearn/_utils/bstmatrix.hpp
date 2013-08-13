/* Copyright 2013 Lars Buitinck */

#include <map>
#include <utility>

#include "FSBAllocator.hh"

namespace seqlearn {
    /*
     * BST backing a sparse matrix of doubles.
     */
    class BSTMatrix {
        typedef std::pair<size_t, size_t> KeyT;
        //std::map<std::pair<size_t, size_t>, double> tree;
        std::map<KeyT, double, std::less<KeyT>,
                 FSBAllocator<std::pair<KeyT const, double>>> tree;

      public:
        // Add factor * B to this.
        void add(BSTMatrix const *B, double factor)
        {
            for (auto it(B->tree.begin()), end(B->tree.end()); it != end; ++it)
                tree[it->first] += factor * it->second;
        }

        // self[i, j] += v
        void add(size_t i, size_t j, double v)
        {
            tree[std::make_pair(i, j)] += v;
        }

        // Add B to unraveled, C-contiguous array A.
        void add_to_dense(double *A, size_t ncols)
        {
            for (auto it(tree.begin()), end(tree.end()); it != end; ++it) {
                size_t i = it->first.first, j = it->first.second;
                A[i * ncols + j] += it->second;
            }
        }

        double get(size_t i, size_t j) const
        {
            auto it = tree.find(std::make_pair(i, j));
            if (it == tree.end())
                return 0.;
            else
                return it->second;
        }

        void mul(double v)
        {
            for (auto it(tree.begin()), end(tree.end()); it != end; ++it)
                it->second *= v;
        }

        void put(size_t i, size_t j, double v)
        {
            tree[std::make_pair(i, j)] = v;
        }

        size_t size() const
        {
            return tree.size();
        }
    };
}
