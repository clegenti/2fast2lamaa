#pragma once

// A grid-aligned index of the map's cells (VDB-like, 3 levels), to replace the hash map and the octrees
// of MapDistField: a cell's position is implied by where it is stored, so only its pointer is kept.
//
//   leaf block   8x8x8 voxels: a 512-bit occupancy mask (one uint64 per z-slice, bit x + 8y), a
//                512-bit edge mask, and the cells' pointers packed in mask order (the rank of a voxel
//                is the number of occupied voxels before it)
//   super-block  8x8x8 blocks: a 512-bit mask of the non-empty blocks (and of those with an edge
//                cell), the blocks packed by value in mask order
//   root         hash map from a super-block's packed coordinates to the super-block
//
// Voxel (x, y, z) has its centre at cellCentre(x), cellCentre(y), cellCentre(z): recomputed, never
// stored, with one float formula shared by every user so that the centres are the same bits
// everywhere. Distances are computed as the octree computes them (sqDist), so searches return the
// same cells as the octree of centres did, with the same distances.
//
// One writer (insert/erase/setEdge/clear), any number of concurrent readers, never both at once.
// Voxel coordinates must lie within +-2^24 (exact as floats in the centre formula): +-2500 km at
// 0.15 m cells.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

namespace vdb
{

// Centre of voxel i along an axis, for the cell size cs and its half hcs. One fused multiply-add in
// float, whatever the compiler's contraction settings, so that every caller gets the same bits
inline double cellCentre(int i, float cs, float hcs)
{
    return (double)std::fma((float)i, cs, hcs);
}

// Squared distance from the differences of coordinates, summed as thuni::Octree's loop is compiled
// with the package's flags (FMA contraction): explicit fused multiply-adds, so that the order of the
// roundings does not depend on where the compiler inlines it
inline double sqDistDiff(double dx, double dy, double dz)
{
    return std::fma(dz, dz, std::fma(dy, dy, dx*dx));
}

// Squared distance, computed as thuni::Octree computes it
inline double sqDist(const double* p, const double* q)
{
    return sqDistDiff(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

struct VoxelCoord
{
    int x, y, z;
    bool operator<(const VoxelCoord& o) const
    {
        if(x != o.x) return x < o.x;
        if(y != o.y) return y < o.y;
        return z < o.z;
    }
    bool operator==(const VoxelCoord& o) const { return x == o.x && y == o.y && z == o.z; }
};

template <typename T>
class VoxelIndex
{
    public:
        explicit VoxelIndex(double cell_size)
            : cs_(cell_size)
            , inv_cs_(1.0/cell_size)
            , cs_f_((float)cell_size)
            , hcs_f_((float)cell_size/2.0f)
        {}

        ~VoxelIndex() = default;
        VoxelIndex(const VoxelIndex&) = delete;
        VoxelIndex& operator=(const VoxelIndex&) = delete;

        double centre(int i) const { return cellCentre(i, cs_f_, hcs_f_); }

        T* find(int x, int y, int z) const
        {
            const Block* b = findBlock(x >> 3, y >> 3, z >> 3);
            if(b == nullptr) return nullptr;
            const unsigned w = z & 7, bit = (x & 7) + 8*(y & 7);
            if(!((b->occ[w] >> bit) & 1)) return nullptr;
            return b->cells[rank(b->occ, b->prefix, w, bit)];
        }

        T* findWithEdge(int x, int y, int z, bool& is_edge) const
        {
            is_edge = false;
            const Block* b = findBlock(x >> 3, y >> 3, z >> 3);
            if(b == nullptr) return nullptr;
            const unsigned w = z & 7, bit = (x & 7) + 8*(y & 7);
            if(!((b->occ[w] >> bit) & 1)) return nullptr;
            is_edge = (b->edge[w] >> bit) & 1;
            return b->cells[rank(b->occ, b->prefix, w, bit)];
        }

        // False (and nothing changed) if the voxel already holds a cell
        bool insert(int x, int y, int z, T* cell)
        {
            const int sx = x >> 6, sy = y >> 6, sz = z >> 6;
            auto [it, created] = root_.try_emplace(superKey(sx, sy, sz));
            SuperBlock& s = it->second;
            if(created)
            {
                s.sx = sx; s.sy = sy; s.sz = sz;
                uint64_t* g = groups_[superKey(sx >> 3, sy >> 3, sz >> 3)].data();
                g[sz & 7] |= 1ull << ((sx & 7) + 8*(sy & 7));
            }
            const unsigned bw = (z >> 3) & 7, bbit = ((x >> 3) & 7) + 8*((y >> 3) & 7);
            const unsigned br = rank(s.mask, s.prefix, bw, bbit);
            if(!((s.mask[bw] >> bbit) & 1))
            {
                s.blocks.insert(s.blocks.begin() + br, Block());
                setBit(s.mask, s.prefix, bw, bbit);
            }
            Block& b = s.blocks[br];
            const unsigned w = z & 7, bit = (x & 7) + 8*(y & 7);
            if((b.occ[w] >> bit) & 1) return false;
            const unsigned r = rank(b.occ, b.prefix, w, bit);
            if(b.n == b.cap)
            {
                const unsigned cap = b.cap + kCellsGrowth;
                T** cells = (T**)std::realloc(b.cells, cap*sizeof(T*));
                if(cells == nullptr) throw std::bad_alloc();
                b.cells = cells;
                b.cap = cap;
            }
            std::memmove(b.cells + r + 1, b.cells + r, (b.n - r)*sizeof(T*));
            b.cells[r] = cell;
            b.n++;
            setBit(b.occ, b.prefix, w, bit);
            size_++;
            return true;
        }

        // The cell removed, or nullptr if the voxel was empty
        T* erase(int x, int y, int z)
        {
            auto it = root_.find(superKey(x >> 6, y >> 6, z >> 6));
            if(it == root_.end()) return nullptr;
            SuperBlock& s = it->second;
            const unsigned bw = (z >> 3) & 7, bbit = ((x >> 3) & 7) + 8*((y >> 3) & 7);
            if(!((s.mask[bw] >> bbit) & 1)) return nullptr;
            const unsigned br = rank(s.mask, s.prefix, bw, bbit);
            Block& b = s.blocks[br];
            const unsigned w = z & 7, bit = (x & 7) + 8*(y & 7);
            if(!((b.occ[w] >> bit) & 1)) return nullptr;
            const unsigned r = rank(b.occ, b.prefix, w, bit);
            T* cell = b.cells[r];
            std::memmove(b.cells + r, b.cells + r + 1, (b.n - r - 1)*sizeof(T*));
            b.n--;
            clearBit(b.occ, b.prefix, w, bit);
            if((b.edge[w] >> bit) & 1)
            {
                b.edge[w] &= ~(1ull << bit);
                s.n_edges--;
                n_edges_--;
                if(!anyBit(b.edge)) s.edge_mask[bw] &= ~(1ull << bbit);
            }
            size_--;
            if(b.n == 0)
            {
                s.blocks.erase(s.blocks.begin() + br);
                clearBit(s.mask, s.prefix, bw, bbit);
                if(s.blocks.empty())
                {
                    const int sx = s.sx, sy = s.sy, sz = s.sz;
                    root_.erase(it);
                    auto git = groups_.find(superKey(sx >> 3, sy >> 3, sz >> 3));
                    git->second[sz & 7] &= ~(1ull << ((sx & 7) + 8*(sy & 7)));
                    if(!anyBit(git->second.data())) groups_.erase(git);
                }
            }
            return cell;
        }

        // Marks the voxel's cell as an edge cell. False if the voxel is empty or already marked
        bool setEdge(int x, int y, int z)
        {
            auto it = root_.find(superKey(x >> 6, y >> 6, z >> 6));
            if(it == root_.end()) return false;
            SuperBlock& s = it->second;
            const unsigned bw = (z >> 3) & 7, bbit = ((x >> 3) & 7) + 8*((y >> 3) & 7);
            if(!((s.mask[bw] >> bbit) & 1)) return false;
            Block& b = s.blocks[rank(s.mask, s.prefix, bw, bbit)];
            const unsigned w = z & 7, bit = (x & 7) + 8*(y & 7);
            if(!((b.occ[w] >> bit) & 1) || ((b.edge[w] >> bit) & 1)) return false;
            b.edge[w] |= 1ull << bit;
            s.edge_mask[bw] |= 1ull << bbit;
            s.n_edges++;
            n_edges_++;
            return true;
        }

        // The cells (of all, or only the edge ones) whose centre is at a squared distance d2 < r*r
        // from q, with d2, in no particular order
        void radius(const double* q, double r, bool edge_only, std::vector<T*>& cells, std::vector<double>& d2) const
        {
            cells.clear();
            d2.clear();
            if(root_.empty() || (edge_only && n_edges_ == 0)) return;
            RadiusQuery rq;
            rq.q = q;
            rq.r2 = r*r;
            rq.edge_only = edge_only;
            rq.cells = &cells;
            rq.d2 = &d2;
            voxelRange(q, r, rq.lo, rq.hi);
            const int slo[3] = {rq.lo[0] >> 6, rq.lo[1] >> 6, rq.lo[2] >> 6};
            const int shi[3] = {rq.hi[0] >> 6, rq.hi[1] >> 6, rq.hi[2] >> 6};
            const double probes = double(shi[0] - slo[0] + 1)*double(shi[1] - slo[1] + 1)*double(shi[2] - slo[2] + 1);
            if(probes <= (double)root_.size())
            {
                for(int sx = slo[0]; sx <= shi[0]; ++sx)
                    for(int sy = slo[1]; sy <= shi[1]; ++sy)
                        for(int sz = slo[2]; sz <= shi[2]; ++sz)
                        {
                            auto it = root_.find(superKey(sx, sy, sz));
                            if(it != root_.end()) radiusSuper(it->second, rq);
                        }
            }
            else
            {
                for(const auto& kv : root_)
                {
                    const SuperBlock& s = kv.second;
                    if(s.sx < slo[0] || s.sx > shi[0] || s.sy < slo[1] || s.sy > shi[1] || s.sz < slo[2] || s.sz > shi[2]) continue;
                    radiusSuper(s, rq);
                }
            }
        }

        // The k cells (of all, or only the edge ones) whose centres are nearest to q, nearest first,
        // with their squared distances (and voxels). Exact; ties are broken by the voxel coordinates
        // (smallest first). Fewer than k if there are fewer cells
        void knn(const double* q, int k, bool edge_only, std::vector<T*>& cells, std::vector<double>& d2, std::vector<VoxelCoord>* voxels = nullptr) const
        {
            cells.clear();
            d2.clear();
            if(voxels) voxels->clear();
            if(k <= 0 || root_.empty() || (edge_only && n_edges_ == 0)) return;
            KnnQuery kq;
            kq.q = q;
            kq.k = k;
            kq.edge_only = edge_only;
            thread_local std::vector<Cand> heap;
            heap.clear();
            kq.heap = &heap;

            // Boxes of growing half-width h around q, each scanned afresh. Once k cells are held, the
            // box held every cell nearer than the k-th if the k-th is nearer than h (a centre outside
            // the box is farther than h along one axis); otherwise the next box reaches the k-th's
            // distance, and is sure to be the last. Far from the map, when a box would span more
            // super-blocks than the map has, the super-blocks are visited nearest first instead
            double h = kFirstBox*cs_;
            while(true)
            {
                voxelRange(q, h, kq.lo, kq.hi);
                const int slo[3] = {kq.lo[0] >> 6, kq.lo[1] >> 6, kq.lo[2] >> 6};
                const int shi[3] = {kq.hi[0] >> 6, kq.hi[1] >> 6, kq.hi[2] >> 6};
                const double probes = double(shi[0] - slo[0] + 1)*double(shi[1] - slo[1] + 1)*double(shi[2] - slo[2] + 1);
                heap.clear();
                if(probes > std::min((double)root_.size(), kMaxBoxProbes))
                {
                    nearestFirst(kq);
                    break;
                }
                for(int sx = slo[0]; sx <= shi[0]; ++sx)
                    for(int sy = slo[1]; sy <= shi[1]; ++sy)
                        for(int sz = slo[2]; sz <= shi[2]; ++sz)
                        {
                            if(kq.full() && superLowerBound(sx, sy, sz, q) > kq.worst()*kPruneSlack) continue;
                            auto it = root_.find(superKey(sx, sy, sz));
                            if(it != root_.end()) knnSuper(it->second, kq);
                        }
                if(kq.full())
                {
                    if(kq.worst() < h*h*kStopSlack) break;
                    h = std::sqrt(kq.worst())*kGrowSlack;
                }
                else
                {
                    h *= 3;
                }
            }
            for(const Cand& c : heap)
            {
                cells.push_back(c.cell);
                d2.push_back(c.d2);
                if(voxels) voxels->push_back(c.v);
            }
        }

        // f(VoxelCoord, T*, bool is_edge) for every cell
        template <typename F>
        void forEach(F&& f) const
        {
            for(const auto& kv : root_)
            {
                const SuperBlock& s = kv.second;
                unsigned bi = 0;
                for(unsigned bw = 0; bw < 8; ++bw)
                {
                    for(uint64_t bits = s.mask[bw]; bits; bits &= bits - 1, ++bi)
                    {
                        const unsigned bbit = __builtin_ctzll(bits);
                        const Block& b = s.blocks[bi];
                        const int ox = (s.sx*8 + (int)(bbit & 7))*8, oy = (s.sy*8 + (int)(bbit >> 3))*8, oz = (s.sz*8 + (int)bw)*8;
                        unsigned ci = 0;
                        for(unsigned w = 0; w < 8; ++w)
                        {
                            for(uint64_t vb = b.occ[w]; vb; vb &= vb - 1, ++ci)
                            {
                                const unsigned bit = __builtin_ctzll(vb);
                                f(VoxelCoord{ox + (int)(bit & 7), oy + (int)(bit >> 3), oz + (int)w}, b.cells[ci], (bool)((b.edge[w] >> bit) & 1));
                            }
                        }
                    }
                }
            }
        }

        size_t size() const { return size_; }
        size_t numEdges() const { return n_edges_; }
        size_t numSuperBlocks() const { return root_.size(); }
        size_t numBlocks() const
        {
            size_t n = 0;
            for(const auto& kv : root_) n += kv.second.blocks.size();
            return n;
        }

        // Does not delete the cells
        void clear()
        {
            root_.clear();
            groups_.clear();
            size_ = 0;
            n_edges_ = 0;
        }

        // Bytes held by the index (not the cells), allocator overheads excluded
        size_t memoryBytes() const
        {
            size_t bytes = root_.values().capacity()*sizeof(typename Root::value_type) + root_.bucket_count()*sizeof(typename Root::bucket_type)
                         + groups_.values().capacity()*sizeof(typename Groups::value_type) + groups_.bucket_count()*sizeof(typename Groups::bucket_type);
            for(const auto& kv : root_)
            {
                bytes += kv.second.blocks.capacity()*sizeof(Block);
                for(const Block& b : kv.second.blocks) bytes += b.cap*sizeof(T*);
            }
            return bytes;
        }

    private:
        static constexpr unsigned kCellsGrowth = 8;
        static constexpr double kFirstBox = 1.5;        // cells
        static constexpr double kMaxBoxProbes = 64;
        // Bounds are computed in rounded arithmetic: a margin far above its errors, so that a bound
        // never discards a cell that is closer (it costs nothing measurable)
        static constexpr double kPruneSlack = 1.0 + 1e-9;
        static constexpr double kStopSlack = 1.0 - 1e-9;
        static constexpr double kGrowSlack = 1.0 + 1e-6;

        struct Block
        {
            uint64_t occ[8] = {};
            uint64_t edge[8] = {};
            uint16_t prefix[8] = {};    // occupied voxels in the slices before
            T** cells = nullptr;
            uint16_t n = 0;
            uint16_t cap = 0;

            Block() = default;
            Block(const Block&) = delete;
            Block& operator=(const Block&) = delete;
            Block(Block&& o) noexcept { moveFrom(o); }
            Block& operator=(Block&& o) noexcept
            {
                if(this != &o)
                {
                    std::free(cells);
                    moveFrom(o);
                }
                return *this;
            }
            ~Block() { std::free(cells); }

            void moveFrom(Block& o)
            {
                std::memcpy(occ, o.occ, sizeof(occ));
                std::memcpy(edge, o.edge, sizeof(edge));
                std::memcpy(prefix, o.prefix, sizeof(prefix));
                cells = o.cells;
                n = o.n;
                cap = o.cap;
                o.cells = nullptr;
                o.n = o.cap = 0;
            }
        };

        struct SuperBlock
        {
            uint64_t mask[8] = {};      // non-empty blocks
            uint64_t edge_mask[8] = {}; // blocks with at least one edge cell
            uint16_t prefix[8] = {};
            int sx = 0, sy = 0, sz = 0;
            uint32_t n_edges = 0;
            std::vector<Block> blocks;
        };

        using Root = ankerl::unordered_dense::map<uint64_t, SuperBlock>;
        // Groups of 8x8x8 super-blocks: the mask of those that exist, for the searches far from the map
        using Groups = ankerl::unordered_dense::map<uint64_t, std::array<uint64_t, 8>>;

        struct Cand
        {
            double d2;
            VoxelCoord v;
            T* cell;
        };

        struct KnnQuery
        {
            const double* q;
            int k;
            bool edge_only;
            int lo[3], hi[3];  // voxels scanned (clamped to each block)
            std::vector<Cand>* heap;  // sorted, nearest first
            bool full() const { return (int)heap->size() == k; }
            double worst() const { return heap->back().d2; }
        };

        struct RadiusQuery
        {
            const double* q;
            double r2;
            bool edge_only;
            int lo[3], hi[3];
            std::vector<T*>* cells;
            std::vector<double>* d2;
        };

        static uint64_t superKey(int sx, int sy, int sz)
        {
            constexpr uint64_t m = (1ull << 21) - 1;
            return ((uint64_t(uint32_t(sx)) & m) << 42) | ((uint64_t(uint32_t(sy)) & m) << 21) | (uint64_t(uint32_t(sz)) & m);
        }

        static unsigned rank(const uint64_t* words, const uint16_t* prefix, unsigned w, unsigned bit)
        {
            return prefix[w] + (unsigned)__builtin_popcountll(words[w] & ((1ull << bit) - 1));
        }

        static void setBit(uint64_t* words, uint16_t* prefix, unsigned w, unsigned bit)
        {
            words[w] |= 1ull << bit;
            for(unsigned i = w + 1; i < 8; ++i) prefix[i]++;
        }

        static void clearBit(uint64_t* words, uint16_t* prefix, unsigned w, unsigned bit)
        {
            words[w] &= ~(1ull << bit);
            for(unsigned i = w + 1; i < 8; ++i) prefix[i]--;
        }

        static bool anyBit(const uint64_t* words)
        {
            uint64_t o = 0;
            for(int i = 0; i < 8; ++i) o |= words[i];
            return o != 0;
        }

        static double sq(double v) { return v*v; }

        const Block* findBlock(int bx, int by, int bz) const
        {
            auto it = root_.find(superKey(bx >> 3, by >> 3, bz >> 3));
            if(it == root_.end()) return nullptr;
            const SuperBlock& s = it->second;
            const unsigned bw = bz & 7, bbit = (bx & 7) + 8*(by & 7);
            if(!((s.mask[bw] >> bbit) & 1)) return nullptr;
            return &s.blocks[rank(s.mask, s.prefix, bw, bbit)];
        }

        // Squared distance from q to the box of the centres of voxels [lo, lo + n)
        double boxLowerBound(const int* lo, int n, const double* q) const
        {
            double d = 0;
            for(int a = 0; a < 3; ++a)
            {
                const double cmin = centre(lo[a]), cmax = centre(lo[a] + n - 1);
                const double g = q[a] < cmin ? cmin - q[a] : (q[a] > cmax ? q[a] - cmax : 0.0);
                d += g*g;
            }
            return d;
        }
        double blockLowerBound(int bx, int by, int bz, const double* q) const
        {
            const int lo[3] = {bx*8, by*8, bz*8};
            return boxLowerBound(lo, 8, q);
        }
        double superLowerBound(int sx, int sy, int sz, const double* q) const
        {
            const int lo[3] = {sx*64, sy*64, sz*64};
            return boxLowerBound(lo, 64, q);
        }

        // Voxels whose centres may lie within [q - h, q + h] on each axis: a superset, by a margin
        // above the float rounding of the centres
        void voxelRange(const double* q, double h, int* lo, int* hi) const
        {
            for(int a = 0; a < 3; ++a)
            {
                const double m = 0.01 + 2.5e-7*(std::abs(q[a]) + h)*inv_cs_;
                lo[a] = (int)std::floor((q[a] - h)*inv_cs_ - 0.5 - m);
                hi[a] = (int)std::floor((q[a] + h)*inv_cs_ - 0.5 + m);
            }
        }

        void radiusSuper(const SuperBlock& s, RadiusQuery& rq) const
        {
            int bl[3], bh[3];
            const int so[3] = {s.sx*8, s.sy*8, s.sz*8};
            for(int a = 0; a < 3; ++a)
            {
                bl[a] = std::max((rq.lo[a] >> 3) - so[a], 0);
                bh[a] = std::min((rq.hi[a] >> 3) - so[a], 7);
                if(bl[a] > bh[a]) return;
            }
            const uint64_t xrow = (0xFFull >> (7 - bh[0])) & (0xFFull << bl[0]);
            uint64_t xy = 0;
            for(int by = bl[1]; by <= bh[1]; ++by) xy |= xrow << (8*by);
            const uint64_t* smask = rq.edge_only ? s.edge_mask : s.mask;
            for(int bz = bl[2]; bz <= bh[2]; ++bz)
            {
                for(uint64_t bits = smask[bz] & xy; bits; bits &= bits - 1)
                {
                    const unsigned bbit = __builtin_ctzll(bits);
                    const int bx = so[0] + (int)(bbit & 7), by = so[1] + (int)(bbit >> 3), bzz = so[2] + bz;
                    if(blockLowerBound(bx, by, bzz, rq.q) >= rq.r2*kPruneSlack) continue;
                    radiusBlock(s.blocks[rank(s.mask, s.prefix, bz, bbit)], bx*8, by*8, bzz*8, rq);
                }
            }
        }

        void radiusBlock(const Block& b, int ox, int oy, int oz, RadiusQuery& rq) const
        {
            int vl[3], vh[3];
            const int o[3] = {ox, oy, oz};
            for(int a = 0; a < 3; ++a)
            {
                vl[a] = std::max(rq.lo[a] - o[a], 0);
                vh[a] = std::min(rq.hi[a] - o[a], 7);
                if(vl[a] > vh[a]) return;
            }
            const uint64_t xrow = (0xFFull >> (7 - vh[0])) & (0xFFull << vl[0]);
            const uint64_t* words = rq.edge_only ? b.edge : b.occ;
            const double* q = rq.q;
            double dxs[8];
            for(int lx = vl[0]; lx <= vh[0]; ++lx) dxs[lx] = centre(ox + lx) - q[0];
            for(int lz = vl[2]; lz <= vh[2]; ++lz)
            {
                const uint64_t wz = words[lz];
                if(!wz) continue;
                const double cz = centre(oz + lz), dz = cz - q[2], dz2 = dz*dz;
                for(int ly = vl[1]; ly <= vh[1]; ++ly)
                {
                    uint64_t row = (wz >> (8*ly)) & xrow;
                    if(!row) continue;
                    const double cy = centre(oy + ly), dy = cy - q[1];
                    // A lower bound of the squared distance of the row's cells (rounded additions of
                    // non-negative terms are monotone)
                    if(dy*dy + dz2 >= rq.r2) continue;
                    for(; row; row &= row - 1)
                    {
                        const unsigned lx = __builtin_ctzll(row);
                        const double d = sqDistDiff(dxs[lx], dy, dz);
                        if(d < rq.r2)
                        {
                            const unsigned bit = lx + 8*ly;
                            rq.cells->push_back(b.cells[rank(b.occ, b.prefix, lz, bit)]);
                            rq.d2->push_back(d);
                        }
                    }
                }
            }
        }

        static bool better(double d, const VoxelCoord& v, const Cand& c)
        {
            return d < c.d2 || (d == c.d2 && v < c.v);
        }

        static void offer(KnnQuery& kq, double d, const VoxelCoord& v, T* cell)
        {
            auto& h = *kq.heap;
            if(kq.full())
            {
                if(!better(d, v, h.back())) return;
                h.pop_back();
            }
            size_t i = h.size();
            h.push_back(Cand{d, v, cell});
            while(i > 0 && better(d, v, h[i - 1]))
            {
                h[i] = h[i - 1];
                --i;
            }
            h[i] = Cand{d, v, cell};
        }

        void scanBlock(const Block& b, int ox, int oy, int oz, KnnQuery& kq) const
        {
            int vl[3], vh[3];
            const int o[3] = {ox, oy, oz};
            for(int a = 0; a < 3; ++a)
            {
                vl[a] = std::max(kq.lo[a] - o[a], 0);
                vh[a] = std::min(kq.hi[a] - o[a], 7);
                if(vl[a] > vh[a]) return;
            }
            const uint64_t xrow = (0xFFull >> (7 - vh[0])) & (0xFFull << vl[0]);
            const uint64_t* words = kq.edge_only ? b.edge : b.occ;
            const double* q = kq.q;
            double dxs[8];
            for(int lx = vl[0]; lx <= vh[0]; ++lx) dxs[lx] = centre(ox + lx) - q[0];
            for(int lz = vl[2]; lz <= vh[2]; ++lz)
            {
                const uint64_t wz = words[lz];
                if(!wz) continue;
                const double cz = centre(oz + lz), dz = cz - q[2], dz2 = dz*dz;
                if(kq.full() && dz2 > kq.worst()*kPruneSlack) continue;
                for(int ly = vl[1]; ly <= vh[1]; ++ly)
                {
                    uint64_t row = (wz >> (8*ly)) & xrow;
                    if(!row) continue;
                    const double cy = centre(oy + ly), dy = cy - q[1];
                    if(kq.full() && dy*dy + dz2 > kq.worst()*kPruneSlack) continue;
                    for(; row; row &= row - 1)
                    {
                        const unsigned lx = __builtin_ctzll(row);
                        const double d = sqDistDiff(dxs[lx], dy, dz);
                        const VoxelCoord v{ox + (int)lx, oy + ly, oz + lz};
                        if(kq.full() && !better(d, v, kq.heap->back())) continue;
                        offer(kq, d, v, b.cells[rank(b.occ, b.prefix, lz, lx + 8*ly)]);
                    }
                }
            }
        }

        // The super-block's blocks within the query's voxel range
        void knnSuper(const SuperBlock& s, KnnQuery& kq) const
        {
            int bl[3], bh[3];
            const int so[3] = {s.sx*8, s.sy*8, s.sz*8};
            for(int a = 0; a < 3; ++a)
            {
                bl[a] = std::max((kq.lo[a] >> 3) - so[a], 0);
                bh[a] = std::min((kq.hi[a] >> 3) - so[a], 7);
                if(bl[a] > bh[a]) return;
            }
            const uint64_t xrow = (0xFFull >> (7 - bh[0])) & (0xFFull << bl[0]);
            uint64_t xy = 0;
            for(int by = bl[1]; by <= bh[1]; ++by) xy |= xrow << (8*by);
            const uint64_t* smask = kq.edge_only ? s.edge_mask : s.mask;
            for(int bw = bl[2]; bw <= bh[2]; ++bw)
            {
                for(uint64_t bits = smask[bw] & xy; bits; bits &= bits - 1)
                {
                    const unsigned bbit = __builtin_ctzll(bits);
                    const int bx = so[0] + (int)(bbit & 7), by = so[1] + (int)(bbit >> 3), bz = so[2] + bw;
                    if(kq.full() && blockLowerBound(bx, by, bz, kq.q) > kq.worst()*kPruneSlack) continue;
                    scanBlock(s.blocks[rank(s.mask, s.prefix, bw, bbit)], bx*8, by*8, bz*8, kq);
                }
            }
        }

        // Every cell, nearest first, with no voxel range: the groups of super-blocks nearest first,
        // in each the super-blocks nearest first, in each the blocks nearest first, so that the first
        // cells found prune the rest. Bounded by the number of groups (a few hundred at 10M cells)
        void nearestFirst(KnnQuery& kq) const
        {
            for(int a = 0; a < 3; ++a)
            {
                kq.lo[a] = std::numeric_limits<int>::min()/2;
                kq.hi[a] = std::numeric_limits<int>::max()/2;
            }
            thread_local std::vector<std::pair<double, uint64_t>> groups;
            groups.clear();
            for(const auto& kv : groups_)
            {
                const int gx = unpack(kv.first, 42), gy = unpack(kv.first, 21), gz = unpack(kv.first, 0);
                const int lo[3] = {gx*512, gy*512, gz*512};
                groups.emplace_back(boxLowerBound(lo, 512, kq.q), kv.first);
            }
            std::sort(groups.begin(), groups.end());
            thread_local std::vector<std::pair<double, const SuperBlock*>> supers;
            for(const auto& [glb, gkey] : groups)
            {
                if(kq.full() && glb > kq.worst()*kPruneSlack) break;
                const auto& mask = groups_.find(gkey)->second;
                const int gx = unpack(gkey, 42), gy = unpack(gkey, 21), gz = unpack(gkey, 0);
                supers.clear();
                for(unsigned w = 0; w < 8; ++w)
                {
                    for(uint64_t bits = mask[w]; bits; bits &= bits - 1)
                    {
                        const unsigned bit = __builtin_ctzll(bits);
                        const int sx = gx*8 + (int)(bit & 7), sy = gy*8 + (int)(bit >> 3), sz = gz*8 + (int)w;
                        const double lb = superLowerBound(sx, sy, sz, kq.q);
                        if(kq.full() && lb > kq.worst()*kPruneSlack) continue;
                        const SuperBlock& sb = root_.find(superKey(sx, sy, sz))->second;
                        if(kq.edge_only && sb.n_edges == 0) continue;
                        supers.emplace_back(lb, &sb);
                    }
                }
                std::sort(supers.begin(), supers.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
                for(const auto& [lb, sb] : supers)
                {
                    if(kq.full() && lb > kq.worst()*kPruneSlack) break;
                    superNearestFirst(*sb, kq);
                }
            }
        }

        void superNearestFirst(const SuperBlock& s, KnnQuery& kq) const
        {
            thread_local std::vector<std::pair<double, std::pair<const Block*, int>>> blocks;
            const uint64_t* smask = kq.edge_only ? s.edge_mask : s.mask;
            blocks.clear();
            for(unsigned bw = 0; bw < 8; ++bw)
            {
                for(uint64_t bits = smask[bw]; bits; bits &= bits - 1)
                {
                    const unsigned bbit = __builtin_ctzll(bits);
                    const int bi = s.sx*8 + (int)(bbit & 7), bj = s.sy*8 + (int)(bbit >> 3), bk = s.sz*8 + (int)bw;
                    const double blb = blockLowerBound(bi, bj, bk, kq.q);
                    if(kq.full() && blb > kq.worst()*kPruneSlack) continue;
                    // The block, with its position in the super-block
                    blocks.emplace_back(blb, std::make_pair(&s.blocks[rank(s.mask, s.prefix, bw, bbit)], (int)(bbit + 64*bw)));
                }
            }
            std::sort(blocks.begin(), blocks.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            for(const auto& [blb, bp] : blocks)
            {
                if(kq.full() && blb > kq.worst()*kPruneSlack) break;
                const int bbit = bp.second & 63, bw = bp.second >> 6;
                scanBlock(*bp.first, (s.sx*8 + (bbit & 7))*8, (s.sy*8 + (bbit >> 3))*8, (s.sz*8 + bw)*8, kq);
            }
        }

        // A coordinate packed by superKey (21 bits, sign extended)
        static int unpack(uint64_t key, int shift)
        {
            return (int)((int64_t)(key << (43 - shift)) >> 43);
        }

        double cs_, inv_cs_;
        float cs_f_, hcs_f_;
        Root root_;
        Groups groups_;
        size_t size_ = 0;
        size_t n_edges_ = 0;
};

} // namespace vdb
