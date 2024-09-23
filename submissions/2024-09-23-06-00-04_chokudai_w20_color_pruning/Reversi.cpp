#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <filesystem>
#include <intrin.h>
#include <omp.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
int __builtin_popcount(int bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}
/* enable __uint128_t in MSVC */
#include <boost/multiprecision/cpp_int.hpp>
using __uint128_t = boost::multiprecision::uint128_t;
#endif

namespace {
    using namespace std;

    namespace HashMapImpl {
        using u32 = uint32_t;
        using u64 = uint64_t;

        template <typename Key, typename Data>
        struct HashMapBase;

        template <typename Key, typename Data>
        struct itrB
            : iterator<bidirectional_iterator_tag, Data, ptrdiff_t, Data*, Data&> {
            using base =
                iterator<bidirectional_iterator_tag, Data, ptrdiff_t, Data*, Data&>;
            using ptr = typename base::pointer;
            using ref = typename base::reference;

            u32 i;
            HashMapBase<Key, Data>* p;

            explicit constexpr itrB() : i(0), p(nullptr) {}
            explicit constexpr itrB(u32 _i, HashMapBase<Key, Data>* _p) : i(_i), p(_p) {}
            explicit constexpr itrB(u32 _i, const HashMapBase<Key, Data>* _p)
                : i(_i), p(const_cast<HashMapBase<Key, Data>*>(_p)) {}
            friend void swap(itrB& l, itrB& r) { swap(l.i, r.i), swap(l.p, r.p); }
            friend bool operator==(const itrB& l, const itrB& r) { return l.i == r.i; }
            friend bool operator!=(const itrB& l, const itrB& r) { return l.i != r.i; }
            const ref operator*() const {
                return const_cast<const HashMapBase<Key, Data>*>(p)->data[i];
            }
            ref operator*() { return p->data[i]; }
            ptr operator->() const { return &(p->data[i]); }

            itrB& operator++() {
                assert(i != p->cap && "itr::operator++()");
                do {
                    i++;
                    if (i == p->cap) break;
                    if (p->occupied_flag[i] && !p->deleted_flag[i]) break;
                } while (true);
                return (*this);
            }
            itrB operator++(int) {
                itrB it(*this);
                ++(*this);
                return it;
            }
            itrB& operator--() {
                do {
                    i--;
                    if (p->occupied_flag[i] && !p->deleted_flag[i]) break;
                    assert(i != 0 && "itr::operator--()");
                } while (true);
                return (*this);
            }
            itrB operator--(int) {
                itrB it(*this);
                --(*this);
                return it;
            }
        };

        template <typename Key, typename Data>
        struct HashMapBase {
            using u32 = uint32_t;
            using u64 = uint64_t;
            using iterator = itrB<Key, Data>;
            using itr = iterator;

        protected:
            template <typename K>
            inline u64 randomized(const K& key) const {
                return u64(key) ^ r;
            }

            template <typename K,
                enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
                enable_if_t<is_integral<K>::value, nullptr_t> = nullptr>
            inline u32 inner_hash(const K& key) const {
                return (randomized(key) * 11995408973635179863ULL) >> shift;
            }
            template <
                typename K, enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
                enable_if_t<is_integral<decltype(K::first)>::value, nullptr_t> = nullptr,
                enable_if_t<is_integral<decltype(K::second)>::value, nullptr_t> = nullptr>
            inline u32 inner_hash(const K& key) const {
                u64 a = randomized(key.first), b = randomized(key.second);
                a *= 11995408973635179863ULL;
                b *= 10150724397891781847ULL;
                return (a + b) >> shift;
            }
            template <typename K,
                enable_if_t<is_same<K, Key>::value, nullptr_t> = nullptr,
                enable_if_t<is_integral<typename K::value_type>::value, nullptr_t> =
                nullptr>
            inline u32 inner_hash(const K& key) const {
                static constexpr u64 mod = (1LL << 61) - 1;
                static constexpr u64 base = 950699498548472943ULL;
                u64 res = 0;
                for (auto& elem : key) {
                    __uint128_t x = __uint128_t(res) * base + (randomized(elem) & mod);
                    res = (x & mod) + (x >> 61);
                }
                __uint128_t x = __uint128_t(res) * base;
                res = (x & mod) + (x >> 61);
                if (res >= mod) res -= mod;
                return res >> (shift - 3);
            }

            template <typename D = Data,
                enable_if_t<is_same<D, Key>::value, nullptr_t> = nullptr>
            inline u32 hash(const D& dat) const {
                return inner_hash(dat);
            }
            template <
                typename D = Data,
                enable_if_t<is_same<decltype(D::first), Key>::value, nullptr_t> = nullptr>
            inline u32 hash(const D& dat) const {
                return inner_hash(dat.first);
            }

            template <typename D = Data,
                enable_if_t<is_same<D, Key>::value, nullptr_t> = nullptr>
            inline Key data_to_key(const D& dat) const {
                return dat;
            }
            template <
                typename D = Data,
                enable_if_t<is_same<decltype(D::first), Key>::value, nullptr_t> = nullptr>
            inline Key data_to_key(const D& dat) const {
                return dat.first;
            }

            void reallocate(u32 ncap) {
                vector<Data> ndata(ncap);
                vector<bool> nf(ncap);
                shift = 64 - __lg(ncap);
                for (u32 i = 0; i < cap; i++) {
                    if (occupied_flag[i] && !deleted_flag[i]) {
                        u32 h = hash(data[i]);
                        while (nf[h]) h = (h + 1) & (ncap - 1);
                        ndata[h] = move(data[i]);
                        nf[h] = true;
                    }
                }
                data.swap(ndata);
                occupied_flag.swap(nf);
                cap = ncap;
                occupied = s;
                deleted_flag.resize(cap);
                fill(std::begin(deleted_flag), std::end(deleted_flag), false);
            }

            inline bool extend_rate(u32 x) const { return x * 2 >= cap; }

            inline bool shrink_rate(u32 x) const {
                return HASHMAP_DEFAULT_SIZE < cap && x * 10 <= cap;
            }

            inline void extend() { reallocate(cap << 1); }

            inline void shrink() { reallocate(cap >> 1); }

        public:
            u32 cap, s, occupied;
            vector<Data> data;
            vector<bool> occupied_flag, deleted_flag;
            u32 shift;
            static u64 r;
            static constexpr uint32_t HASHMAP_DEFAULT_SIZE = 4;

            explicit HashMapBase()
                : cap(HASHMAP_DEFAULT_SIZE),
                s(0),
                occupied(0),
                data(cap),
                occupied_flag(cap),
                deleted_flag(cap),
                shift(64 - __lg(cap)) {}

            itr begin() const {
                u32 h = 0;
                while (h != cap) {
                    if (occupied_flag[h] && !deleted_flag[h]) break;
                    h++;
                }
                return itr(h, this);
            }
            itr end() const { return itr(this->cap, this); }

            friend itr begin(const HashMapBase& h) { return h.begin(); }
            friend itr end(const HashMapBase& h) { return h.end(); }

            itr find(const Key& key) const {
                u32 h = inner_hash(key);
                while (true) {
                    if (occupied_flag[h] == false) return this->end();
                    if (data_to_key(data[h]) == key) {
                        if (deleted_flag[h] == true) return this->end();
                        return itr(h, this);
                    }
                    h = (h + 1) & (cap - 1);
                }
            }

            bool contain(const Key& key) const { return find(key) != this->end(); }

            itr insert(const Data& d) {
                u32 h = hash(d);
                while (true) {
                    if (occupied_flag[h] == false) {
                        if (extend_rate(occupied + 1)) {
                            extend();
                            h = hash(d);
                            continue;
                        }
                        data[h] = d;
                        occupied_flag[h] = true;
                        ++occupied, ++s;
                        return itr(h, this);
                    }
                    if (data_to_key(data[h]) == data_to_key(d)) {
                        if (deleted_flag[h] == true) {
                            data[h] = d;
                            deleted_flag[h] = false;
                            ++s;
                        }
                        return itr(h, this);
                    }
                    h = (h + 1) & (cap - 1);
                }
            }

            // tips for speed up :
            // if return value is unnecessary, make argument_2 false.
            itr erase(itr it, bool get_next = true) {
                if (it == this->end()) return this->end();
                s--;
                if (!get_next) {
                    this->deleted_flag[it.i] = true;
                    if (shrink_rate(s)) shrink();
                    return this->end();
                }
                itr nxt = it;
                nxt++;
                this->deleted_flag[it.i] = true;
                if (shrink_rate(s)) {
                    Data d = data[nxt.i];
                    shrink();
                    it = find(data_to_key(d));
                }
                return nxt;
            }

            itr erase(const Key& key) { return erase(find(key)); }

            int count(const Key& key) { return find(key) == end() ? 0 : 1; }

            bool empty() const { return s == 0; }

            int size() const { return s; }

            void clear() {
                fill(std::begin(occupied_flag), std::end(occupied_flag), false);
                fill(std::begin(deleted_flag), std::end(deleted_flag), false);
                s = occupied = 0;
            }

            void reserve(int n) {
                if (n <= 0) return;
                n = 1 << min(23, __lg(n) + 2);
                if (cap < u32(n)) reallocate(n);
            }
        };

        template <typename Key, typename Data>
        uint64_t HashMapBase<Key, Data>::r =
            chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now().time_since_epoch())
            .count();

    }  // namespace HashMapImpl

    /**
     * @brief Hash Map(base)　(ハッシュマップ・基底クラス)
     */

    template <typename Key>
    struct HashSet : HashMapImpl::HashMapBase<Key, Key> {
        using HashMapImpl::HashMapBase<Key, Key>::HashMapBase;
    };

}

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '['; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << ']'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << "[" << p.first << ", " << p.second << "]"; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << "["; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << "]"; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << "["; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << "]"; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format_string(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
//#ifdef _MSC_VER
#define ENABLE_DUMP
//#endif
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.8e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.8e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next_u64(); }
    inline uint64_t next_u64() { x ^= x << 7; return x ^= x >> 9; }
    inline uint32_t next_u32() { return next_u64() >> 32; }
    inline uint32_t next_u32(uint32_t mod) { return ((uint64_t)next_u32() * mod) >> 32; }
    inline uint32_t next_u32(uint32_t l, uint32_t r) { return l + next_u32(r - l + 1); }
    inline double next_double() { return next_u32() * e; }
    inline double next_double(double c) { return next_double() * c; }
    inline double next_double(double l, double r) { return next_double(r - l) + l; }
private:
    static constexpr uint32_t M = UINT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { auto r = rnd.next_u32(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
std::string join(const std::string& delim, const std::vector<std::string>& elems) {
    if (elems.empty()) return "";
    std::string res = elems[0];
    for (int i = 1; i < (int)elems.size(); i++) {
        res += delim + elems[i];
    }
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif

struct LogTable {
    static constexpr int M = 65536;
    static constexpr int mask = M - 1;
    double l[M];
    LogTable() : l() {
        unsigned long long x = 88172645463325252ULL;
        double log_u64max = log(2) * 64;
        for (int i = 0; i < M; i++) {
            x = x ^ (x << 7);
            x = x ^ (x >> 9);
            l[i] = log(double(x)) - log_u64max;
        }
    }
    inline double operator[](int i) const { return l[i & mask]; }
} log_table;



constexpr int dy[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
constexpr int dx[] = { 0, 1, 1, 1, 0, -1, -1, -1 };

constexpr int NMAX = 32;
constexpr int CMAX = 6;
constexpr int WALL = -1;
constexpr int EMPTY = 0;

template<typename T> using NArr = std::array<T, NMAX>;
template<typename T> using NNArr = std::array<NArr<T>, NMAX>;

namespace NInput {

    int N;
    int C;
    NNArr<int> S;
    NNArr<int> T; // NOTE: 空きマスは -1
    std::array<int, 8> NT;
    int num_empty;
    int perfect_score;

    void load(std::istream& in) {
        in >> N >> C;
        assert(8 <= N && N <= 30);
        for (int y = 0; y < NMAX; y++) {
            S[y].fill(WALL);
            T[y].fill(WALL);
        }
        NT.fill(0);
        num_empty = 0;
        perfect_score = 0;
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                in >> S[y][x];
                num_empty += S[y][x] == 0;
            }
        }
        int ntarget = 0;
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                in >> T[y][x];
                if (T[y][x] >= 0) NT[T[y][x]]++;
                ntarget += T[y][x] > 0;
            }
        }
        perfect_score = ntarget + ntarget * ntarget;
        dump(perfect_score);
    }

    void load(const int seed) {
        std::ifstream ifs(::format_string("../../tester/in/%d.in", seed));
        load(ifs);
    }

}

namespace NHash {

    std::array<std::array<std::array<uint64_t, 8>, NMAX>, NMAX> table;

    void initialize() {
        Xorshift rnd;
        for (int y = 0; y < NMAX; y++) {
            for (int x = 0; x < NMAX; x++) {
                for (int c = 0; c < 8; c++) {
                    table[y][x][c] = rnd.next_u64();
                }
            }
        }
    }

}

struct Flip {
    // yyyyyxxxxxpppnnn : 16bit
    uint16_t data;
    Flip() = default;
    Flip(int y, int x, int pc, int nc) { set(y, x, pc, nc); }
    inline void set(int y, int x, int pc, int nc) {
        data = (uint16_t(y) << 11) | (uint16_t(x) << 6) | (uint16_t(pc) << 3) | uint16_t(nc);
    }
    std::tuple<int, int, int, int> to_tuple() const {
        int nc = data & 0b111; nc = (nc == 7) ? -1 : nc;
        int pc = (data >> 3) & 0b111; pc = (pc == 7) ? -1 : pc;
        int x = (data >> 6) & 0b11111;
        int y = data >> 11;
        return { y, x, pc, nc };
    }
};

struct Flips {
    static constexpr size_t MAX_FLIPS = 128;
    std::array<Flip, MAX_FLIPS> data;
    size_t sz = 0;
    inline void set(int y, int x, int pc, int nc) { data[sz++].set(y, x, pc, nc); }
    inline void reset() { sz = 0; }
};

struct Board {

    static constexpr size_t SIZE = NMAX * NMAX * 3;

    std::bitset<SIZE> data;

    void initialize(const NNArr<int>& S) {
        for (int y = 0; y < NMAX; y++) {
            for (int x = 0; x < NMAX; x++) {
                set(y, x, S[y][x]);
            }
        }
    }

    inline void set(int y, int x, int c) {
        c = (c == -1) ? 7 : c;
        int p = ((y << 5) | x) * 3;
        data[p] = c & 1;
        data[p + 1] = (c >> 1) & 1;
        data[p + 2] = (c >> 2) & 1;
    }

    inline int get(int y, int x) const {
        int p = ((y << 5) | x) * 3;
        int c = int(data[p]) + (int(data[p + 1]) << 1) + (int(data[p + 2]) << 2);
        return c == 7 ? -1 : c;
    }

};

struct Operation {
    uint64_t b64; // placeability
    int y, x, c;
};

struct State {

    Board S;
    std::array<short, 8> NS;
    short placed;
    short matched;
    uint64_t hash;

    void initialize() {
        S.initialize(NInput::S);
        NS.fill(0);
        hash = 0;
        for (int y = 1; y <= NInput::N; y++) {
            for (int x = 1; x <= NInput::N; x++) {
                int c = NInput::S[y][x];
                if (c >= 0) {
                    NS[c]++;
                    hash ^= NHash::table[y][x][c];
                }
            }
        }
        placed = 0;
        matched = compute_matched();
    }

    inline int compute_matched() const {
        using namespace NInput;
        int m = 0;
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                if (T[y][x] <= 0) continue;
                m += S.get(y, x) == T[y][x];
            }
        }
        return m;
    }

    inline double calc_score() const {
        return placed + (int)matched * matched;
    }

    // c*8+d bit 目が立っている -> c を置くことで方向 d を裏返せる
    uint64_t check_placeability(int y, int x) const {
        uint64_t b64 = 0;
        if (S.get(y, x)) return b64;
        for (int d = 0; d < 8; d++) {
            int ny = y + dy[d], nx = x + dx[d];
            int c = S.get(ny, nx);
            if (c <= 0) continue;
            // c 以外は置ける可能性がある
            while (true) {
                ny += dy[d];
                nx += dx[d];
                int nc = S.get(ny, nx);
                if (nc <= 0) break;
                if (nc != c) {
                    b64 |= 1ULL << (nc * 8 + d); // 色 b[ny][nx] は b[y][x] に置くことができる
                }
            }
        }
        return b64;
    }

    void try_change(std::array<short, 8>& NNS, uint64_t& h, int& p, int& m, int y, int x, int nc) const {
        using namespace NInput;
        int pc = S.get(y, x);
        p += int(pc == 0);
        m += T[y][x] ? (int(nc == T[y][x]) - int(pc == T[y][x])) : 0;
        h ^= NHash::table[y][x][pc] ^ NHash::table[y][x][nc];
        NNS[pc]--; NNS[nc]++;
    }

    inline bool is_pruned(const std::array<short, 8>& NNS) const {
        using namespace NInput;
        for (int c = 1; c <= C; c++) {
            if (NT[c] && !NNS[c]) return true;
        }
        return false;
    }

    std::pair<double, uint64_t> try_move(const Operation& op) const {
        using namespace NInput;
        auto NNS(NS);
        const auto& [b64, y, x, c] = op;
        assert(!S.get(y, x));
        uint64_t nhash = hash;
        int nplaced = placed, nmatched = matched;
        try_change(NNS, nhash, nplaced, nmatched, y, x, c);
        int b8 = (b64 >> (c * 8)) & 0xFF;
        for (int d = 0; d < 8; d++) if (b8 >> d & 1) {
            int ny = y + dy[d], nx = x + dx[d];
            while (S.get(ny, nx) != c) {
                try_change(NNS, nhash, nplaced, nmatched, ny, nx, c);
                ny += dy[d];
                nx += dx[d];
            }
        }
        double nscore = is_pruned(NNS) ? -1e9 : nplaced + nmatched * nmatched;
        return { nscore, nhash };
    }

    void change(int y, int x, int nc) {
        using namespace NInput;
        int pc = S.get(y, x);
        placed += int(pc == 0);
        matched += T[y][x] ? (int(nc == T[y][x]) - int(pc == T[y][x])) : 0;
        NS[pc]--; NS[nc]++;
        S.set(y, x, nc);
        hash ^= NHash::table[y][x][pc] ^ NHash::table[y][x][nc];
    }

    void apply_move(const Operation& op) {
        const auto& [b64, y, x, c] = op;
        assert(!S.get(y, x));
        change(y, x, c);
        int b8 = (b64 >> (c * 8)) & 0xFF;
        for (int d = 0; d < 8; d++) if (b8 >> d & 1) {
            int ny = y + dy[d], nx = x + dx[d];
            while (S.get(ny, nx) != c) {
                change(ny, nx, c);
                ny += dy[d];
                nx += dx[d];
            }
        }
    }

};

struct History;
using HistoryPtr = std::shared_ptr<History>;
struct History {
    Operation op;
    HistoryPtr parent;
    History(const Operation& op_, HistoryPtr parent_)
        : op(op_), parent(parent_) {}
};

struct Stack {

    HistoryPtr head;

    Operation top() { return head->op; }
    Stack push(const Operation& op) {
        return Stack({ std::make_shared<History>(op, head) });
    }
    Stack pop() {
        return Stack({ head->parent });
    }
};

struct Node {

    State state;
    Stack move_history;

    Node() = default;
    Node(const State& state_) : state(state_) {}
    void advance(const Operation& op) {
        state.apply_move(op);
    }

    bool operator<(const Node& rhs) const {
        auto s1 = state.calc_score(), s2 = rhs.state.calc_score();
        return s1 == s2 ? state.hash > rhs.state.hash : s1 > s2;
    }
};

struct TemporaryNode {

    double score;
    int node_index;
    Operation op;

    TemporaryNode(double score_, int node_index_, const Operation& op_)
        : score(score_), node_index(node_index_), op(op_) {}
};

Node beam_search(const State& initial_state, const int beam_width) {
    using namespace NInput;

    std::vector<Node> nodes, next_nodes;
    nodes.emplace_back(initial_state);
    nodes.back().move_history = Stack{ nullptr };

    std::vector<TemporaryNode> temp_nodes;
    HashSet<uint64_t> seen;
    
    double best_score = initial_state.calc_score();
    Node best_node = nodes.back();

    for (int turn = 1;; turn++) {
        temp_nodes.clear();
        seen.clear();

        for (int node_index = 0; node_index < (int)nodes.size(); node_index++) {
            const auto& state = nodes[node_index].state;
            for (int y = 1; y <= N; y++) {
                for (int x = 1; x <= N; x++) {
                    auto b64 = state.check_placeability(y, x);
                    if (!b64) continue;
                    for (int c = 1; c <= C; c++) {
                        if ((b64 >> (c << 3)) & 0xFF) {
                            Operation op{ b64, y, x, c };
                            auto [nscore, nhash] = state.try_move(op);
                            if (seen.count(nhash)) continue;
                            temp_nodes.emplace_back(nscore, node_index, op);
                            seen.insert(nhash);
                        }
                    }
                }
            }
        }

        int node_size = (int)temp_nodes.size();

        if (node_size == 0) break;

        if (node_size > beam_width) {
            std::nth_element(temp_nodes.begin(), temp_nodes.begin() + beam_width, temp_nodes.end(),
                [](const TemporaryNode& n1, const TemporaryNode& n2) {
                    return n1.score > n2.score;
                }
            );
        }

        for (int i = 0; i < std::min(beam_width, node_size); i++) {
            int node_index = temp_nodes[i].node_index;
            next_nodes.emplace_back(nodes[node_index]);
            next_nodes.back().advance(temp_nodes[i].op);
            next_nodes.back().move_history = nodes[node_index].move_history.push(temp_nodes[i].op);
        }

        std::swap(nodes, next_nodes);
        next_nodes.clear();

        for (int i = 0; i < (int)nodes.size(); i++) {
            if (chmax(best_score, nodes[i].state.calc_score())) {
                best_node = nodes[i];
            }
        }
        dump(turn, best_score);
    }

    return best_node;
}

Node chokudai_search(const State& initial_state, const int beam_width, double duration) {
    using namespace NInput;

    Timer timer;

    std::vector<std::set<Node>> turn_to_nodes(num_empty + 1);
    std::vector<HashSet<uint64_t>> turn_to_hashes(num_empty + 1);
    double best_score;
    Node best_node;
    {
        Node initial_node(initial_state);
        initial_node.move_history = Stack{ nullptr };
        turn_to_nodes[0].insert(initial_node);
        turn_to_hashes[0].insert(initial_node.state.hash);
        best_score = initial_state.calc_score();
        best_node = initial_node;
    }

    std::vector<TemporaryNode> temp_nodes;

    int next_dump_time = 100, dump_interval = 100;
    while (true) {
        for (int turn = 0; turn < num_empty; turn++) {
            auto elapsed = timer.elapsed_ms();
            if (elapsed > next_dump_time) {
                dump(elapsed, best_score);
                next_dump_time += dump_interval;
            }
            if (elapsed > duration) {
                return best_node;
            }
            auto& nodes = turn_to_nodes[turn];
            auto& next_nodes = turn_to_nodes[turn + 1];
            auto& next_hashes = turn_to_hashes[turn + 1];
            auto thresh = next_nodes.empty() ? -1 : next_nodes.rbegin()->state.calc_score();
            if (nodes.empty()) continue;
            auto node = *nodes.begin();
            nodes.erase(nodes.begin());
            const auto& state = node.state;
            for (int y = 1; y <= N; y++) {
                for (int x = 1; x <= N; x++) {
                    auto b64 = state.check_placeability(y, x);
                    if (!b64) continue;
                    for (int c = 1; c <= C; c++) {
                        if ((b64 >> (c << 3)) & 0xFF) {
                            Operation op{ b64, y, x, c };
                            auto [nscore, nhash] = state.try_move(op);
                            if (nscore < thresh) continue;
                            if (next_hashes.count(nhash)) continue;
                            auto next_node(node);
                            next_node.advance(op);
                            next_node.move_history = node.move_history.push(op);
                            next_nodes.insert(next_node);
                            next_hashes.insert(nhash);
                            if (chmax(best_score, nscore)) {
                                best_node = next_node;
                            }
                            //if (seen.count(nhash)) continue;
                            //temp_nodes.emplace_back(nscore, node_index, op);
                            //seen.insert(nhash);
                        }
                    }
                }
            }
            while (next_nodes.size() > beam_width) {
                next_nodes.erase(std::prev(next_nodes.end()));
            }
        }
    }

    return best_node;
}



int main(int argc, char** argv) {

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    // 通常オセロの四隅のように、一度置いたら二度と裏返せない"急所"が存在する
    // 盤面がプレイアウトで生成されることから、急所に異なる色が配置されるようなケースは枝刈りしてよい
    // 盤面の評価も、急所のみで行ったり急所の評価を重めにする等したほうがよい

    // 一次元オセロで位置 x にある種類 k のトークンを裏返せるか？
    // セルには 空白セル・壁セル・種類 k のトークン・種類 k 以外のトークン がある

    // 両端の少なくとも一方が壁セル：stable
    // 両端は壁セルではないとする
    // x にいくつ種類 k のトークンが隣接しても一つとみなしてよいので、両端は種類 k のトークンでもないとする
    // 両端が空白：両端に種類 k 以外のトークンを配置すれば裏返るので、unstable
    // 一方が空白で他方がトークン k'!=k：空白に k' を配置すれば裏返るので、unstable
    // 両端が k 以外のトークン：

    const bool LOCAL_MODE = argc > 1 && std::string(argv[1]) == "local";
    const int seed = 1;

    if (LOCAL_MODE) {
        NInput::load(seed);
    }
    else {
        NInput::load(std::cin);
    }

    NHash::initialize();

    {
        State state;
        state.initialize();

        auto result = chokudai_search(state, 20, 9000);

        //auto result = beam_search(state, 10);

        std::vector<Operation> moves;
        Stack move_history = result.move_history;
        while (move_history.head) {
            Operation op = move_history.top();
            moves.emplace_back(op);
            move_history = move_history.pop();
        }
        std::reverse(moves.begin(), moves.end());

        std::cout << moves.size() << '\n';
        for (const auto& [b64, y, x, c] : moves) {
            std::cout << y - 1 << ' ' << x - 1 << ' ' << c << '\n';
        }
    }

    //auto moves = state.run();

    //std::cout << moves.size() << '\n';
    //for (const auto& [y, x, c] : moves) {
    //    std::cout << y - 1 << ' ' << x - 1 << ' ' << c << '\n';
    //}

    return 0;
}