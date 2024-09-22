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
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#endif



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
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
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
    NNArr<int> T;

    void load(std::istream& in) {
        in >> N >> C;
        assert(8 <= N && N <= 30);
        for (int y = 0; y < NMAX; y++) {
            S[y].fill(WALL);
            T[y].fill(WALL);
        }
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                in >> S[y][x];
            }
        }
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                in >> T[y][x];
            }
        }
    }

    void load(const int seed) {
        std::ifstream ifs(format("../../tester/in/%d.in", seed));
        load(ifs);
    }

}

struct Move {
    // yyyyyxxxxxpppnnn : 16bit
    uint16_t data;
    Move() = default;
    Move(int y, int x, int pc, int nc) { set(y, x, pc, nc); }
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

namespace NBeam {

    struct Operation {
        uint64_t b64; // placeability
        int y, x, c;
    };

    struct Flips {

        static constexpr size_t NMAX_FLIPS = 128; // 30x30 の盤面なら最大 107 個裏返るはず

        std::array<Move, NMAX_FLIPS> flips;
        int nflips;

        inline void reset() { nflips = 0; }
        inline void set(int y, int x, int pc, int nc) {
            flips[nflips++].set(y, x, pc, nc);
        }

    };

    struct State {

        Board S;
        short placed;
        short matched;

        void initialize() {
            S.initialize(NInput::S);
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

        inline int calc_score() const {
            return placed + (int)matched * matched;
        }

        // c*8+d bit 目が立っている -> c を置くことで方向 d を裏返せる
        uint64_t check_placeability(int y, int x) const {
            uint64_t b64 = 0;
            if (S.get(y, x)) return b64;
            for (int d = 0; d < 8; d++) {
                int ny = y + dy[d], nx = x + dx[d];
                if (S.get(ny, nx) <= 0) continue;
                int c = S.get(ny, nx); // c 以外は置ける可能性がある
                while (true) {
                    ny += dy[d];
                    nx += dx[d];
                    if (S.get(ny, nx) <= 0) break;
                    if (S.get(ny, nx) != c) {
                        b64 |= 1ULL << (S.get(ny, nx) * 8 + d); // 色 b[ny][nx] は b[y][x] に置くことができる
                    }
                }
            }
            return b64;
        }

        void try_change(int& p, int& m, int y, int x, int c) const {
            using namespace NInput;
            p += int(S.get(y, x) == 0);
            m += T[y][x] ? (int(c == T[y][x]) - int(S.get(y, x) == T[y][x])) : 0;
        }

        int try_move(const Operation& op) const {
            using namespace NInput;
            const auto& [b64, y, x, c] = op;
            assert(!S.get(y, x));
            int nplaced = placed, nmatched = matched;
            try_change(nplaced, nmatched, y, x, c);
            int b8 = (b64 >> (c * 8)) & 0xFF;
            for (int d = 0; d < 8; d++) if (b8 >> d & 1) {
                int ny = y + dy[d], nx = x + dx[d];
                while (S.get(ny, nx) != c) {
                    try_change(nplaced, nmatched, ny, nx, c);
                    ny += dy[d];
                    nx += dx[d];
                }
            }
            return nplaced + nmatched * nmatched;
        }

        void change(int y, int x, int c, Flips& flips) {
            using namespace NInput;
            placed += int(S.get(y, x) == 0);
            matched += T[y][x] ? (int(c == T[y][x]) - int(S.get(y, x) == T[y][x])) : 0;
            flips.set(y, x, S.get(y, x), c);
            S.set(y, x, c);
        }

        void apply_move(const Operation& op, Flips& flips) {
            flips.reset();
            const auto& [b64, y, x, c] = op;
            assert(!S.get(y, x));
            change(y, x, c, flips);
            int b8 = (b64 >> (c * 8)) & 0xFF;
            for (int d = 0; d < 8; d++) if (b8 >> d & 1) {
                int ny = y + dy[d], nx = x + dx[d];
                while (S.get(ny, nx) != c) {
                    change(ny, nx, c, flips);
                    ny += dy[d];
                    nx += dx[d];
                }
            }
        }

        std::tuple<bool, int, Operation, Flips> apply_move_greedy() {
            using namespace NInput;
            int max_score = INT_MIN;
            Operation best_op;
            for (int y = 1; y <= N; y++) {
                for (int x = 1; x <= N; x++) {
                    auto b64 = check_placeability(y, x);
                    if (!b64) continue;
                    for (int c = 1; c <= C; c++) {
                        if ((b64 >> (c * 8)) & 0xFF) {
                            Operation op{ b64, y, x, c };
                            int score = try_move(op);
                            if (chmax(max_score, score)) {
                                best_op = op;
                            }
                        }
                    }
                }
            }
            Flips flips;
            if (max_score == INT_MIN) return { false, max_score, best_op, flips };
            apply_move(best_op, flips);
            return { true, max_score, best_op, flips };
        }

        std::vector<std::tuple<int, int, int>> run() {
            std::vector<int> scores;
            std::vector<Flips> flips_list;
            while (true) {
                auto [succeed, score, op, flips] = apply_move_greedy();
                if (!succeed) break;
                scores.push_back(score);
                flips_list.push_back(flips);
            }
            int best_idx = (int)std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));

            std::vector<std::tuple<int, int, int>> moves;
            for (int i = 0; i <= best_idx; i++) {
                const auto& flips = flips_list[i];
                auto [y, x, pc, nc] = flips.flips[0].to_tuple();
                moves.emplace_back(y, x, nc);
            }
            return moves;
        }

    };

}

struct State {

    static constexpr int BUFSIZE = 131072;

    //NNArr<int> S;

    Board S;

    int placed;
    int matched;

    int pointer;
    std::array<Move, BUFSIZE> move_stack;

    int best_score = 0;

    State() { initialize(); }

    void initialize() {
        S.initialize(NInput::S);
        placed = 0;
        matched = compute_matched();
        pointer = 0;
    }

    inline int eval() const {
        return placed + matched * matched;
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

    // c*8+d bit 目が立っている -> c を置くことで方向 d を裏返せる
    uint64_t can_place(int y, int x) {
        uint64_t b64 = 0;
        if (S.get(y, x)) return b64;
        for (int d = 0; d < 8; d++) {
            int ny = y + dy[d], nx = x + dx[d];
            if (S.get(ny, nx) <= 0) continue;
            int c = S.get(ny, nx); // c 以外は置ける可能性がある
            while (true) {
                ny += dy[d];
                nx += dx[d];
                if (S.get(ny, nx) <= 0) break;
                if (S.get(ny, nx) != c) {
                    b64 |= 1ULL << (S.get(ny, nx) * 8 + d); // 色 b[ny][nx] は b[y][x] に置くことができる
                }
            }
        }
        return b64;
    }

    void dry_change(int& p, int& m, int y, int x, int c) const {
        using namespace NInput;
        p += int(S.get(y, x) == 0);
        m += T[y][x] ? (int(c == T[y][x]) - int(S.get(y, x) == T[y][x])) : 0;
    }

    int calc_diff(uint64_t b64, int y, int x, int c) const {
        using namespace NInput;
        assert(!S.get(y, x));
        int nplaced = placed, nmatched = matched;
        dry_change(nplaced, nmatched, y, x, c);
        int b8 = (b64 >> (c * 8)) & 0xFF;
        for (int d = 0; d < 8; d++) if (b8 >> d & 1) {
            int ny = y + dy[d], nx = x + dx[d];
            while (S.get(ny, nx) != c) {
                dry_change(nplaced, nmatched, ny, nx, c);
                ny += dy[d];
                nx += dx[d];
            }
        }
        return nplaced + nmatched * nmatched - eval();
    }

    void change(int y, int x, int c) {
        using namespace NInput;
        placed += int(S.get(y, x) == 0);
        matched += T[y][x] ? (int(c == T[y][x]) - int(S.get(y, x) == T[y][x])) : 0;
        move_stack[pointer++].set(y, x, S.get(y, x), c);
        S.set(y, x, c);
    }

    void place(uint64_t b64, int y, int x, int c) {
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

    bool place_greedy() {
        using namespace NInput;
        int max_diff = INT_MIN, max_y, max_x, max_c;
        uint64_t max_b64;
        for (int y = 1; y <= N; y++) {
            for (int x = 1; x <= N; x++) {
                auto b64 = can_place(y, x);
                if (!b64) continue;
                for (int c = 1; c <= C; c++) {
                    if ((b64 >> (c * 8)) & 0xFF) {
                        int diff = calc_diff(b64, y, x, c);
                        if (chmax(max_diff, diff)) {
                            max_y = y;
                            max_x = x;
                            max_c = c;
                            max_b64 = b64;
                        }
                    }
                }
            }
        }
        if (max_diff == INT_MIN) return false;
        place(max_b64, max_y, max_x, max_c);
        return true;
    }

    std::vector<std::tuple<int, int, int>> to_moves() const {
        std::vector<std::tuple<int, int, int>> moves;
        for (int i = 0; i < pointer; i++) {
            auto [y, x, pc, nc] = move_stack[i].to_tuple();
            if (!pc) {
                moves.emplace_back(y, x, nc);
            }
        }
        return moves;
    }

    inline bool undo_single() {
        using namespace NInput;
        pointer--;
        auto [y, x, pc, nc] = move_stack[pointer].to_tuple();
        placed -= int(pc == 0);
        matched -= T[y][x] ? (int(nc == T[y][x]) - int(pc == T[y][x])) : 0;
        S.set(y, x, pc);
        return pc == 0; // pc==0: placing / pc!=0: reversing
    }

    inline void undo() {
        while (!undo_single());
    }

    inline void undo(int p) {
        while (pointer != p) { undo(); }
    }

    void run() {
        int best_score = 0;
        int best_pointer = -1;
        while (place_greedy()) {
            //dump(placed, pointer, placed, matched, eval());
            if (chmax(best_score, eval())) {
                best_pointer = pointer;
                dump(best_pointer, best_score);
            }
        }
        undo(best_pointer);
    }

};

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
    const int seed = 2;

    if (LOCAL_MODE) {
        NInput::load(seed);
    }
    else {
        NInput::load(std::cin);
    }

    NBeam::State state;
    state.initialize();
    auto moves = state.run();

    //State state;
    //state.run();

    //auto moves = state.to_moves();
    std::cout << moves.size() << '\n';
    for (const auto& [y, x, c] : moves) {
        std::cout << y - 1 << ' ' << x - 1 << ' ' << c << '\n';
    }

    return 0;
}