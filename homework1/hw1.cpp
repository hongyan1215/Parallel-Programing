// HW1: Sokoban Solver - Macro-push search with OpenMP parallelization
// Build: g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1
// Usage: ./hw1 <input_file>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {}
#endif

using namespace std;

// ===================== Configuration =====================
static constexpr int THREADS = 6;
static constexpr int W_HEUR = 5;  // Weighted A* factor
static constexpr int BATCH_SIZE = 64;  // Batch size for parallel processing
static constexpr int MAX_DEPTH = 500;

// ===================== Tile Types =====================
enum TileType {
    WALL = 1,
    BOX = 2,
    TARGET = 4,
    SOKOBAN = 8,
    FRAGILE = 16,
    SPACE = 0
};

// ===================== Direction Constants =====================
static const int DY[4] = {-1, 0, +1, 0};  // Up, Left, Down, Right
static const int DX[4] = {0, -1, 0, +1};
static const char DIRC[4] = {'W', 'A', 'S', 'D'};

// ===================== Helper Functions =====================
static inline bool inBounds(int y, int x, int H, int W) {
    return (y >= 0 && y < H && x >= 0 && x < W);
}

static inline int idx2d(int y, int x, int W) {
    return y * W + x;
}

static inline pair<int, int> idx2coord(int idx, int W) {
    return {idx / W, idx % W};
}

// ===================== Static Map Information =====================
struct StaticMap {
    int H = 0, W = 0, N = 0;
    int nBoxes = 0;
    vector<uint8_t> isWall, isFragile, isGoal;
    vector<int> pushDist;  // Distance from each cell to nearest target
    vector<uint8_t> deadSquare;  // Precomputed dead squares
    vector<array<int, 4>> neighbors;  // 4-neighbors for each cell
    vector<uint8_t> validDirs;  // Valid directions for each cell
} SMI;

// ===================== Zobrist Hashing =====================
static vector<uint64_t> ZBbox, ZBplayer;
static void initZobrist() {
    mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    ZBbox.resize(SMI.N);
    ZBplayer.resize(SMI.N);
    for (int i = 0; i < SMI.N; ++i) {
        ZBbox[i] = rng();
        ZBplayer[i] = rng();
    }
}

// ===================== State Representation =====================
struct State {
    vector<int> boxes;  // Box positions (linear indices)
    int player;         // Player position (linear index)
    
    // Incremental hash caching
    mutable uint64_t cached_hash = 0;
    mutable bool hash_valid = false;
    
    inline uint64_t hash() const {
        if (!hash_valid) {
            uint64_t h = ZBplayer[player];
            for (int b : boxes) h ^= ZBbox[b];
            cached_hash = h;
            hash_valid = true;
        }
        return cached_hash;
    }
    
    // Apply move with incremental hash update
    inline void applyMove(int box_idx, int to) {
        uint64_t h = hash();
        int from_box = boxes[box_idx];
        
        // Update player (moves to box's old position)
        h ^= ZBplayer[player];
        player = from_box;
        h ^= ZBplayer[player];
        
        // Update box position
        h ^= ZBbox[from_box];
        boxes[box_idx] = to;
        h ^= ZBbox[to];
        
        cached_hash = h;
        hash_valid = true;
    }
    
    // Check if state is solved
    inline bool isSolved() const {
        for (int b : boxes) {
            if (!SMI.isGoal[b]) return false;
        }
        return true;
    }
    
    // Count boxes on goals
    inline int boxesOnGoals() const {
        int count = 0;
        for (int b : boxes) {
            if (SMI.isGoal[b]) count++;
        }
        return count;
    }
};

// ===================== Move Generation =====================
struct Move {
    int box_idx, to, pushFrom;
    char push_char;
};

// ===================== Predecessor Record =====================
struct Predecessor {
    uint64_t from_state_hash;  // Hash of the previous state
    Move move;                 // The move that led to this state
    int heuristic_value;       // Incremental heuristic value
    bool is_deadlock_pruned;   // Whether this move was pruned due to deadlock
    
    Predecessor() = default;
    Predecessor(uint64_t from_hash, const Move& m, int h_val, bool pruned = false)
        : from_state_hash(from_hash), move(m), heuristic_value(h_val), is_deadlock_pruned(pruned) {}
};

// ===================== Search Node =====================
struct Node {
    State s;
    int g = 0;  // Cost from start
    int h = 0;  // Heuristic estimate
    int f = 0;  // f = g + w*h
    
    Node() = default;
};

// ===================== Node Comparison for Priority Queue =====================
struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        return a.f != b.f ? a.f > b.f :
               a.h != b.h ? a.h > b.h : a.g > b.g;
    }
};

// ===================== Concurrent Priority Queue =====================
class ConcurrentPQ {
    priority_queue<Node, vector<Node>, NodeCmp> pq_;
    mutable mutex m_;
    
public:
    void push(Node n) {
        lock_guard<mutex> lk(m_);
        pq_.push(std::move(n));
    }
    
    void push_bulk(vector<Node>& nodes) {
        if (nodes.empty()) return;
        lock_guard<mutex> lk(m_);
        for (auto& n : nodes) pq_.push(std::move(n));
        nodes.clear();
    }
    
    bool try_pop(Node& out) {
        lock_guard<mutex> lk(m_);
        if (pq_.empty()) return false;
        out = std::move(const_cast<Node&>(pq_.top()));
        pq_.pop();
        return true;
    }
    
    size_t size() const {
        lock_guard<mutex> lk(m_);
        return pq_.size();
    }
    
    bool empty() const {
        lock_guard<mutex> lk(m_);
        return pq_.empty();
    }
};

// ===================== Bloom Filter for Visited States =====================
struct BloomFilter {
    size_t nbits, nwords;
    unique_ptr<atomic<uint64_t>[]> bits;
    
    BloomFilter(size_t nb = (1ULL << 24)) {
        nbits = nb;
        nwords = (nbits + 63) / 64;
        bits = make_unique<atomic<uint64_t>[]>(nwords);
        for (size_t i = 0; i < nwords; ++i) {
            bits[i].store(0, memory_order_relaxed);
        }
    }
    
    static inline uint64_t mix64(uint64_t z) {
        z ^= z >> 33;
        z *= 0xff51afd7ed558ccdULL;
        z ^= z >> 33;
        return z;
    }
    
    bool test_and_set(uint64_t h) {
        uint64_t h1 = mix64(h);
        uint64_t h2 = mix64(h ^ 0x9e3779b97f4a7c15ULL);
        size_t i1 = (h1 % nbits), i2 = (h2 % nbits);
        uint64_t m1 = 1ULL << (i1 & 63), m2 = 1ULL << (i2 & 63);
        size_t w1 = i1 >> 6, w2 = i2 >> 6;
        uint64_t o1 = bits[w1].fetch_or(m1, memory_order_relaxed);
        uint64_t o2 = bits[w2].fetch_or(m2, memory_order_relaxed);
        return (o1 & m1) && (o2 & m2);
    }
};

// ===================== Input Parser =====================
static inline bool isAllowedTile(char c) {
    return (c == 'x' || c == 'X' || c == 'o' || c == 'O' || 
            c == '.' || c == ' ' || c == '#' || c == '@' || c == '!');
}

static unsigned char charToBit(char c) {
    switch (c) {
        case '#': return WALL;
        case 'x': return BOX;
        case 'X': return BOX | TARGET;
        case '.': return TARGET;
        case '@': return FRAGILE;
        case 'o': return SOKOBAN;
        case 'O': return SOKOBAN | TARGET;
        case '!': return SOKOBAN | FRAGILE;
        default:  return SPACE;
    }
}

static pair<State, StaticMap> loadMap(const string& filename) {
    ifstream fin(filename);
    if (!fin) throw runtime_error("Failed to open input file");
    
    vector<string> lines;
    string line;
    while (getline(fin, line)) {
        lines.push_back(std::move(line));
    }
    if (lines.empty()) throw runtime_error("Input file is empty");
    
    int H = (int)lines.size();
    int W = (int)lines[0].size();
    int N = H * W;
    
    vector<uint8_t> wall(N, 0), fragile(N, 0), goal(N, 0);
    vector<int> boxes;
    int player = -1;
    
    int nBoxes = 0, nGoals = 0;
    for (int y = 0; y < H; ++y) {
        const string& row = lines[y];
        for (int x = 0; x < W; ++x) {
            char c = row[x];
            if (!isAllowedTile(c)) throw runtime_error("Invalid character in input");
            unsigned char b = charToBit(c);
            int id = y * W + x;
            
            if (b & WALL) wall[id] = 1;
            if (b & FRAGILE) fragile[id] = 1;
            if (b & TARGET) { goal[id] = 1; nGoals++; }
            if (b & BOX) { boxes.push_back(id); nBoxes++; }
            if (c == 'o' || c == 'O' || c == '!') player = id;
        }
    }
    
    if (player == -1) throw runtime_error("No player found");
    if (nBoxes != nGoals) throw runtime_error("Number of boxes != number of targets");
    
    StaticMap sm;
    sm.H = H; sm.W = W; sm.N = N; sm.nBoxes = nBoxes;
    sm.isWall = std::move(wall);
    sm.isFragile = std::move(fragile);
    sm.isGoal = std::move(goal);
    sm.pushDist.assign(N, INT_MAX);
    sm.deadSquare.assign(N, 0);
    
    // Precompute neighbors and valid directions
    sm.neighbors.resize(N);
    sm.validDirs.resize(N);
    for (int i = 0; i < N; ++i) {
        int y = i / W, x = i % W;
        uint8_t dirs = 0;
        for (int d = 0; d < 4; ++d) {
            int ny = y + DY[d], nx = x + DX[d];
            if (inBounds(ny, nx, H, W)) {
                sm.neighbors[i][d] = ny * W + nx;
                dirs |= (1 << d);
            } else {
                sm.neighbors[i][d] = -1;
            }
        }
        sm.validDirs[i] = dirs;
    }
    
    // Multi-source BFS from goals to compute push distances
    deque<int> dq;
    for (int i = 0; i < N; ++i) {
        if (sm.isGoal[i]) {
            sm.pushDist[i] = 0;
            dq.push_back(i);
        }
    }
    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();
        int du = sm.pushDist[u];
        uint8_t dirs = sm.validDirs[u];
        for (int d = 0; d < 4; ++d) {
            if (!(dirs & (1 << d))) continue;
            int v = sm.neighbors[u][d];
            if (sm.isWall[v] || sm.isFragile[v]) continue;
            if (sm.pushDist[v] > du + 1) {
                sm.pushDist[v] = du + 1;
                dq.push_back(v);
            }
        }
    }
    
    // Precompute dead squares (corners not on targets)
    for (int i = 0; i < N; ++i) {
        if (sm.isGoal[i]) continue;
        int y = i / W, x = i % W;
        bool U = (y == 0) || sm.isWall[i - W];
        bool D = (y == H - 1) || sm.isWall[i + W];
        bool L = (x == 0) || sm.isWall[i - 1];
        bool R = (x == W - 1) || sm.isWall[i + 1];
        if ((U && L) || (U && R) || (D && L) || (D && R)) {
            sm.deadSquare[i] = 1;
        }
    }
    
    State s;
    s.boxes = std::move(boxes);
    s.player = player;
    s.hash_valid = false;
    return {s, sm};
}

// ===================== Heuristic Function =====================
static int heuristic(const State& s) {
    int sum = 0;
    for (int b : s.boxes) {
        int d = SMI.pushDist[b];
        if (d == INT_MAX || SMI.deadSquare[b]) return INT_MAX / 4;
        sum += d;
        if (sum > INT_MAX / 4) return INT_MAX / 4;
    }
    return sum;
}

// ===================== Player Reachability BFS =====================
struct BFSGrid {
    vector<int8_t> prev;
};

static thread_local vector<uint8_t> occ_buffer;

static inline void buildBoxMask(const State& s, vector<uint8_t>& occ) {
    if (occ.size() != SMI.N) occ.resize(SMI.N);
    fill(occ.begin(), occ.end(), 0);
    for (int b : s.boxes) occ[b] = 1;
}

static BFSGrid playerBFS(const State& s) {
    BFSGrid g;
    g.prev.assign(SMI.N, -1);
    if (occ_buffer.size() != SMI.N) occ_buffer.resize(SMI.N);
    buildBoxMask(s, occ_buffer);
    
    deque<int> dq;
    dq.push_back(s.player);
    g.prev[s.player] = 8;  // Special marker for start
    
    while (!dq.empty()) {
        int u = dq.front();
        dq.pop_front();
        uint8_t dirs = SMI.validDirs[u];
        for (int d = 0; d < 4; ++d) {
            if (!(dirs & (1 << d))) continue;
            int v = SMI.neighbors[u][d];
            if (g.prev[v] != -1 || SMI.isWall[v] || occ_buffer[v]) continue;
            g.prev[v] = (int8_t)d;
            dq.push_back(v);
        }
    }
    return g;
}

static string reconstructPath(int start, int goal, const BFSGrid& g) {
    if (start == goal) return string();
    if (g.prev[goal] == -1) return string();
    
    string path;
    path.reserve(64);
    int cur = goal;
    while (cur != start) {
        int d = g.prev[cur];
        if (d < 0 || d > 3) break;
        path.push_back(DIRC[d]);
        cur = SMI.neighbors[cur][d ^ 2];  // Reverse direction
    }
    reverse(path.begin(), path.end());
    return path;
}

// ===================== Move Generation =====================
static vector<Move> generateMoves(const State& s, const BFSGrid& reach) {
    vector<Move> moves;
    moves.reserve(32);
    buildBoxMask(s, occ_buffer);
    
    for (size_t bi = 0; bi < s.boxes.size(); ++bi) {
        int b = s.boxes[bi];
        uint8_t dirs = SMI.validDirs[b];
        for (int d = 0; d < 4; ++d) {
            if (!(dirs & (1 << d))) continue;
            int to = SMI.neighbors[b][d];
            if (to < 0) continue;
            
            // Player needs to be in opposite direction
            if (!(dirs & (1 << (d ^ 2)))) continue;
            int pf = SMI.neighbors[b][d ^ 2];
            if (pf < 0) continue;
            
            // Check if push is valid
            if (SMI.isWall[to] || SMI.isFragile[to] || occ_buffer[to]) continue;
            if (SMI.isWall[pf] || occ_buffer[pf]) continue;
            if (pf != s.player && reach.prev[pf] == -1) continue;
            
            moves.push_back({(int)bi, to, pf, DIRC[d]});
        }
    }
    return moves;
}

// ===================== Deadlock Detection =====================
static bool is2x2Dead(const vector<uint8_t>& boxOcc, int id) {
    if (SMI.isGoal[id]) return false;
    int H = SMI.H, W = SMI.W;
    int y = id / W, x = id % W;
    
    auto isOcc = [&](int p) -> bool {
        return (p < 0 || p >= SMI.N) ? true : (SMI.isWall[p] || boxOcc[p]);
    };
    
    // Check 2x2 patterns
    if (y > 0 && x > 0 && isOcc(id) && isOcc(id - 1) && 
        isOcc(id - W) && isOcc(id - W - 1)) return true;
    if (y > 0 && x < W - 1 && isOcc(id) && isOcc(id + 1) && 
        isOcc(id - W) && isOcc(id - W + 1)) return true;
    if (y < H - 1 && x > 0 && isOcc(id) && isOcc(id - 1) && 
        isOcc(id + W) && isOcc(id + W - 1)) return true;
    if (y < H - 1 && x < W - 1 && isOcc(id) && isOcc(id + 1) && 
        isOcc(id + W) && isOcc(id + W + 1)) return true;
    return false;
}

// Additional deadlock: box adjacent to a wall forms a corridor with no targets
// Rule: If a box is against a wall on one side and not on a target, then along
// the perpendicular axis (until hitting a wall or fragile), at least one side
// must encounter a target; otherwise it's a deadlock.
static bool isWallCorridorDead(int id) {
    if (SMI.isGoal[id]) return false;
    int H = SMI.H, W = SMI.W;
    int y = id / W, x = id % W;

    auto inb = [&](int ry, int rx) -> bool { return ry >= 0 && ry < H && rx >= 0 && rx < W; };
    // Returns: 1 if hits goal before block with side intact; 0 if blocked (wall/fragile) with side intact and no goal; -1 if side breaks (not wall) before block → rule not applicable for this direction
    auto scanWithSideTri = [&](int dy, int dx, int sy, int sx) -> int {
        int cy = y + dy, cx = x + dx;
        while (inb(cy, cx)) {
            int p = cy * W + cx;
            // corridor side must remain a wall; if it opens, rule no longer applies
            int sY = cy + sy, sX = cx + sx;
            if (inb(sY, sX)) {
                if (!SMI.isWall[sY * W + sX]) return -1;
            }
            // stop conditions for the scan cell itself
            if (SMI.isWall[p] || SMI.isFragile[p]) return 0;
            if (SMI.isGoal[p]) return 1;
            cy += dy;
            cx += dx;
        }
        // Out of bounds in scan direction is effectively a wall block
        return 0;
    };

    bool leftWall = (x == 0) || SMI.isWall[id - 1];
    bool rightWall = (x == W - 1) || SMI.isWall[id + 1];
    bool upWall = (y == 0) || SMI.isWall[id - W];
    bool downWall = (y == H - 1) || SMI.isWall[id + W];

    // If against left or right wall, require a target in at least one vertical direction
    if (leftWall) {
        int up = scanWithSideTri(-1, 0, 0, -1);
        int down = scanWithSideTri(1, 0, 0, -1);
        // Only dead if both directions are blocked with side intact and no goals
        if (up == 0 && down == 0) return true;
    }
    if (rightWall) {
        int up = scanWithSideTri(-1, 0, 0, +1);
        int down = scanWithSideTri(1, 0, 0, +1);
        if (up == 0 && down == 0) return true;
    }
    // If against top or bottom wall, require a target in at least one horizontal direction
    if (upWall) {
        int left = scanWithSideTri(0, -1, -1, 0);
        int right = scanWithSideTri(0, +1, -1, 0);
        if (left == 0 && right == 0) return true;
    }
    if (downWall) {
        int left = scanWithSideTri(0, -1, +1, 0);
        int right = scanWithSideTri(0, +1, +1, 0);
        if (left == 0 && right == 0) return true;
    }
    return false;
}

// ===================== Path Reconstruction =====================
static string reconstructFullPath(uint64_t goal_hash, const unordered_map<uint64_t, Predecessor>& came_from, const State& start_state) {
    vector<Move> macro_moves;
    uint64_t current_hash = goal_hash;
    
    // Backtrack from goal to start
    while (came_from.count(current_hash)) {
        const Predecessor& pred = came_from.at(current_hash);
        macro_moves.push_back(pred.move);
        current_hash = pred.from_state_hash;
    }
    
    // Reverse to get correct order (start -> goal)
    reverse(macro_moves.begin(), macro_moves.end());
    
    // Reconstruct full path by simulating each macro move
    string full_path;
    State current_state = start_state;
    
    for (const Move& macro_move : macro_moves) {
        // Get player BFS to find path to push position
        BFSGrid reach = playerBFS(current_state);
        string walk_path = reconstructPath(current_state.player, macro_move.pushFrom, reach);
        
        if (macro_move.pushFrom != current_state.player && walk_path.empty()) {
            throw runtime_error("Cannot reach push position during reconstruction");
        }
        
        // Add walk path and push move
        full_path += walk_path;
        full_path.push_back(macro_move.push_char);
        
        // Apply the move to current state
        current_state.applyMove(macro_move.box_idx, macro_move.to);
    }
    
    return full_path;
}

// ===================== Main Solver =====================
static string solve(const string& filename) {
    auto [s0, sm] = loadMap(filename);
    SMI = std::move(sm);
    initZobrist();
    omp_set_num_threads(THREADS);
    
    // Check if already solved
    if (s0.isSolved()) return string();
    
    int h0 = heuristic(s0);
    if (h0 >= INT_MAX / 8) throw runtime_error("Unsolvable (initial heuristic)");
    
    Node start;
    start.s = s0;
    start.g = 0;
    start.h = h0;
    start.f = start.g + W_HEUR * start.h;
    
    ConcurrentPQ open;
    open.push(start);
    
    BloomFilter bloom;
    bloom.test_and_set(start.s.hash());
    
    // Thread-safe predecessor tracking
    unordered_map<uint64_t, Predecessor> came_from;
    mutex came_from_mtx;
    
    atomic<bool> found(false);
    uint64_t goal_state_hash = 0;
    mutex ans_mtx;
    
    while (!open.empty() && !found.load(memory_order_relaxed)) {
        vector<vector<Node>> thread_children(THREADS);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            vector<Node> local_out;
            local_out.reserve(BATCH_SIZE * 3);
            
            for (int k = 0; k < BATCH_SIZE && !found.load(memory_order_relaxed); ++k) {
                Node cur;
                if (!open.try_pop(cur)) break;
                if (cur.g > MAX_DEPTH) continue;
                
                if (cur.s.isSolved()) {
                    lock_guard<mutex> lk(ans_mtx);
                    if (!found.load()) {
                        found.store(true);
                        goal_state_hash = cur.s.hash();
                    }
                    break;
                }
                
                BFSGrid reach = playerBFS(cur.s);
                vector<Move> moves = generateMoves(cur.s, reach);
                if (moves.empty()) continue;
                
                // Score and filter moves
                struct Scored {
                    Move m;
                    int nh;
                    bool toGoal;
                    int gain;
                };
                vector<Scored> scored;
                scored.reserve(moves.size());
                
                for (const auto& mv : moves) {
                    int from = cur.s.boxes[mv.box_idx];
                    
                    // Simulate move for deadlock check
                    occ_buffer[from] = 0;
                    occ_buffer[mv.to] = 1;
                    
                    bool dead = (SMI.deadSquare[mv.to] || is2x2Dead(occ_buffer, mv.to) || isWallCorridorDead(mv.to));
                    
                    // Restore
                    occ_buffer[mv.to] = 0;
                    occ_buffer[from] = 1;
                    
                    if (dead) continue;
                    
                    // Incremental heuristic
                    int df = SMI.pushDist[from];
                    int dt = SMI.pushDist[mv.to];
                    if (dt == INT_MAX) continue;
                    int nh = cur.h - df + dt;
                    if (nh >= INT_MAX / 8) continue;
                    
                    bool fromGoal = SMI.isGoal[from];
                    bool toGoal = SMI.isGoal[mv.to];
                    int gain = cur.h - nh;
                    
                    scored.push_back({mv, nh, toGoal, gain});
                }
                if (scored.empty()) continue;
                
                // Sort by priority: goals first, then by gain
                auto cmp = [](const Scored& A, const Scored& B) {
                    if (A.toGoal != B.toGoal) return A.toGoal > B.toGoal;
                    if (A.gain != B.gain) return A.gain > B.gain;
                    return A.nh < B.nh;
                };
                sort(scored.begin(), scored.end(), cmp);
                
                // Process top moves
                int take = min(12, (int)scored.size());
                for (int i = 0; i < take; ++i) {
                    const auto& sc = scored[i];
                    
                    // Check if player can reach push position (without reconstructing path)
                    if (sc.m.pushFrom != cur.s.player && reach.prev[sc.m.pushFrom] == -1) continue;
                    
                    // Create new state
                    State ns = cur.s;
                    ns.applyMove(sc.m.box_idx, sc.m.to);
                    uint64_t new_hash = ns.hash();
                    
                    if (bloom.test_and_set(new_hash)) continue;
                    
                    // Record predecessor (thread-safe, only write once per state)
                    {
                        lock_guard<mutex> lk(came_from_mtx);
                        if (came_from.find(new_hash) == came_from.end()) {
                            came_from[new_hash] = Predecessor(cur.s.hash(), sc.m, sc.nh, false);
                        }
                    }
                    
                    Node nxt;
                    nxt.s = std::move(ns);
                    nxt.g = cur.g + 1;
                    nxt.h = sc.nh;
                    nxt.f = nxt.g + W_HEUR * nxt.h;
                    
                    if (nxt.s.isSolved()) {
                        lock_guard<mutex> lk(ans_mtx);
                        if (!found.load()) {
                            found.store(true);
                            goal_state_hash = new_hash;
                        }
                        break;
                    }
                    local_out.push_back(std::move(nxt));
                }
                if (found.load()) break;
            }
            thread_children[tid] = std::move(local_out);
        }
        
        if (found.load()) break;
        for (int t = 0; t < THREADS; ++t) {
            if (!thread_children[t].empty()) {
                open.push_bulk(thread_children[t]);
            }
        }
    }
    
    if (!found.load()) throw runtime_error("No solution found");
    
    // Reconstruct full path from goal to start
    return reconstructFullPath(goal_state_hash, came_from, s0);
}

// ===================== Main Function =====================
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    if (argc < 2) {
        cerr << "Usage: ./hw1 <input_file>\n";
        return 1;
    }
    
    try {
        string ans = solve(argv[1]);
        cout << ans << '\n';
        cout.flush();
        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 2;
    }
}
