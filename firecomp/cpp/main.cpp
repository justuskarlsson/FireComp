#include <robin_hood.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/Parallel.h>

#include <cstdio>

#include <queue>
#include <tuple>
#include <vector>
#include <optional>
#include <atomic>
#include <chrono>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;

template <
    typename K,
    typename V,
    typename Hash = std::hash<K>,
    typename KeyEqual = std::equal_to<K>>
using unordered_map = robin_hood::unordered_flat_map<K, V, Hash, KeyEqual>;
// using unordered_map = std::unordered_map<K, V, Hash, KeyEqual>;

struct Pbar
{
    int n;
    int i;
    std::vector<int> checks = std::vector<int>(1001, 0);
    int cur_check;
    char line_buf[64];
    std::string title;

    std::chrono::steady_clock::time_point start_time;

    void start(int n, const std::string &title)
    {
        this->n = n;
        this->i = 0;
        this->start_time = std::chrono::steady_clock::now();
        for (int i = 0; i < 1000; i++)
        {
            checks[i] = (n - 1) * (i / 1000.0f);
        }
        checks.back() = (int)2e9;
        this->title = title;
        this->cur_check = 0;
    }

    void update(int delta = 1)
    {
        i += delta;
        if (i >= checks[cur_check])
        {
            int next_check = cur_check;
            while (i >= checks[next_check])
            {
                next_check++;
            }
            // TODO: Calculate vel and work done based on next_check
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            int m = elapsed / 60;
            int s = elapsed % 60;
            float percentage = (next_check - 1) / 10.0f;
            float ratio = percentage * 0.01f;
            int time_total = elapsed / (ratio + 1e-6);
            int time_left = time_total - elapsed;
            int ml = time_left / 60;
            int sl = time_left % 60;
            snprintf(line_buf, 64, "%s: %.1f%% %02d:%02d->%02d:%02d", title.c_str(), percentage, m, s, ml, sl);
            printf("\r\033[2K%-60.60s", line_buf);
            fflush(stdout);
            cur_check = next_check;
        }
    }

    void stop()
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        int m = elapsed / 60;
        int s = elapsed % 60;
        printf("\r\033[2K %s: 100%% %02d:%02d\n", title.c_str(), m, s);
        fflush(stdout);
    }
};

int UNKNOWN = 0;
int NON_FIRE = 1;
int FIRE = 2;

torch::Tensor get_deltas(int t_range, int y_range, int x_range)
{
    int n = (2 * t_range + 1) * (2 * x_range + 1) * (2 * y_range + 1) - 1;
    torch::Tensor deltas = torch::zeros({n, 3}, torch::dtype(torch::kInt32));
    int i = 0;
    for (int t = -t_range; t < t_range + 1; t++)
    {
        for (int y = -y_range; y < y_range + 1; y++)
        {
            for (int x = -x_range; x < x_range + 1; x++)
            {
                if (t == 0 && x == 0 && y == 0)
                    continue;
                deltas[i][0] = t;
                deltas[i][1] = y;
                deltas[i][2] = x;
                i++;
            }
        }
    }
    return deltas;
}

struct Component
{
    int idx;
    int size;
    int num_fire;
    // bbox3d
    int t_min;
    int t_max;
    int y_min;
    int y_max;
    int x_min;
    int x_max;
    float avg_xy_neighbors;
    float ignition_ratio;
    float t_ratio;

    py::dict to_dict() const
    {
        return py::dict(
            "idx"_a = idx,
            "size"_a = size,
            "num_fire"_a = num_fire,
            "t_min"_a = t_min,
            "t_max"_a = t_max,
            "y_min"_a = y_min,
            "y_max"_a = y_max,
            "x_min"_a = x_min,
            "x_max"_a = x_max,
            "avg_xy_neighbors"_a = avg_xy_neighbors,
            "ignition_ratio"_a = ignition_ratio,
            "t_ratio"_a = t_ratio);
    }
};

struct SearchCellKey
{
    int x, y, t;
    bool operator==(const SearchCellKey &other) const
    {
        return x == other.x && y == other.y && t == other.t;
    }
};

struct KeyHash
{
    // pack to 64-bit then splitmix64
    static inline uint64_t splitmix64(uint64_t z)
    {
        z += 0x9e3779b97f4a7c15ull;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    size_t operator()(SearchCellKey const &k) const
    {
        // 22+21+21 bits fits typical ranges; adjust if needed
        uint64_t u = (uint64_t)(uint32_t)k.x | ((uint64_t)(uint32_t)k.y << 22) | ((uint64_t)(uint32_t)k.t << 43);
        return (size_t)splitmix64(u);
    }
};

struct SearchCell
{
    int cls;
    int component;
    uint8_t xy_neighbors = 0;
    bool ignition = true;
};

torch::Tensor single_bfs_deltas = get_deltas(7, 2, 2);
torch::Tensor xy_deltas = get_deltas(0, 2, 2);

Component
single_bfs(SearchCellKey src, unordered_map<SearchCellKey, SearchCell, KeyHash> &cells, int component_idx)
{

    // Initialize component
    Component component{
        component_idx, // idx
        0,             // size
        1,             // num_fire: will find its way back to src
        src.t, src.t, src.y, src.y, src.x, src.x};

    // Precompute neighbor deltas
    auto deltas_a = single_bfs_deltas.accessor<int, 2>();
    int n_deltas = single_bfs_deltas.size(0);

    // BFS
    std::queue<SearchCellKey> q, q_false_ignition;
    q.push(src);
    std::vector<SearchCellKey> all_nodes;
    all_nodes.push_back(src);
    while (!q.empty())
    {
        SearchCellKey key = q.front();
        q.pop();

        for (int i = 0; i < n_deltas; i++)
        {
            int x2 = key.x + deltas_a[i][2];
            int y2 = key.y + deltas_a[i][1];
            int t2 = key.t + deltas_a[i][0];

            SearchCellKey neighbor_key{x2, y2, t2};
            auto it = cells.find(neighbor_key);
            if (it == cells.end())
                continue;
            SearchCell &neighbor_cell = it->second;
            if (neighbor_cell.cls != FIRE)
                continue;

            int dt = deltas_a[i][0];
            if (dt > 0)
            { // if cell->neighbor is future, neighbor wasn't an ignition
                neighbor_cell.ignition = false;
                // Also mark all neighbors of false ignition as false
                q_false_ignition.push(neighbor_key);
            }
            if (dt == 0)
            {
                neighbor_cell.xy_neighbors++;
            }

            if (neighbor_cell.component != -1)
                continue;

            // Mark visited
            neighbor_cell.component = component_idx;
            component.num_fire++;
            // Update bounding box
            component.t_min = std::min(component.t_min, t2);
            component.t_max = std::max(component.t_max, t2);
            component.y_min = std::min(component.y_min, y2);
            component.y_max = std::max(component.y_max, y2);
            component.x_min = std::min(component.x_min, x2);
            component.x_max = std::max(component.x_max, x2);

            all_nodes.push_back(neighbor_key);
            q.push({x2, y2, t2});
        }
    }
    auto xy_deltas_a = xy_deltas.accessor<int, 2>();
    int n_xy_deltas = xy_deltas.size(0);
    unordered_map<SearchCellKey, bool, KeyHash> false_ignition_visited;
    while (!q_false_ignition.empty())
    {
        SearchCellKey key = q_false_ignition.front();
        q_false_ignition.pop();
        false_ignition_visited[key] = true;
        for (int i = 0; i < n_xy_deltas; i++)
        {
            int x2 = key.x + xy_deltas_a[i][2];
            int y2 = key.y + xy_deltas_a[i][1];
            int t2 = key.t;
            SearchCellKey neighbor_key{x2, y2, t2};
            if (false_ignition_visited[neighbor_key])
                continue;
            auto it = cells.find(neighbor_key);
            if (it == cells.end())
                continue;
            SearchCell &neighbor_cell = it->second;
            neighbor_cell.ignition = false;
            q_false_ignition.push(neighbor_key);
            false_ignition_visited[neighbor_key] = true;
        }
    }
    // Find number ignition components
    std::vector<SearchCellKey> ignited_sources;
    for (auto &key : all_nodes)
    {
        SearchCell &cell = cells[key];
        if (cell.ignition)
        {
            ignited_sources.push_back(key);
        }
    }
    component.ignition_ratio = 0.0f;
    for (auto &src : ignited_sources)
    {
        SearchCell &cell = cells[src];
        if (cell.ignition)
        {
            component.ignition_ratio += 1.0f;
        }
        else
        {
            continue;
        }
        cell.ignition = false;
        std::queue<SearchCellKey> q;
        q.push(src);
        while (!q.empty())
        {
            SearchCellKey key = q.front();
            q.pop();
            for (int i = 0; i < n_xy_deltas; i++)
            {
                int x2 = key.x + xy_deltas_a[i][2];
                int y2 = key.y + xy_deltas_a[i][1];
                int t2 = key.t;
                SearchCellKey neighbor_key{x2, y2, t2};
                auto it = cells.find(neighbor_key);
                if (it == cells.end())
                    continue;
                SearchCell &neighbor_cell = it->second;
                if (!neighbor_cell.ignition)
                {
                    continue;
                }
                neighbor_cell.ignition = false;
                q.push(neighbor_key);
            }
        }
    }
    for (auto &key : ignited_sources)
    {
        cells[key].ignition = true;
    }
    return component;
}

/**
 * fire_cls: T Y X : 0-9
 * nodes: TYX 3
 */
std::tuple<std::vector<py::dict>, torch::Tensor, torch::Tensor, torch::Tensor> search(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor width,
    torch::Tensor height,
    torch::Tensor cls,
    torch::Tensor t)
{
    std::vector<Component> components_list;
    auto x_a = x.accessor<float, 1>();
    auto y_a = y.accessor<float, 1>();
    auto width_a = width.accessor<float, 1>();
    auto height_a = height.accessor<float, 1>();
    auto cls_a = cls.accessor<int8_t, 1>();
    auto t_a = t.accessor<int32_t, 1>();
    int n = x.size(0);
    Pbar pbar;
    int DT = 3;
    unordered_map<SearchCellKey, SearchCell, KeyHash> cells;
    cells.reserve(n * 2);
    pbar.start(n, "Creating cells");
    for (int i = 0; i < n; i++)
    {
        pbar.update();
        float x = x_a[i];
        float y = y_a[i];
        float width = width_a[i];
        float height = height_a[i];
        float p = 0.0;
        for (int x2 = std::round(x - width / 2 - p); x2 <= std::round(x + width / 2 + p); x2++)
        {
            for (int y2 = std::round(y - height / 2 - p); y2 <= std::round(y + height / 2 + p); y2++)
            {
                SearchCellKey key{x2, y2, t_a[i]};
                // replaces any 'old' values
                cells[key] = SearchCell{cls_a[i], -1};
            }
        }
    }
    pbar.stop();
    torch::Tensor pixel_to_fire = torch::zeros({n}, torch::dtype(torch::kInt32));
    torch::Tensor pixel_ignition = torch::zeros({n}, torch::dtype(torch::kBool));
    auto pixel_to_fire_a = pixel_to_fire.accessor<int, 1>();
    auto pixel_ignition_a = pixel_ignition.accessor<bool, 1>();
    int component_idx = 0;

    pbar.start(cells.size(), "BFS");
    for (auto &[key, cell] : cells)
    {
        if (cell.component != -1)
        {
            continue;
        }
        cell.component = component_idx;
        Component component = single_bfs(key, cells, component_idx);
        pbar.update(component.num_fire);

        components_list.push_back(component);
        component_idx++;
    }
    pbar.stop();
    pbar.start(n, "Assigning pixels");
    for (int i = 0; i < n; i++)
    {
        pbar.update();
        int x = std::round(x_a[i]);
        int y = std::round(y_a[i]);
        int t = t_a[i];
        SearchCellKey key{x, y, t};
        auto it = cells.find(key);
        if (it == cells.end())
        {
            std::cout << "warning, key not found: " << std::endl;
            continue;
        }
        SearchCell &cell = it->second;
        if (cell.component == -1)
        {
            std::cout << "warning, cell still has -1" << std::endl;
        }
        pixel_to_fire_a[i] = cell.component;
        pixel_ignition_a[i] = cell.ignition;
    }
    pbar.stop();
    unordered_map<int, size_t> num_xy_neighbors, cell_counts;
    for (auto &[key, cell] : cells)
    {
        int c = cell.component;
        num_xy_neighbors[c] += cell.xy_neighbors;
        cell_counts[c]++;
    }
    unordered_map<int, float> avg_xy_neighbors;
    for (auto &[c, count] : cell_counts)
    {
        avg_xy_neighbors[c] = num_xy_neighbors[c] / (double)count;
    }

    unordered_map<int, int> counts;
    for (int i = 0; i < n; i++)
    {
        int comp = pixel_to_fire_a[i];
        counts[comp]++;
    }

    std::vector<py::dict> component_dicts;
    component_dicts.reserve(components_list.size());
    pbar.start(components_list.size(), "To pydict");
    for (auto &component : components_list)
    {
        pbar.update();
        int c = component.idx;
        component.size = counts[c];
        component.num_fire = counts[c];
        component.avg_xy_neighbors = avg_xy_neighbors[c];
        component.ignition_ratio = component.ignition_ratio / (float)component.num_fire;
        float xy_area = (component.x_max - component.x_min + 1) * (component.y_max - component.y_min + 1);
        int t_length = component.t_max - component.t_min + 1;
        component.t_ratio = t_length / std::sqrt(xy_area);
        component_dicts.push_back(component.to_dict());
    }
    pbar.stop();
    int n_cells = cells.size();
    torch::Tensor xytci = torch::zeros({n_cells, 5}, torch::dtype(torch::kInt32));
    auto xytci_a = xytci.accessor<int, 2>();
    int i = 0;
    pbar.start(n_cells, "To xytci");
    for (auto [k, cell] : cells)
    {
        pbar.update();
        xytci_a[i][0] = k.x;
        xytci_a[i][1] = k.y;
        xytci_a[i][2] = k.t;
        xytci_a[i][3] = cell.component;
        xytci_a[i][4] = cell.ignition;
        i++;
    }
    pbar.stop();
    return std::make_tuple(component_dicts, pixel_to_fire, pixel_ignition, xytci);
}

std::tuple<torch::Tensor, std::vector<int>> group_by_fire(
    torch::Tensor xytci,
    torch::Tensor sizes_per_c,
    int min_size)
{
    int n = xytci.size(0);
    int num_c = sizes_per_c.size(0);
    auto sizes_per_c_a = sizes_per_c.accessor<int, 1>();
    auto xytci_a = xytci.accessor<int, 2>();
    std::vector<int> c_to_idx(num_c, -1);
    int cur_ok = 0;
    for (int i = 0; i < num_c; i++)
    {
        int size = sizes_per_c_a[i];
        if (size >= min_size)
        {
            c_to_idx[i] = cur_ok;
            cur_ok++;
        }
    }
    std::vector<int> sizes(cur_ok, 0);
    for (int i = 0; i < n; i++)
    {
        int c = xytci_a[i][3];
        if (c_to_idx[c] == -1)
            continue;
        int idx = c_to_idx[c];
        sizes[idx]++;
    }

    int tot_size = 0;
    for (int s : sizes)
        tot_size += s;

    std::vector<int> offsets(cur_ok, 0);
    for (int i = 1; i < cur_ok; i++)
    {
        offsets[i] = offsets[i - 1] + sizes[i - 1];
    }
    std::vector<int> cur_count(cur_ok, 0);
    torch::Tensor xytci_by_fire = torch::zeros({tot_size, 5}, torch::dtype(torch::kInt32));
    auto xytci_by_fire_a = xytci_by_fire.accessor<int, 2>();
    for (int i = 0; i < n; i++)
    {
        int c = xytci_a[i][3];
        if (c_to_idx[c] == -1)
            continue;
        int idx = c_to_idx[c];
        int offset = offsets[idx] + cur_count[idx];
        if (offset >= tot_size)
        {
            std::cout << "err, offset: " << offset << " tot_size: " << tot_size << std::endl;
        }
        xytci_by_fire_a[offset][0] = xytci_a[i][0];
        xytci_by_fire_a[offset][1] = xytci_a[i][1];
        xytci_by_fire_a[offset][2] = xytci_a[i][2];
        xytci_by_fire_a[offset][3] = c;
        xytci_by_fire_a[offset][4] = xytci_a[i][4];
        cur_count[idx]++;
    }
    return std::make_tuple(xytci_by_fire, sizes);
}

std::tuple<torch::Tensor, std::vector<int>> group_by_time(torch::Tensor xytci)
{
    int n = xytci.size(0);
    auto xytci_a = xytci.accessor<int, 2>();
    int max_t = xytci.select(1, 2).max().item<int>();
    int num_t = max_t + 1;
    std::vector<int> sizes(num_t, 0);
    for (int i = 0; i < n; i++)
    {
        int t = xytci_a[i][2];
        sizes[t]++;
    }
    std::vector<int> offsets(num_t, 0);
    for (int i = 1; i < num_t; i++)
    {
        offsets[i] = offsets[i - 1] + sizes[i - 1];
    }
    std::vector<int> cur_count(num_t, 0);
    torch::Tensor xytci_by_t = torch::zeros({n, 5}, torch::dtype(torch::kInt32));
    auto xytci_by_t_a = xytci_by_t.accessor<int, 2>();
    for (int i = 0; i < n; i++)
    {
        int t = xytci_a[i][2];
        int offset = offsets[t] + cur_count[t];
        xytci_by_t_a[offset][0] = xytci_a[i][0];
        xytci_by_t_a[offset][1] = xytci_a[i][1];
        xytci_by_t_a[offset][2] = t;
        xytci_by_t_a[offset][3] = xytci_a[i][3];
        xytci_by_t_a[offset][4] = xytci_a[i][4];
        cur_count[t]++;
    }
    return std::make_tuple(xytci_by_t, sizes);
}

std::tuple<torch::Tensor, std::vector<int>> group_by_blocks(
    int block_num_x,
    int block_num_y,
    torch::Tensor block_idx,
    torch::Tensor local_x,
    torch::Tensor local_y,
    torch::Tensor data // 2d int32[...]

)
{
    int n = block_idx.size(0);
    auto block_idx_a = block_idx.accessor<int, 1>();
    auto local_x_a = local_x.accessor<int, 1>();
    auto local_y_a = local_y.accessor<int, 1>();
    auto data_a = data.accessor<int, 1>();

    // padding, maybe not worth it.
    std::vector<int> block_sizes(block_num_x * block_num_y, 0);
    for (int i = 0; i < n; i++)
    {
        int block_idx = block_idx_a[i];
        block_sizes[block_idx]++;
    }
    std::vector<int> block_offsets(block_num_x * block_num_y, 0);
    for (int i = 1; i < block_num_x * block_num_y; i++)
    {
        block_offsets[i] = block_offsets[i - 1] + block_sizes[i - 1];
    }

    torch::Tensor xyd = torch::zeros({n, 3}, torch::dtype(torch::kInt32));
    auto xyd_a = xyd.accessor<int, 2>();
    std::vector<int> cur_count(block_num_x * block_num_y, 0);
    for (int i = 0; i < n; i++)
    {
        int block_idx = block_idx_a[i];
        int offset = block_offsets[block_idx] + cur_count[block_idx];
        xyd_a[offset][0] = local_x_a[i];
        xyd_a[offset][1] = local_y_a[i];
        xyd_a[offset][2] = data_a[i];
        cur_count[block_idx]++;
    }
    return std::make_tuple(xyd, block_offsets);
}

torch::Tensor get_deltas_yx(int y_range, int x_range)
{
    int n = (2 * y_range + 1) * (2 * x_range + 1);
    torch::Tensor deltas = torch::zeros({n, 2}, torch::dtype(torch::kInt32));
    int i = 0;
    for (int y = -y_range; y < y_range + 1; y++)
    {
        for (int x = -x_range; x < x_range + 1; x++)
        {
            deltas[i][0] = y;
            deltas[i][1] = x;
            i++;
        }
    }
    return deltas;
}

template <typename T, torch::ScalarType dtype>
torch::Tensor
project_impl(
    torch::Tensor data,
    int W,
    int H,
    torch::Tensor xy_coords,
    const std::string &method,
    int kernel_size,
    T no_data_val)
{
    bool method_inv_dist = method == "inv_dist";
    bool method_nearest = method == "nearest";
    bool method_max = method == "max";
    if (!method_inv_dist && !method_nearest && !method_max)
        throw std::runtime_error("Invalid method: " + method);
    if constexpr (!std::is_same_v<T, float>)
    {
        if (method_inv_dist)
            throw std::runtime_error("inv_dist method only supported for float32 data");
    }
    data = data.permute({1, 2, 0}).contiguous();
    auto xy_coords_a = xy_coords.accessor<float, 3>(); // Y X 2
    auto data_a = data.accessor<T, 3>();               // Y X C
    int src_H = data.size(0);
    int src_W = data.size(1);
    int C = data.size(2);
    constexpr int block_size = 384;
    int pad = kernel_size / 2;  // Halo size for boundary handling
    int padded_block_size = block_size + 2 * pad;
    assert(xy_coords.size(0) == src_H);
    assert(xy_coords.size(1) == src_W);
    int NY = std::ceil(H / (float)block_size);
    int NX = std::ceil(W / (float)block_size);
    int num_blocks = NY * NX;
    int num_threads = at::get_num_threads();
    if (num_threads == 1)
    {
        int num_cores = std::thread::hardware_concurrency();
        num_cores = std::min(num_cores, 8);
        at::set_num_threads(num_cores);
        // std::cout << "Set LibTorch threads to " << num_cores << std::endl;
    }

    // Buffer stores: [dst_x, dst_y, val_0, val_1, ..., val_C-1] per element
    // Coordinates are float, values are T
    std::vector<int> block_sizes(num_blocks, 0);
    std::vector<int> block_offsets(num_blocks, 0);
    std::vector<float> coords_buffer; // x, y coords
    std::vector<T> values_buffer;     // channel values

    // === PASS 1: Parallel counting ===
    std::vector<std::atomic<int>> block_sizes_atomic(num_blocks);
    for (auto &a : block_sizes_atomic)
        a.store(0, std::memory_order_relaxed);

    at::parallel_for(0, src_H, 1, [&](int64_t y_start, int64_t y_end)
                     {
        for (int64_t y = y_start; y < y_end; y++)
        {
            for (int x = 0; x < src_W; x++)
            {
                float dst_x = xy_coords_a[y][x][0];
                float dst_y = xy_coords_a[y][x][1];
                int bx = std::floor(dst_x / block_size);
                int by = std::floor(dst_y / block_size);
                if (bx < 0 || bx >= NX || by < 0 || by >= NY)
                    continue;
                int block_idx = by * NX + bx;
                block_sizes_atomic[block_idx].fetch_add(1, std::memory_order_relaxed);
            }
        } });

    // Copy atomic counts to regular vector and compute offsets
    for (int i = 0; i < num_blocks; i++)
        block_sizes[i] = block_sizes_atomic[i].load(std::memory_order_relaxed);

    for (int b = 1; b < num_blocks; b++)
        block_offsets[b] = block_offsets[b - 1] + block_sizes[b - 1];

    int num_block_elements = num_blocks > 0 ? block_offsets[num_blocks - 1] + block_sizes[num_blocks - 1] : 0;
    coords_buffer.resize(num_block_elements * 2);
    values_buffer.resize(num_block_elements * C);

    // === PASS 2: Parallel buffer fill ===
    // Reset atomic counters for insertion indices
    for (auto &a : block_sizes_atomic)
        a.store(0, std::memory_order_relaxed);

    at::parallel_for(0, src_H, 1, [&](int64_t y_start, int64_t y_end)
                     {
        for (int64_t y = y_start; y < y_end; y++)
        {
            for (int x = 0; x < src_W; x++)
            {
                float dst_x = xy_coords_a[y][x][0];
                float dst_y = xy_coords_a[y][x][1];
                int bx = std::floor(dst_x / block_size);
                int by = std::floor(dst_y / block_size);
                if (bx < 0 || bx >= NX || by < 0 || by >= NY)
                    continue;
                int block_idx = by * NX + bx;
                int local_idx = block_sizes_atomic[block_idx].fetch_add(1, std::memory_order_relaxed);
                int elem_idx = block_offsets[block_idx] + local_idx;
                coords_buffer[elem_idx * 2 + 0] = dst_x;
                coords_buffer[elem_idx * 2 + 1] = dst_y;
                for (int c = 0; c < C; c++)
                    values_buffer[elem_idx * C + c] = data_a[y][x][c];
            }
        } });

    torch::Tensor deltas = get_deltas_yx(kernel_size / 2, kernel_size / 2);
    int num_deltas = deltas.size(0);
    auto deltas_a = deltas.accessor<int, 2>();

    // Precompute weights for each delta offset (avoids hypotf in hot loop)
    std::vector<float> delta_weights(num_deltas);
    for (int k = 0; k < num_deltas; k++)
    {
        float dy = static_cast<float>(deltas_a[k][0]);
        float dx = static_cast<float>(deltas_a[k][1]);
        float dist = std::hypotf(dy, dx);
        delta_weights[k] = dist == 0.0f ? 1e5f : 1.0f / dist;
    }

    torch::Tensor out = torch::zeros({H, W, C}, torch::dtype(dtype));
    auto out_a = out.accessor<T, 3>();
    // Padded block tensors for halo/ghost cell handling
    torch::Tensor block_values = torch::zeros({num_blocks, padded_block_size, padded_block_size, C}, torch::dtype(dtype));
    torch::Tensor block_weights = torch::zeros({num_blocks, padded_block_size, padded_block_size, C}, torch::dtype(torch::kFloat32));
    auto block_values_a = block_values.accessor<T, 4>();
    auto block_weights_a = block_weights.accessor<float, 4>();

    at::parallel_for(0, num_blocks, 8, [&](int64_t start, int64_t end)
                     {
        for (int i = start; i < end; i++)
        {
            int yo = (i / NX) * block_size;
            int xo = (i % NX) * block_size;
            for (int j = 0; j < block_sizes[i]; j++)
            {
                int elem_idx = block_offsets[i] + j;
                // Add pad offset so coordinates are in padded block space
                float x = coords_buffer[elem_idx * 2 + 0] - xo + pad;
                float y = coords_buffer[elem_idx * 2 + 1] - yo + pad;
                T *val_ptr = &values_buffer[elem_idx * C];
                
                for (int k = 0; k < num_deltas; k++)
                {
                    int y2 = std::round(y + deltas_a[k][0]);
                    int x2 = std::round(x + deltas_a[k][1]);
                    // Bounds check against padded block size (should rarely fail now)
                    if (y2 < 0 || y2 >= padded_block_size || x2 < 0 || x2 >= padded_block_size)
                        continue;
                    float weight = delta_weights[k];
                    
                    for (int c = 0; c < C; c++)
                    {
                        T val = val_ptr[c];
                        if (val == no_data_val)
                            continue;
                        if constexpr (std::is_same_v<T, float>)
                        {
                            if (method_inv_dist)
                            {
                                block_values_a[i][y2][x2][c] += weight * val;
                                block_weights_a[i][y2][x2][c] += weight;
                                continue;
                            }
                        }
                        if (method_max)
                        {
                            if (val > block_values_a[i][y2][x2][c])
                            {
                                block_values_a[i][y2][x2][c] = val;
                                block_weights_a[i][y2][x2][c] = weight;
                            }
                        }
                        else // nearest
                        {
                            if (weight > block_weights_a[i][y2][x2][c])
                            {
                                block_values_a[i][y2][x2][c] = val;
                                block_weights_a[i][y2][x2][c] = weight;
                            }
                        }
                    }
                }
            }
            // Copy inner (non-halo) region to output
            for (int dy = 0; dy < block_size; dy++)
            {
                int y = yo + dy;
                if (y >= H || y < 0)
                    break;
                // Read from padded position (pad + dy, pad + dx)
                int py = pad + dy;
                for (int dx = 0; dx < block_size; dx++)
                {
                    int x = xo + dx;
                    if (x >= W || x < 0)
                        break;
                    int px = pad + dx;
                    for (int c = 0; c < C; c++)
                    {
                        if (block_weights_a[i][py][px][c] == 0)
                        {
                            out_a[y][x][c] = no_data_val;
                            continue;
                        }
                        if constexpr (std::is_same_v<T, float>)
                        {
                            if (method_inv_dist)
                            {
                                out_a[y][x][c] += block_values_a[i][py][px][c] / block_weights_a[i][py][px][c];
                                continue;
                            }
                        }
                        out_a[y][x][c] = block_values_a[i][py][px][c];
                    }
                }
            }
        } });

    out = out.permute({2, 0, 1}).contiguous();

    return out;
}

// Dispatcher - routes to correct template instantiation based on input dtype
torch::Tensor
project(
    torch::Tensor data,
    int W,
    int H,
    torch::Tensor xy_coords,
    const std::string &method,
    int kernel_size,
    float no_data_val)
{
    if (data.scalar_type() == torch::kFloat32)
    {
        return project_impl<float, torch::kFloat32>(
            data, W, H, xy_coords, method, kernel_size, no_data_val);
    }
    else if (data.scalar_type() == torch::kUInt8)
    {
        return project_impl<uint8_t, torch::kUInt8>(
            data, W, H, xy_coords, method, kernel_size, static_cast<uint8_t>(no_data_val));
    }
    else
    {
        throw std::runtime_error("project: unsupported dtype, expected float32 or uint8");
    }
}

// ============================================================================
// fill_holes: Fill no_data pixels in pre-rasterized data using block-based
// processing for cache locality. Uses padded blocks for proper boundary handling.
// ============================================================================

template <typename T, torch::ScalarType dtype>
torch::Tensor
fill_holes_impl(
    torch::Tensor data,
    int kernel_size,
    T no_data_val,
    const std::string &method)
{
    bool method_inv_dist = method == "inv_dist";
    bool method_nearest = method == "nearest";
    if (!method_inv_dist && !method_nearest)
        throw std::runtime_error("fill_holes: Invalid method: " + method + " (expected nearest or inv_dist)");
    if constexpr (!std::is_same_v<T, float>)
    {
        if (method_inv_dist)
            throw std::runtime_error("fill_holes: inv_dist method only supported for float32 data");
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Input is [C, H, W], permute to [H, W, C] for cache-friendly access
    data = data.permute({1, 2, 0}).contiguous();
    auto data_a = data.accessor<T, 3>(); // H W C
    int H = data.size(0);
    int W = data.size(1);
    int C = data.size(2);

    constexpr int block_size = 384;
    int pad = kernel_size / 2;
    int padded_block_size = block_size + 2 * pad;

    int NY = std::ceil(H / (float)block_size);
    int NX = std::ceil(W / (float)block_size);
    int num_blocks = NY * NX;

    // Ensure multi-threading
    int num_threads = at::get_num_threads();
    if (num_threads == 1)
    {
        int num_cores = std::thread::hardware_concurrency();
        num_cores = std::min(num_cores, 8);
        at::set_num_threads(num_cores);
    }

    // Precompute kernel deltas and weights
    torch::Tensor deltas = get_deltas_yx(kernel_size / 2, kernel_size / 2);
    int num_deltas = deltas.size(0);
    auto deltas_a = deltas.accessor<int, 2>();

    std::vector<float> delta_weights(num_deltas);
    for (int k = 0; k < num_deltas; k++)
    {
        float dy = static_cast<float>(deltas_a[k][0]);
        float dx = static_cast<float>(deltas_a[k][1]);
        float dist = std::hypotf(dy, dx);
        delta_weights[k] = dist == 0.0f ? 1e5f : 1.0f / dist;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Output tensor and padded block tensors
    torch::Tensor out = torch::zeros({H, W, C}, torch::dtype(dtype));
    auto out_a = out.accessor<T, 3>();

    // Padded block tensor for each block (holds input data + halo)
    torch::Tensor block_input = torch::full({num_blocks, padded_block_size, padded_block_size, C}, 
                                             static_cast<float>(no_data_val), torch::dtype(dtype));
    auto block_input_a = block_input.accessor<T, 4>();

    auto t2 = std::chrono::high_resolution_clock::now();

    // Copy input data into padded blocks (with halo regions)
    at::parallel_for(0, num_blocks, 1, [&](int64_t start, int64_t end)
                     {
        for (int block_idx = start; block_idx < end; block_idx++)
        {
            int by = block_idx / NX;
            int bx = block_idx % NX;
            int yo = by * block_size;  // Block origin in global coords
            int xo = bx * block_size;

            // Copy data into padded block (including halo from adjacent regions)
            for (int py = 0; py < padded_block_size; py++)
            {
                int gy = yo - pad + py;  // Global y coordinate
                if (gy < 0 || gy >= H)
                    continue;
                for (int px = 0; px < padded_block_size; px++)
                {
                    int gx = xo - pad + px;  // Global x coordinate
                    if (gx < 0 || gx >= W)
                        continue;
                    for (int c = 0; c < C; c++)
                    {
                        block_input_a[block_idx][py][px][c] = data_a[gy][gx][c];
                    }
                }
            }
        } });

    auto t3 = std::chrono::high_resolution_clock::now();

    // Process blocks in parallel - fill holes using padded block data
    at::parallel_for(0, num_blocks, 1, [&](int64_t start, int64_t end)
                     {
        for (int block_idx = start; block_idx < end; block_idx++)
        {
            int by = block_idx / NX;
            int bx = block_idx % NX;
            int yo = by * block_size;
            int xo = bx * block_size;

            // Process inner region (non-halo) of this block
            for (int dy = 0; dy < block_size; dy++)
            {
                int gy = yo + dy;
                if (gy >= H)
                    break;
                int py = pad + dy;  // Position in padded block

                for (int dx = 0; dx < block_size; dx++)
                {
                    int gx = xo + dx;
                    if (gx >= W)
                        break;
                    int px = pad + dx;

                    for (int c = 0; c < C; c++)
                    {
                        T val = block_input_a[block_idx][py][px][c];

                        // If pixel has data, copy directly
                        if (val != no_data_val)
                        {
                            out_a[gy][gx][c] = val;
                            continue;
                        }

                        // Pixel is no_data - search kernel neighborhood in padded block
                        if (method_nearest)
                        {
                            float best_weight = -1.0f;
                            T best_val = no_data_val;

                            for (int k = 0; k < num_deltas; k++)
                            {
                                int ny = py + deltas_a[k][0];
                                int nx = px + deltas_a[k][1];
                                // Bounds check within padded block
                                if (ny < 0 || ny >= padded_block_size || nx < 0 || nx >= padded_block_size)
                                    continue;

                                T neighbor_val = block_input_a[block_idx][ny][nx][c];
                                if (neighbor_val == no_data_val)
                                    continue;

                                float weight = delta_weights[k];
                                if (weight > best_weight)
                                {
                                    best_weight = weight;
                                    best_val = neighbor_val;
                                }
                            }
                            out_a[gy][gx][c] = best_val;
                        }
                        else // inv_dist
                        {
                            float sum_weighted = 0.0f;
                            float sum_weights = 0.0f;

                            for (int k = 0; k < num_deltas; k++)
                            {
                                int ny = py + deltas_a[k][0];
                                int nx = px + deltas_a[k][1];
                                if (ny < 0 || ny >= padded_block_size || nx < 0 || nx >= padded_block_size)
                                    continue;

                                T neighbor_val = block_input_a[block_idx][ny][nx][c];
                                if (neighbor_val == no_data_val)
                                    continue;

                                float weight = delta_weights[k];
                                if constexpr (std::is_same_v<T, float>)
                                {
                                    sum_weighted += weight * neighbor_val;
                                    sum_weights += weight;
                                }
                            }

                            if constexpr (std::is_same_v<T, float>)
                            {
                                if (sum_weights > 0)
                                    out_a[gy][gx][c] = sum_weighted / sum_weights;
                                else
                                    out_a[gy][gx][c] = no_data_val;
                            }
                        }
                    }
                }
            }
        } });

    auto t4 = std::chrono::high_resolution_clock::now();

    // Permute back to [C, H, W]
    out = out.permute({2, 0, 1}).contiguous();

    auto t5 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::cout << "  [fill_holes] setup: " << ms(t0, t1) << "ms, "
              << "alloc: " << ms(t1, t2) << "ms, "
              << "copy_to_blocks: " << ms(t2, t3) << "ms, "
              << "kernel: " << ms(t3, t4) << "ms, "
              << "permute: " << ms(t4, t5) << "ms" << std::endl;

    return out;
}

// Dispatcher for fill_holes
torch::Tensor
fill_holes(
    torch::Tensor data,
    int kernel_size,
    float no_data_val,
    const std::string &method)
{
    if (data.scalar_type() == torch::kFloat32)
    {
        return fill_holes_impl<float, torch::kFloat32>(
            data, kernel_size, no_data_val, method);
    }
    else if (data.scalar_type() == torch::kUInt8)
    {
        return fill_holes_impl<uint8_t, torch::kUInt8>(
            data, kernel_size, static_cast<uint8_t>(no_data_val), method);
    }
    else
    {
        throw std::runtime_error("fill_holes: unsupported dtype, expected float32 or uint8");
    }
}

#if defined(__INTELLISENSE__)
// Skip extension code so IntelliSense won't barf
#else
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("search", search);
    m.def("group_by_fire", group_by_fire);
    m.def("group_by_time", group_by_time);
    m.def("group_by_blocks", group_by_blocks);
    m.def("project", project);
    m.def("fill_holes", fill_holes);
}
#endif
