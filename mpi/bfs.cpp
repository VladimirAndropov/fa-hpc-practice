#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <utility>

using namespace std;

// Pair structure for 2D points
typedef pair<int, int> Point;

// BFS implementation
set<Point> bfs(const vector<vector<int>>& grid, Point start_point, int depth) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    queue<pair<Point, int>> q;
    set<Point> visited;

    q.push({ start_point, 0 });

    while (!q.empty()) {
        auto current = q.front();
        q.pop();
        Point current_point = current.first;
        int current_depth = current.second;

        if (visited.count(current_point)) {
            continue;
        }

        visited.insert(current_point);
        int x = current_point.first;
        int y = current_point.second;

        if (current_depth >= depth) {
            break;
        }

        // Generate neighbors with periodic boundary conditions
        vector<Point> neighbors = {
            {(x + 1) % (int)grid.size(), y},  // right
            {(x - 1 + (int)grid.size()) % (int)grid.size(), y},  // left
            {x, (y + 1) % (int)grid[0].size()},  // down
            {x, (y - 1 + (int)grid[0].size()) % (int)grid[0].size()}  // up
        };

        for (auto& neighbor : neighbors) {
            q.push({ neighbor, current_depth + 1 });
        }
    }

    return visited;
}

void print_grid(const vector<vector<int>>& grid, int rank) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 0; i < size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i) {
            cout << "Process " << rank << ":" << endl;
            for (const auto& row : grid) {
                for (int val : row) {
                    cout << val << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Grid dimensions
    const int rows = 5;
    const int cols = 5;

    // Starting point and depth
    Point start_point = { 0, 0 };
    int depth = 3;

    // Create local grid
    vector<vector<int>> local_grid(rows, vector<int>(cols, 0));

    // Perform BFS
    set<Point> visited_points = bfs(local_grid, start_point, depth);

    // Mark visited points in the grid
    for (const auto& point : visited_points) {
        if (point.first >= 0 && point.first < rows &&
            point.second >= 0 && point.second < cols) {
            local_grid[point.first][point.second] = 1;
        }
    }

    // Print grid state
    print_grid(local_grid, rank);

    MPI_Finalize();
    return 0;
}