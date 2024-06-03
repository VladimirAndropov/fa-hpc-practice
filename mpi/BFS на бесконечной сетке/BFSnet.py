from mpi4py import MPI
import numpy as np
from collections import deque

def bfs(grid, start_point, depth):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Очередь для обхода в ширину
    queue = deque([(start_point, 0)])
    visited = set()

    while queue:
        current_point, current_depth = queue.popleft()
        
        # Проверяем, что точка не была посещена
        if current_point in visited:
            continue
        
        # Помечаем текущую точку как посещенную
        visited.add(current_point)
        x, y = current_point
        
        # Проверяем глубину обхода
        if current_depth >= depth:
            break
        
        # Добавляем соседей текущей точки в очередь
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for neighbor in neighbors:
            # Учитываем периодические граничные условия
            neighbor = ((neighbor[0] % grid.shape[0]), (neighbor[1] % grid.shape[1]))
            queue.append((neighbor, current_depth + 1))

    return visited

def print_grid(grid, rank):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    
    for i in range(size):
        if rank == i:
            print(f"Процесс {rank}:")
            print(grid)
        comm.Barrier()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Размер сетки
    grid_size = (5, 5)
    # Начальная точка обхода
    start_point = (0, 0)
    # Глубина обхода
    depth = 3
    
    # Создаем сетку для текущего процесса
    local_grid = np.zeros(grid_size, dtype=int)
    
    # Выполняем обход в ширину на сетке
    visited_points = bfs(local_grid, start_point, depth)
    
    # Выводим состояние сетки после обхода
    print_grid(local_grid, rank)

if __name__ == "__main__":
    main()