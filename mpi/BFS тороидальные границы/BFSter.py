from mpi4py import MPI
import numpy as np

def propagate_wave(grid, start_point, num_steps):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Размер сетки
    grid_size = grid.shape
    
    # Копируем сетку для обновления значений
    new_grid = np.copy(grid)
    
    # Выполняем распространение волны заданное количество шагов
    for step in range(num_steps):
        # Обновляем значения в соседних ячейках с учетом периодических граничных условий
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                for neighbor in neighbors:
                    x, y = neighbor
                    # Учитываем периодические граничные условия
                    x = x % grid_size[0]
                    y = y % grid_size[1]
                    new_grid[x, y] = max(new_grid[x, y], grid[i, j])
        
        # Обновляем текущее состояние сетки
        grid[:] = new_grid[:]
    
    return grid

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
    # Начальная точка распространения волны
    start_point = (0, 0)
    # Количество шагов распространения волны
    num_steps = 3
    
    # Создаем сетку для текущего процесса
    local_grid = np.zeros(grid_size, dtype=int)
    
    # Задаем начальную точку для распространения волны
    local_grid[start_point] = 1
    
    # Выполняем распространение волны на сетке
    propagated_grid = propagate_wave(local_grid, start_point, num_steps)
    
    # Выводим состояние сетки после распространения волны
    print_grid(propagated_grid, rank)

if __name__ == "__main__":
    main()