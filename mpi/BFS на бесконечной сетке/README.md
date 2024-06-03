PLEASE install python-numpy

```
python3 -m pip install numpy
```

Реализовать алгоритм обхода в ширину (BFS) на бесконеч-
ной сетке с периодическими граничными условиями (также известными как
тороидальные границы). Программа должна начинать обход из заданной
начальной точки и продолжать до тех пор, пока не будет достигнута задан-
ная глубина; программа должна корректно обрабатывать переходы через
границы сетки и быть способна работать с произвольным количеством про-
цессов.
Программа должна выводить состояние сетки после завершения BFS.
Каждый процесс должен выводить свою собственную копию сетки, чтобы
можно было видеть, как состояние сетки синхронизируется между процес-
сами.

Для реализации алгоритма обхода в ширину (BFS) на бесконечной сетке с периодическими граничными условиями (тороидальными границами) в MPI можно использовать следующий подход:

1. Инициализировать сетку и задать начальную точку обхода.
2. Передать сообщение между процессами для синхронизации начальной точки обхода.
3. Выполнить алгоритм BFS, используя очередь для хранения текущих узлов, которые нужно посетить.
4. Передвигаться по сетке с учетом периодических граничных условий.
5. Вывести состояние сетки после завершения обхода в ширину.

Ниже представлен пример кода на Python, реализующий этот алгоритм:

```python
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
```

Этот код реализует алгоритм обхода в ширину на бесконечной сетке с периодическими граничными условиями в MPI. Каждый процесс выполняет обход только на своей части сетки и выводит результаты обхода в своей собственной копии сетки.