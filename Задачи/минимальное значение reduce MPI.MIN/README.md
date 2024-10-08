Напишите программу, которая использует функцию
reduce, чтобы найти минимальное значение среди всех значений, сгенериро-
ванных всеми процессами. Каждый процесс должен сгенерировать случай-
ное число. Затем используйте reduce с операцией MPI.MIN, чтобы найти ми-
нимальное значение. В конце процесс с рангом 0 должен вывести
минимальное значение;

Для реализации программы, использующей функцию `reduce` с операцией `MPI.MIN` для поиска минимального значения среди всех значений, сгенерированных всеми процессами, выполните следующие шаги:

```python
from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Генерация случайного числа на каждом процессе
    random_number = np.random.randint(0, 100)
    print(f"Процесс {rank}: Сгенерированное случайное число - {random_number}")

    # Поиск минимального значения среди всех процессов с помощью reduce
    min_value = comm.reduce(random_number, op=MPI.MIN, root=0)

    # Вывод минимального значения на процессе с рангом 0
    if rank == 0:
        print(f"Минимальное значение: {min_value}")

if __name__ == "__main__":
    main()
```

### Описание программы:
1. Инициализация MPI и получение ранга процесса и количества процессов.
2. Генерация случайного числа на каждом процессе.
3. Использование функции `reduce` с операцией `MPI.MIN` для поиска минимального значения среди всех процессов. Каждый процесс передает свое случайное число, а процесс с рангом 0 получает минимальное значение.
4. Вывод минимального значения на процессе с рангом 0.

### Запуск программы
Для запуска программы с использованием нескольких процессов MPI, выполните следующую команду в командной строке:

```sh
mpiexec -n 4 python your_script.py
```

Здесь `your_script.py` — это имя вашего файла Python, а `-n 4` указывает на запуск программы с четырьмя процессами. Вы можете изменить количество процессов по вашему усмотрению.

### Пример вывода
Пример вывода будет зависеть от сгенерированных случайных чисел на каждом процессе. Процесс с рангом 0 выведет минимальное из этих чисел.
__________

Вариант 2: Напишите программу, которая использует функцию
reduce, чтобы найти максимальное значение среди всех значений, сгенери-
рованных всеми процессами. Каждый процесс должен сгенерировать слу-
чайное число. Затем используйте reduce с операцией MPI.MAX, чтобы
найти максимальное значение. В конце процесс с рангом 0 должен вывести
максимальное значение.

Вот пример программы, которая использует функцию `reduce` с операцией `MPI.MAX` для поиска максимального значения среди всех значений, сгенерированных всеми процессами:

```python
from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Генерация случайного числа на каждом процессе
    random_number = np.random.randint(0, 100)
    print(f"Процесс {rank}: Сгенерированное случайное число - {random_number}")

    # Поиск максимального значения среди всех процессов с помощью reduce
    max_value = comm.reduce(random_number, op=MPI.MAX, root=0)

    # Вывод максимального значения на процессе с рангом 0
    if rank == 0:
        print(f"Максимальное значение: {max_value}")

if __name__ == "__main__":
    main()
```

### Описание программы:
1. Инициализация MPI и получение ранга процесса и количества процессов.
2. Генерация случайного числа на каждом процессе.
3. Использование функции `reduce` с операцией `MPI.MAX` для поиска максимального значения среди всех процессов. Каждый процесс передает свое случайное число, а процесс с рангом 0 получает максимальное значение.
4. Вывод максимального значения на процессе с рангом 0.

### Запуск программы
Для запуска программы с использованием нескольких процессов MPI, выполните следующую команду в командной строке:

```sh
mpiexec -n 4 python your_script.py
```

Здесь `your_script.py` — это имя вашего файла Python, а `-n 4` указывает на запуск программы с четырьмя процессами. Вы можете изменить количество процессов по вашему усмотрению.

### Пример вывода
Пример вывода будет зависеть от сгенерированных случайных чисел на каждом процессе. Процесс с рангом 0 выведет максимальное из этих чисел.