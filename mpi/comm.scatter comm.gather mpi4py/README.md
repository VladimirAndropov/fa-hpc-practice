Напишите программу на Python, используя модуль `mpi4py`, которая
создает список чисел на процессе с рангом 0, распространяет эти числа на
все процессы с помощью функции `comm.scatter`, затем каждый процесс
увеличивает свое число на 1 и собирает обновленные числа обратно на про-
цессе с рангом 0 с помощью функции `comm.gather`. На процессе с рангом
0 должен быть выведен исходный список чисел и обновленный список чи-
сел.

Для реализации программы, использующей `mpi4py` и функции `comm.scatter` и `comm.gather` для распределения и сбора данных, выполните следующие шаги. Программа будет создавать список чисел на процессе с рангом 0, распространять эти числа на все процессы, каждый процесс увеличивает свое число на 1, и затем собирать обновленные числа обратно на процессе с рангом 0.

### Программа

```python
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Создание списка чисел на процессе с рангом 0
    if rank == 0:
        data = list(range(size))  # Создаем список чисел от 0 до size-1
        print(f"Исходный список чисел: {data}")
    else:
        data = None

    # Распределение чисел на все процессы
    number = comm.scatter(data, root=0)

    # Увеличение числа на 1 в каждом процессе
    number += 1

    # Сбор обновленных чисел на процессе с рангом 0
    new_data = comm.gather(number, root=0)

    # Вывод исходного и обновленного списка чисел на процессе с рангом 0
    if rank == 0:
        print(f"Обновленный список чисел: {new_data}")

if __name__ == "__main__":
    main()
```

### Описание программы:
1. **Инициализация MPI**: импортируется модуль `MPI` из `mpi4py` и создается объект `comm`, представляющий коммуникатор MPI.
2. **Получение ранга процесса и количества процессов**: каждый процесс получает свой ранг с помощью метода `Get_rank()` и количество процессов с помощью метода `Get_size()`.
3. **Создание списка чисел на процессе с рангом 0**: если ранг процесса равен 0, создается список чисел от 0 до `size - 1`.
4. **Распределение чисел**: список `data` передается от процесса с рангом 0 всем остальным процессам с помощью функции `comm.scatter`. Каждый процесс получает одно число из списка.
5. **Обработка данных**: каждый процесс увеличивает полученное число на 1.
6. **Сбор обновленных чисел**: обновленные числа собираются обратно на процессе с рангом 0 с помощью функции `comm.gather`.
7. **Вывод данных**: на процессе с рангом 0 выводится исходный и обновленный список чисел.

### Запуск программы

Для запуска программы с использованием нескольких процессов MPI, выполните следующую команду в командной строке:

```sh
mpiexec -n 4 python your_script.py
```

Здесь `your_script.py` — это имя вашего файла Python, а `-n 4` указывает на запуск программы с четырьмя процессами. Вы можете изменить количество процессов по вашему усмотрению.

### Пример вывода

Когда вы запускаете программу, на процессе с рангом 0 будет выведен исходный и обновленный список чисел:

```
Исходный список чисел: [0, 1, 2, 3]
Обновленный список чисел: [1, 2, 3, 4]
```

Это подтверждает, что каждый процесс успешно получил одно число из списка, увеличил его на 1 и отправил обратно на процесс с рангом 0, где был собран и выведен обновленный список чисел.