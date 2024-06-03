Реализуйте на MPI разбиение процессов приложения на
две группы, в одной из которых осуществляется обмен по кольцевой топо-
логии, а в другой – коммуникации по схеме «мастер – рабочие» (с исполь-
зованием любых изученных функций).

Вот пример программы на MPI, которая разделяет процессы на две группы и осуществляет коммуникации в каждой группе по разным схемам: кольцевая топология и схема "мастер-рабочие":

```python
from mpi4py import MPI

def ring_communicate(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    left_neighbor = (rank - 1) % size
    right_neighbor = (rank + 1) % size

    sendbuf = rank
    recvbuf = None

    for _ in range(size):
        comm.Sendrecv(sendbuf, dest=right_neighbor, recvbuf=recvbuf, source=left_neighbor)
        print(f"Процесс {rank}: Получено от процесса {left_neighbor}, отправлено процессу {right_neighbor}")
        sendbuf = recvbuf

def master_worker_communicate(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Мастер процесс
        data_to_send = list(range(1, size))
        for worker_rank in range(1, size):
            comm.send(data_to_send, dest=worker_rank)
        print(f"Мастер процесс {rank} отправил данные всем рабочим")
    else:
        # Рабочий процесс
        data_received = comm.recv(source=0)
        print(f"Рабочий процесс {rank} получил данные от мастер процесса: {data_received}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Разбиение процессов на две группы
    group1 = comm.Get_group().Incl(list(range(0, size, 2)))
    group2 = comm.Get_group().Incl(list(range(1, size, 2)))

    # Создание коммуникаторов для каждой группы
    comm_group1 = comm.Create(group1)
    comm_group2 = comm.Create(group2)

    if comm_group1 != MPI.COMM_NULL:
        print(f"Группа 1: Процессы {comm_group1.rank} из {comm_group1.size}")
        ring_communicate(comm_group1)
    if comm_group2 != MPI.COMM_NULL:
        print(f"Группа 2: Процессы {comm_group2.rank} из {comm_group2.size}")
        master_worker_communicate(comm_group2)

    comm_group1.Free()
    comm_group2.Free()

if __name__ == "__main__":
    main()
```

### Описание программы:
1. Импорт библиотеки MPI.
2. Определение функции `ring_communicate`, которая реализует обмен данными по кольцевой топологии между процессами в группе.
3. Определение функции `master_worker_communicate`, которая реализует коммуникацию схемы "мастер-рабочие" между процессами в группе.
4. Определение основной функции `main`, которая создает две группы процессов на основе рангов процессов, создает коммуникаторы для каждой группы и вызывает соответствующие функции для каждой группы.

### Запуск программы:
Для запуска программы на нескольких процессах MPI, выполните следующую команду в командной строке:
```sh
mpiexec -n <num_processes> python your_script.py
```
где `<num_processes>` - количество процессов, а `your_script.py` - имя вашего файла Python.

### Пример вывода:
Пример вывода будет зависеть от количества процессов и их рангов. Каждая группа процессов будет выводить информацию о своей коммуникации.