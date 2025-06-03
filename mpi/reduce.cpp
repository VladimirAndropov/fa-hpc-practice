#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <locale>

int main(int argc, char** argv) {

    setlocale(LC_ALL,"ru_RU.UTF-8");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Инициализация генератора случайных чисел с учетом ранга
    std::srand(std::time(nullptr) + rank);

    // Генерация случайного числа от 0 до 99
    int random_number = std::rand() % 100;
    std::cout << "Процесс " << rank << ": Сгенерированное случайное число - "
        << random_number << std::endl;

    // Поиск минимального значения среди всех процессов
    int min_value;
    MPI_Reduce(&random_number, &min_value, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    // Вывод минимального значения на процессе с рангом 0
    if (rank == 0) {
        std::cout << "Минимальное значение: " << min_value << std::endl;
    }

    MPI_Finalize();
    return 0;
}