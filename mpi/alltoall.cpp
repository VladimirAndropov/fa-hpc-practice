#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Инициализация генератора случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd() + rank); // Добавляем rank для разных seed на разных процессах
    std::uniform_int_distribution<> dis(0, 99);

    // Создание массива случайных чисел на каждом процессе (size x 5)
    std::vector<std::vector<int>> data(size, std::vector<int>(5));
    for (auto& row : data) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }

    // Вывод исходных данных
    std::cout << "Process " << rank << ": Initial data:\n";
    for (const auto& row : data) {
        for (int val : row) std::cout << val << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Подготовка данных для alltoall
    std::vector<int> sendbuf(size * 5);
    std::vector<int> recvbuf(size * 5);

    // Копируем данные в sendbuf (построчно)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 5; ++j) {
            sendbuf[i * 5 + j] = data[i][j];
        }
    }

    // Распределение данных между всеми процессами
    MPI_Alltoall(sendbuf.data(), 5, MPI_INT, recvbuf.data(), 5, MPI_INT, MPI_COMM_WORLD);

    // Преобразование полученных данных в матрицу (size x 5)
    std::vector<std::vector<int>> scattered_data(size, std::vector<int>(5));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 5; ++j) {
            scattered_data[i][j] = recvbuf[i * 5 + j];
        }
    }

    // Вычисление средних значений
    std::vector<double> mean_values(size);
    for (int i = 0; i < size; ++i) {
        double sum = std::accumulate(scattered_data[i].begin(), scattered_data[i].end(), 0.0);
        mean_values[i] = sum / scattered_data[i].size();
    }

    // Вывод средних значений
    std::cout << "Process " << rank << ": Mean values:\n";
    for (double val : mean_values) std::cout << val << " ";
    std::cout << "\n\n";

    // Сортировка полученных данных
    for (auto& row : scattered_data) {
        std::sort(row.begin(), row.end());
    }

    // Подготовка к отправке обратно
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 5; ++j) {
            sendbuf[i * 5 + j] = scattered_data[i][j];
        }
    }

    // Отправка результатов обратно всем процессам
    MPI_Alltoall(sendbuf.data(), 5, MPI_INT, recvbuf.data(), 5, MPI_INT, MPI_COMM_WORLD);

    // Преобразование полученных данных в матрицу
    std::vector<std::vector<int>> gathered_data(size, std::vector<int>(5));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 5; ++j) {
            gathered_data[i][j] = recvbuf[i * 5 + j];
        }
    }

    // Вывод полученных данных
    std::cout << "Process " << rank << ": Gathered data:\n";
    for (const auto& row : gathered_data) {
        for (int val : row) std::cout << val << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}