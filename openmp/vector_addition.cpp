#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

// Функция для заполнения вектора случайными вещественными значениями
void fillVector(std::vector<double> &vec) {
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main() {
    // Размер векторов
    const size_t N = 50000000;
    double start; 
    double end; 

    // Выделение памяти для векторов
    std::vector<double> vecA(N);
    std::vector<double> vecB(N);
    std::vector<double> vecC(N);

    // Заполнение векторов случайными значениями
    srand(static_cast<unsigned>(time(0)));
    fillVector(vecA);
    fillVector(vecB);

    // Начало измерения времени
    start = omp_get_wtime(); 

    // Сложение векторов с использованием OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        vecC[i] = vecA[i] + vecB[i];
    }

    // Конец измерения времени
    end = omp_get_wtime(); 

    // Вывод результатов
    printf("Work took %f seconds\n", end - start);
    std::cout << "Size of resulting vector: " << vecC.size() << std::endl;
    std::cout << "Example element from resulting vector: " << vecC[N / 2] << std::endl;

    return 0;
}