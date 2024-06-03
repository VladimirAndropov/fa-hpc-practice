import numpy as np
import pyopencl as cl

# Параметры
num_points = 10**6  # Количество точек

# OpenCL программа для метода Монте-Карло
kernel_code = """
__kernel void monte_carlo_pi(
    const int num_points,
    __global const float* random_x,
    __global const float* random_y,
    __global int* count)
{
    int i = get_global_id(0);
    if (i >= num_points)
        return;

    float x = random_x[i];
    float y = random_y[i];
    if (x * x + y * y <= 1.0f)
        atomic_add(count, 1);
}
"""

# Настройка OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Генерация случайных точек
random_x = np.random.rand(num_points).astype(np.float32)
random_y = np.random.rand(num_points).astype(np.float32)
count = np.zeros(1).astype(np.int32)

# Создание буферов на устройстве
mf = cl.mem_flags
random_x_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=random_x)
random_y_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=random_y)
count_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=count)

# Компиляция и выполнение программы
program = cl.Program(context, kernel_code).build()
monte_carlo_kernel = program.monte_carlo_pi
monte_carlo_kernel.set_args(np.int32(num_points), random_x_buf, random_y_buf, count_buf)

# Запуск программы на GPU
global_size = (num_points,)
cl.enqueue_nd_range_kernel(queue, monte_carlo_kernel, global_size, None).wait()

# Копирование результатов обратно на хост
cl.enqueue_copy(queue, count, count_buf).wait()

# Вычисление значения Пи
pi_estimate = 4.0 * count[0] / num_points

# Вывод результатов
print(f"Приближенное значение числа Пи: {pi_estimate}")