import numpy as np
import pyopencl as cl

# Параметры
t0 = 0.0  # начальное время
y0 = 1.0  # начальное значение y
h = 0.01  # шаг интегрирования
num_steps = 1000  # количество шагов

# OpenCL программа для метода Эйлера
kernel_code = """
__kernel void euler(
    const float t0,
    const float h,
    const int num_steps,
    __global float* y)
{
    int i = get_global_id(0);
    if (i >= num_steps)
        return;

    float t = t0 + i * h;
    float yi = y[i];
    float f = -2 * yi;  // Определяем f(t, y) = -2y
    y[i + 1] = yi + h * f;
}
"""

# Настройка OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Буфер для хранения результатов
y = np.zeros(num_steps + 1).astype(np.float32)
y[0] = y0

# Создание буфера на устройстве
mf = cl.mem_flags
y_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)

# Компиляция и выполнение программы
program = cl.Program(context, kernel_code).build()
euler_kernel = program.euler
euler_kernel.set_args(np.float32(t0), np.float32(h), np.int32(num_steps), y_buf)

# Запуск программы на GPU
global_size = (num_steps,)
cl.enqueue_nd_range_kernel(queue, euler_kernel, global_size, None).wait()

# Копирование результатов обратно на хост
cl.enqueue_copy(queue, y, y_buf).wait()

# Вывод результатов
print("Результаты интегрирования методом Эйлера:")
for i in range(10):
    print(f"y({t0 + i * h}) = {y[i]}")