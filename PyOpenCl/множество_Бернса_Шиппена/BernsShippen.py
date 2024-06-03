import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

# Параметры
width = 800
height = 800
max_iter = 256

# OpenCL программа для генерации множества Бернса-Шиппена
kernel_code = """
__kernel void burns_ship(
    const int width,
    const int height,
    const int max_iter,
    const float x_min,
    const float x_max,
    const float y_min,
    const float y_max,
    __global int* output)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);

    if (gid_x >= width || gid_y >= height)
        return;

    float x0 = x_min + gid_x * (x_max - x_min) / (float)(width - 1);
    float y0 = y_min + gid_y * (y_max - y_min) / (float)(height - 1);
    float x = x0;
    float y = y0;

    int iteration = 0;
    while (x*x + y*y <= 4.0f && iteration < max_iter) {
        float xtemp = x*x - y*y + x0;
        y = fabs(2*x*y) + y0;
        x = fabs(xtemp);
        iteration++;
    }

    int index = gid_y * width + gid_x;
    output[index] = iteration;
}
"""

# Настройка OpenCL
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Параметры области для визуализации фрактала
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0

# Создание буфера для хранения результатов
output = np.zeros((height, width)).astype(np.int32)
output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

# Компиляция и выполнение программы
program = cl.Program(context, kernel_code).build()
burns_ship_kernel = program.burns_ship
burns_ship_kernel.set_args(np.int32(width), np.int32(height), np.int32(max_iter),
                           np.float32(x_min), np.float32(x_max), np.float32(y_min), np.float32(y_max),
                           output_buf)

# Определение размера рабочей группы
global_size = (width, height)

# Запуск программы на GPU
cl.enqueue_nd_range_kernel(queue, burns_ship_kernel, global_size, None).wait()

# Копирование результатов обратно на хост
cl.enqueue_copy(queue, output, output_buf).wait()

# Визуализация результатов
plt.imshow(output, extent=(x_min, x_max, y_min, y_max), cmap='inferno')
plt.colorbar()
plt.title("Множество Бернса-Шиппена")
plt.show()