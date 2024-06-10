const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (isMainThread) {
  // Входные данные
  const processes = [100, 200, 300, 400]; // Время выполнения процессов (число инструкций)
  const processors = [2000, 300]; // Скорость процессоров (инструкций в секунду)
  const D = 5; // Максимальное время выполнения в секундах

  // Функция для распределения процессов
  function distributeProcesses(processes, processors, D) {
    // Сортировка процессов и процессоров
    processes.sort((a, b) => b - a);
    processors.sort((a, b) => b - a);

    // Распределение процессов
    let assignments = new Array(processors.length).fill(0);
    for (let process of processes) {
      let minIndex = assignments.indexOf(Math.min(...assignments));
      assignments[minIndex] += process / processors[minIndex];
    }

    // Проверка, не превышает ли общее время выполнения D секунд
    return Math.max(...assignments) <= D;
  }

  // Проверка возможности распределения процессов
  const canDistribute = distributeProcesses(processes, processors, D);
  console.log(`Можно ли распределить процессы за ${D} секунд: ${canDistribute}`);

  // Создание worker_threads для параллельного выполнения
  if (canDistribute) {
    for (let i = 0; i < processors.length; i++) {
      new Worker(__filename, { workerData: { process: processes[i], speed: processors[i] } });
    }
  }
} else {
  // Код для worker_threads
  const { process, speed } = workerData;
  const executionTime = process / speed;
  setTimeout(() => {
    console.log(`Процесс выполнен за ${executionTime} секунд`);
    parentPort.postMessage('done');
  }, executionTime * 1000);
}