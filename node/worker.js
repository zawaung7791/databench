// One worker thread per core; processes its range of the shared buffers.
const { parentPort } = require("worker_threads");
const { workerHandler } = require("./bench");
parentPort.on("message", workerHandler((m) => parentPort.postMessage(m)));
