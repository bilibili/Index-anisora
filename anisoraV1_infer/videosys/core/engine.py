import os
from functools import partial
from typing import Any, Optional

import torch
import torch.distributed as dist

import videosys

from .mp_utils import ProcessWorkerWrapper, ResultHandler, WorkerMonitor, get_distributed_init_method, get_open_port


class VideoSysEngine:
    """
    this is partly inspired by vllm
    """

    def __init__(self, config):
        self.config = config
        self.parallel_worker_tasks = None
        self._init_worker(config.pipeline_cls)

    def _init_worker(self, pipeline_cls):
        world_size = self.config.num_gpus

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Set OMP_NUM_THREADS to 1 if it is not set explicitly, avoids CPU
        # contention amongst the shards
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"

        # NOTE: The two following lines need adaption for multi-node
        assert world_size <= torch.cuda.device_count()

        # change addr for multi-node
        distributed_init_method = get_distributed_init_method("127.0.0.1", get_open_port())

        if world_size == 1:
            self.workers = []
            self.worker_monitor = None
        else:
            result_handler = ResultHandler()
            self.workers = [
                ProcessWorkerWrapper(
                    result_handler,
                    partial(
                        self._create_pipeline,
                        pipeline_cls=pipeline_cls,
                        rank=rank,
                        local_rank=rank,
                        distributed_init_method=distributed_init_method,
                    ),
                )
                for rank in range(1, world_size)
            ]

            self.worker_monitor = WorkerMonitor(self.workers, result_handler)
            result_handler.start()
            self.worker_monitor.start()

        self.driver_worker = self._create_pipeline(
            pipeline_cls=pipeline_cls, distributed_init_method=distributed_init_method
        )

    # TODO: add more options here for pipeline, or wrap all options into config
    def _create_pipeline(self, pipeline_cls, rank=0, local_rank=0, distributed_init_method=None):
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"] = str(self.config.num_gpus)
        videosys.initialize(rank=rank, world_size=self.config.num_gpus, init_method=distributed_init_method)

        pipeline = pipeline_cls(self.config)
        return pipeline

    def _run_workers(
        self,
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""

        # Start the workers first.
        worker_outputs = [worker.execute_method(method, *args, **kwargs) for worker in self.workers]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return worker_outputs

        driver_worker_method = getattr(self.driver_worker, method)
        driver_worker_output = driver_worker_method(*args, **kwargs)

        # Get the results of the workers.
        return [driver_worker_output] + [output.get() for output in worker_outputs]

    def _driver_execute_model(self, *args, **kwargs):
        return self.driver_worker.generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._run_workers("generate", *args, **kwargs)[0]

    def stop_remote_worker_execution_loop(self) -> None:
        if self.parallel_worker_tasks is None:
            return

        parallel_worker_tasks = self.parallel_worker_tasks
        self.parallel_worker_tasks = None
        # Ensure that workers exit model loop cleanly
        # (this will raise otherwise)
        self._wait_for_tasks_completion(parallel_worker_tasks)

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        for result in parallel_worker_tasks:
            result.get()

    def save_video(self, video, output_path):
        return self.driver_worker.save_video(video, output_path)

    def shutdown(self):
        if (worker_monitor := getattr(self, "worker_monitor", None)) is not None:
            worker_monitor.close()
        dist.destroy_process_group()

    def __del__(self):
        self.shutdown()
