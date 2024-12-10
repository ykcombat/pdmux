# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional

import psutil
import torch

from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch, ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True)
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port)
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int32, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # for pd multiplexing
        self.split_prefill_queue = Queue()
        self.forward_stream = torch.cuda.Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_cpu_group(self):
        return self.worker.get_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool,
        )

    def forward_thread_func(self):
        try:
            with torch.cuda.stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    @torch.no_grad()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        split_batch_pt = 0
        split_batch_lists = [None] * 2

        while True:
            decode_model_worker_batch = None
            split_prefill_batch = None
            if not self.input_queue.empty():
                decode_model_worker_batch, decode_future_token_ids_ct = self.input_queue.get()
            if not self.split_prefill_queue.empty():
                split_prefill_batch, prefill_future_token_ids_ct, forward_times, prefill_finished = self.split_prefill_queue.get()
            if not decode_model_worker_batch and not split_prefill_batch:
                continue

            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            if decode_model_worker_batch:
                batch_lists[batch_pt % 2] = decode_model_worker_batch
                batch_pt += 1

                # Create event
                self.launch_done = threading.Event()
                copy_done = torch.cuda.Event()

                # Resolve future tokens in the input
                input_ids = decode_model_worker_batch.input_ids
                resolve_future_token_ids(input_ids, self.future_token_ids_map)

                # Run forward
                logits_output, next_token_ids = self.worker.forward_batch_generation(
                    decode_model_worker_batch, self.launch_done
                )

                # Update the future token ids map
                bs = len(decode_model_worker_batch.seq_lens)
                self.future_token_ids_map[
                    decode_future_token_ids_ct + 1 : decode_future_token_ids_ct + bs + 1
                ] = next_token_ids

                # Copy results to the CPU
                if decode_model_worker_batch.return_logprob:
                    logits_output.next_token_logprobs = logits_output.next_token_logprobs[
                        torch.arange(len(next_token_ids), device=self.device),
                        next_token_ids,
                    ].to("cpu", non_blocking=True)
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = (
                            logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                        )
                        logits_output.normalized_prompt_logprobs = (
                            logits_output.normalized_prompt_logprobs.to(
                                "cpu", non_blocking=True
                            )
                        )
                next_token_ids = next_token_ids.to("cpu", non_blocking=True)
                copy_done.record()

                self.output_queue.put((copy_done, logits_output, next_token_ids))


            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.

            if split_prefill_batch:
                split_batch_lists[split_batch_pt % 2] = split_prefill_batch
                split_batch_pt += 1

                # Create event
                self.split_launch_done = threading.Event()
                copy_done = torch.cuda.Event()

                # Resolve future tokens in the input
                prefill_model_worker_batch = split_prefill_batch.get_model_worker_batch()
                input_ids = prefill_model_worker_batch.input_ids
                resolve_future_token_ids(input_ids, self.future_token_ids_map)

                # Run forward
                logits_output = None
                next_token_ids = None
                for _ in range(forward_times):
                    logits_output, next_token_ids = self.worker.forward_batch_split_prefill(
                        split_prefill_batch
                    )
                self.split_launch_done.set()

                if prefill_finished:
                    # Update the future token ids map
                    bs = len(prefill_model_worker_batch.seq_lens)
                    self.future_token_ids_map[
                        prefill_future_token_ids_ct + 1 : prefill_future_token_ids_ct + bs + 1
                    ] = next_token_ids

                    # Copy results to the CPU
                    if prefill_model_worker_batch.return_logprob:
                        logits_output.next_token_logprobs = logits_output.next_token_logprobs[
                            torch.arange(len(next_token_ids), device=self.device),
                            next_token_ids,
                        ].to("cpu", non_blocking=True)
                        if logits_output.input_token_logprobs is not None:
                            logits_output.input_token_logprobs = (
                                logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                            )
                            logits_output.normalized_prompt_logprobs = (
                                logits_output.normalized_prompt_logprobs.to(
                                    "cpu", non_blocking=True
                                )
                            )
                    next_token_ids = next_token_ids.to("cpu", non_blocking=True)
                    copy_done.record()

                    self.output_queue.put((copy_done, logits_output, next_token_ids))


            

    def resolve_batch_result(self, bid: int, split_prefill=False):
        copy_done, logits_output, next_token_ids = self.output_queue.get()
        copy_done.synchronize()
        if split_prefill:
            self.split_launch_done.wait()
        else:
            self.launch_done.wait()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = (
                    logits_output.input_token_logprobs.tolist()
                )
                logits_output.normalized_prompt_logprobs = (
                    logits_output.normalized_prompt_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            scaling_penalties=sampling_info.scaling_penalties,
            linear_penalties=sampling_info.linear_penalties,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        torch.cuda.current_stream().synchronize()

        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids
    
    def forward_batch_split_prefill(self, schedule_batch: ScheduleBatch, forward_times=1):
        model_worker_batch = schedule_batch.get_model_worker_batch()
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            scaling_penalties=sampling_info.scaling_penalties,
            linear_penalties=sampling_info.linear_penalties,
        )
        if schedule_batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.worker.model_runner)
            forward_batch.forward_mode = schedule_batch.forward_mode
            schedule_batch.split_forward_batch = forward_batch

        schedule_batch.split_index = schedule_batch.split_index + forward_times

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        torch.cuda.current_stream().synchronize()

        # Push a new batch to the queue
        prefill_finished = schedule_batch.split_index == schedule_batch.split_max_index
        self.split_prefill_queue.put((schedule_batch, self.future_token_ids_ct, forward_times, prefill_finished))

        # Allocate output future objects
        future_next_token_ids = None
        if prefill_finished:
            bs = len(model_worker_batch.seq_lens)
            future_next_token_ids = torch.arange(
                -(self.future_token_ids_ct + 1),
                -(self.future_token_ids_ct + 1 + bs),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.future_token_ids_ct = (
                self.future_token_ids_ct + bs
            ) % self.future_token_ids_limit
        return None, future_next_token_ids

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))
