import logging
import os
import datetime
import pprint
from typing import Any, Mapping, Optional
import glob
from trainer import MaxTextTrainer
from cloud_utils import upload_file_to_gcs, iterate_and_write_to_gcs, get_job_submission_id
from dataclasses import dataclass, field
from elastic_utils import ActorInfo, ActorInfoCollection

import re
import ray
import time
from ray_tpu import RayTpuManager


_BASE_GCS_BUCKET = "gs://elaxtext-us-central2/"
tz = datetime.timezone.utc
ft = "%Y-%m-%dT%H:%M:%S%z"
current_time_in_utc = datetime.datetime.now(tz=tz).strftime(ft)
EXPERIMENT_NAME = f"run-t-{current_time_in_utc}"
EXPERIMENT_BUCKET = os.path.join(_BASE_GCS_BUCKET, EXPERIMENT_NAME)
COMPILE_CACHE_DIR = os.path.join(_BASE_GCS_BUCKET, "compile_cache")
STEPS_PER_LOOP = 10
POLL_RATE_IN_S = 5

#### Configurations
# Flags that go into MaxText
MAXTEXT_CONFIG = dict(
    ici_fsdp_parallelism=8,
    tokenizer_path="assets/tokenizer",
    steps=100,
    per_device_batch_size=8,
    profiler="xplane",
    global_parameter_scale=1,
    checkpoint_period=10,
    enable_checkpointing=True,
    remat_policy="full",
    base_emb_dim=6144,
    base_num_kv_heads=24,
    base_num_query_heads=24,
    base_mlp_dim=24576,
    base_num_decoder_layers=16,
    base_output_directory="gs://elaxtext-us-central2/runs/",
    dataset_path="gs://elaxtext-us-central2/maxtext-data",
)


# XLA runtime args
XLA_RUNTIME_FLAGS = (
    "TPU_MEGACORE=MEGACORE_DENSE "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_megascale_barrier=true "
    "--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 "
    "--xla_enable_async_collective_permute=true "
    "--xla_jf_rematerialization_percent_shared_memory_limit=97 "
    "--xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_threshold_for_allgather_cse=10 "
    "--xla_tpu_prefuse_self_attention=false --xla_tpu_rwb_fusion=false "
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_dcn_max_overlap_estimation=32.0 "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=10 "
    "--megascale_enable_async_host_commands=true "
    "--xla_tpu_spmd_rng_bit_generator_unsafe=true"
)

# Env vars that run on all TPU VMs.
MACHINE_ENV_VARS = {
    #"TPU_STDERR_LOG_LEVEL": "0",
    #"TPU_MIN_LOG_LEVEL": "0",
    #"TF_CPP_MIN_LOG_LEVEL": "0",
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto",
    "LIBTPU_INIT_ARGS": XLA_RUNTIME_FLAGS,
}


# TODO - merge ElasticState and ActorInfo
@dataclass
class ElasticState:
    """Convenience class to handle Ray-related state."""
    handles: list[ray.actor.ActorHandle] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    futures: list[ray.actor.ActorHandle] = field(default_factory=list)

    def clear(self):
        logging.info("Clearing elastic state.")
        for future in self.futures:
            ray.cancel(future)
        for handle in self.handles:
            ray.kill(handle)
        self.handles = []
        self.names = []
        self.futures = []
        ActorInfoCollection.clear()
        # Quick wait for ray state to update.
        time.sleep(3)


def log_and_store_experiment_config():
    pp = pprint.PrettyPrinter(indent=4)
    logging.info("MaxText job config: %s", pp.pformat(MAXTEXT_CONFIG))
    logging.info("XLA runtime flags: %s", pp.pformat(XLA_RUNTIME_FLAGS))
    logging.info("Machine env vars: %s", pp.pformat(MACHINE_ENV_VARS))


def setup_loggers():
    logging.basicConfig(level=logging.INFO)


def serialize_config_to_name(config: Mapping[str, Any]) -> str:
    """Seralizes an experiment config to a name."""
    return f"emb_{config['base_emb_dim']}_kv_{config['base_num_kv_heads']}_q_{config['base_num_query_heads']}_mlp_{config['base_mlp_dim']}_nl_{config['base_num_decoder_layers']}"


def main():
    ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
    run_name = get_job_submission_id()
    logging.info("This job ID: %s", run_name)

    config = MAXTEXT_CONFIG
    logging.info("Maxtext config: %s", pprint.pformat(config))

    total_steps = config["steps"]
    step = 0

    state = ElasticState()

    while step < total_steps:
        # Run initialization steps in case of:
        # 1) cold start,
        # 2) after failure, or
        # 3) in reduced batch (elastic) mode.
        # In any case, this is detectable if there are any available
        # TPU resources being unused.
        tpu_resources = RayTpuManager.get_available_resources()
        if tpu_resources:
            if state.handles:
                # Implies that we're currently running in elastic mode.
                logging.info(
                    "Detected available TPU resources (%s), implying we are running in elastic mode. "
                    "Re-scaling the run.", tpu_resources)
                state.clear()
                # After state is cleared, get the full cluster view of the resources.
                tpu_resources = RayTpuManager.get_available_resources()

            logging.info("Detected tpu resources: %s", tpu_resources)
            if len(tpu_resources.keys()) > 1:
                logging.warning(f"Detected more than one registered TPU type in the cluster: {list(tpu_resources.keys())}. "
                                "Currently we only support on TPU pod slice at a time.")

            pod_type = list(tpu_resources.keys())[0]
            logging.info("Running on pod type: %s", pod_type)

            compile_cache_dir = os.path.join(
                COMPILE_CACHE_DIR,
                serialize_config_to_name(config),
                f"{len(tpu_resources[pod_type])}_{pod_type}")
            logging.info("Setting compile cache dir to %s", compile_cache_dir)
            config["jax_cache_dir"] = compile_cache_dir

            # TODO - set the batch size, learning rate, dcn setting
            state.handles = RayTpuManager.create_actor(
                tpus=tpu_resources[pod_type],
                actor_def=MaxTextTrainer,
                multislice=True,
                config=config)
            state.names = ray.get([h.get_host_name.remote() for h in state.handles])
            try:
                logging.info("Initializing workload")
                ray.get([h.initialize.remote(run_name=run_name) for h in state.handles])
            except Exception as e:
                logging.info("Caught error during initializations: %s", e)
                logging.info("Shutting down.")
                ray.shutdown()
                raise e

        # Train loop
        logging.info("Running MaxText for %d steps", STEPS_PER_LOOP)
        failure_detected = False
        results = []
        for handle, name in zip(state.handles, state.names):
            future = handle.train.remote(num_steps=STEPS_PER_LOOP)
            ActorInfoCollection.register(ActorInfo(future=future, handle=handle, host_name=name))
            state.futures.append(future)

        while state.futures:
            done_futures, state.futures = ray.wait(state.futures, timeout=POLL_RATE_IN_S)
            for f in done_futures:
                try:
                    results.append(ray.get(f))
                except ray.exceptions.RayActorError as e:
                    failed = ActorInfoCollection.get(f)
                    failure_detected = True
                    logging.info("Detected failure for host %s. Error: %s", failed.host_name, e)
            if failure_detected:
                break

        if failure_detected:
            logging.info("Detected previous error, resetting Ray state.")
            state.clear()
        else:
            step = results[0]
            logging.info("Finished train loop up to step %d.", step)

    logging.info("Finished running MaxText!")
    logging.info("Beginning to postprocess run results...")

    """
    try:
        ray.get(runner.process_artifacts())
    except Exception as e:
        logging.info("Caught error: %s", e)
        logging.info("Shutting down.")
        ray.shutdown()
        raise e
    """

    logging.info("See experiment results: gsutil ls %s", EXPERIMENT_BUCKET)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main()
