"""Experimental utilities for running Ray with Cloud TPU."""
import re
import logging
from typing import List, Mapping, Union, Type
import socket
import ray
from dataclasses import dataclass
import time


@dataclass
class RayTpu:
    name: str
    num_hosts: int
    head_ip: str
    topology: str


class RayTpuManager:

    @classmethod
    def get_available_resources(cls) -> Mapping[str, RayTpu]:
        resources = {}
        tpu_pattern = re.compile(r"TPU-(.+)-head")

        @ray.remote
        def _get_tpu_pod_metadata():
            """Gets the TPU metadata from TPU leaders."""
            # avoid race conditions
            time.sleep(3)
            tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
            num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
            ip = socket.gethostbyname(socket.gethostname())
            return tpu_name, num_hosts, ip

        available_resources = ray.available_resources()
        logging.info("Ray available resources: %s", available_resources)
        for key, value in available_resources.items():
            match = tpu_pattern.match(key)
            if match:
                topology = f"{match.group(1)}"
                topology_key = key
                num_tpu_pods = int(value)
                logging.info(
                    "Found %d TPU pods of type: %s",
                    num_tpu_pods, topology)
                metadata_handles = []
                for _ in range(num_tpu_pods):
                    metadata_handles.append(_get_tpu_pod_metadata.options(
                        resources={topology_key: 1}
                    ).remote())
                logging.debug("Gathering TPU pod metadata")
                metadata = ray.get(metadata_handles)

                resources[topology] = []
                for tpu_name, num_hosts, head_ip in metadata:
                    resources[topology].append(RayTpu(
                        name=tpu_name,
                        num_hosts=num_hosts,
                        head_ip=head_ip,
                        topology=topology,
                    ))
        return resources

    @classmethod
    def create_actor(cls,
                     tpus: List[RayTpu],
                     multislice: bool,
                     actor_def: Union[ray.actor.ActorClass, Type],
                     *args, **kwargs) -> List[ray.actor.ActorHandle]:
        """Schedules an actor definition on a set of TPUs.

        Args:
            tpus: The list of TPU information.
            actor_def: The definition of the actor, as a class or as a remote class.
            multislice: Whether or not to schedule this actor with multislice technology.
                If set to true, this injects the metadata needed to schedule a multislice workload.
                Else, this will be treated as individual pod slices.

        Returns:
            A list of ActorHandles.
        """
        if isinstance(actor_def, type):
            actor_def = ray.remote(actor_def)
        elif not isinstance(actor_def, ray.actor.ActorClass):
            raise AssertionError(f"`actor_def` should be a class definition or ActorClass, got {type(actor_def)}")

        handles = []

        if multislice:
            logging.info("Scheduling multislice actors.")
            coordinator_port = 8081
            for tpu_id, tpu in enumerate(tpus):
                mxla_env = {
                    "MEGASCALE_COORDINATOR_ADDRESS": f"{tpus[0].head_ip}:{coordinator_port}",
                    "MEGASCALE_NUM_SLICES": str(len(tpus)),
                    "MEGASCALE_PORT": f"{coordinator_port}",
                    "MEGASCALE_SLICE_ID": str(tpu_id),
                }
                # Schedule on the lead worker first to consume the HEAD resource
                handles += [actor_def.options(
                    runtime_env={"env_vars": mxla_env},
                    resources={"TPU": 4, tpu.name: 1, f"TPU-{tpu.topology}-head": 1}).remote(*args, **kwargs)]
                time.sleep(1)
                # Schedule the remaining workers.
                handles += [actor_def.options(
                    runtime_env={"env_vars": mxla_env},
                    resources={"TPU": 4, tpu.name: 1}).remote(*args, **kwargs) for _ in range(tpu.num_hosts - 1)]
        else:
            for tpu in tpus:
                # Schedule on the lead worker first to consume the HEAD resource
                handles += [actor_def.options(resources={"TPU": 4, tpu.name: 1, f"TPU-{tpu.topology}-head": 1}).remote(*args, **kwargs)]
                time.sleep(1)
                handles += [actor_def.options(resources={"TPU": 4, tpu.name: 1}).remote(*args, **kwargs) for _ in range(tpu.num_hosts - 1)]
        return handles
