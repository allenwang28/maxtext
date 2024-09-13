import glob
import os
from typing import Optional
import logging
from google.cloud import storage
import ray
from ray.job_submission import JobSubmissionClient


def upload_file_to_gcs(local_file_path: str, gcs_dir: str, file_prefix: Optional[str] = None):
    """Uploads a file to Google Cloud Storage.

    Args:
        local_file_path (str): The path to the local file to upload.
        gcs_dir (str): The GCS directory to upload the file to, in the format 'gs://bucket-name/path/to/dir'.
        file_prefix (str): An optional string that can be used to prefix the name of the file written in GCS.
    """
    if not os.path.isfile(local_file_path):
        print(f"File not found: {local_file_path}")
        return

    # Initialize GCS client
    client = storage.Client()

    # Parse the bucket name and the path in the bucket from gcs_dir
    bucket_name, gcs_path = gcs_dir[5:].split('/', 1)
    bucket = client.get_bucket(bucket_name)

    # Create a blob (GCS file object) in the bucket
    if file_prefix:
        new_fname = f"{file_prefix}-{os.path.basename(local_file_path)}"
    else:
        new_fname = os.path.basename(local_file_path)
    blob = bucket.blob(os.path.join(gcs_path, new_fname))
    # Upload the file
    blob.upload_from_filename(local_file_path)


def iterate_and_write_to_gcs(
    local_dir: str,
    gcs_dir: str,
    file_prefix: Optional[str] = None,
    remove: bool = False):
    """Iterates over a provided directory, writes to GCS, and (optionally) removes artifacts.

    Args:
        local_dir (str): The local directory to iterate over.
        gcs_dir (str): The GCS directory to write files to.
        file_prefix (str): an optional prefix that can be supplied to the new filename.
        remove (bool, optional): If True, remove local files after uploading. Defaults to False.
    """

    # Iterate over files in the local directory
    for local_file in glob.glob(os.path.join(local_dir, "**"), recursive=True):
        if os.path.isfile(local_file):
            # Upload the file to GCS
            upload_file_to_gcs(local_file, gcs_dir, file_prefix=file_prefix)
            # Remove the file if remove flag is set
            if remove:
                os.remove(local_file)
                logging.debug(f"Removed {local_file}")


def get_job_submission_id() -> str:
    """Returns the Ray job submission ID."""
    c = JobSubmissionClient()
    current_job_id = ray.get_runtime_context().get_job_id()
    jobs = c.list_jobs()
    return [job.submission_id for job in jobs if job.job_id == current_job_id][0]
