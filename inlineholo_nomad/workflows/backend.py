# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Access of the actions to NOMAD uploads and to a working folder shared by the workers.

The activities only use the small interface of :class:`Backend`.  In NOMAD this is
:class:`NomadBackend` (raw files of the upload, ``process_upload`` to add results, the artifacts
folder of the action instance as shared working folder).  :class:`LocalBackend` serves a local
folder per upload and is used by the tests and for running the workflows with a plain Temporal
server.  ``set_backend()`` replaces the backend of the worker process.
"""
import os
import shutil
import tempfile
import time

from temporalio import activity
from temporalio.exceptions import ApplicationError


class Backend:
    def check_access(self, upload_id, user_id):
        """Raise ApplicationError if the user may not run the action on the upload."""

    def is_file(self, upload_id, path):
        raise NotImplementedError

    def listdir(self, upload_id, path):
        """[(name, is_file)] of a folder of the upload."""
        raise NotImplementedError

    def copy_out(self, upload_id, path, local):
        raise NotImplementedError

    def add_files(self, upload_id, files, folder):
        """Add local files to a folder of the upload."""
        raise NotImplementedError

    def work_root(self, instance_id):
        """Working folder of one action instance, shared by all workers (CPU and GPU)."""
        raise NotImplementedError

    def tmp_root(self):
        return None


class NomadBackend(Backend):
    def _upload(self, upload_id):
        from nomad.processing import Upload

        try:
            return Upload.get(upload_id)
        except Exception as e:  # noqa: BLE001
            raise ApplicationError(f"upload {upload_id} not found", non_retryable=True) from e

    def _files(self, upload_id):
        from nomad.files import UploadFiles

        return UploadFiles.get(upload_id)

    def check_access(self, upload_id, user_id):
        up = self._upload(upload_id)
        if user_id not in up.writers:
            raise ApplicationError("only the main author and the co-authors of the upload can run this action on it",
                                   non_retryable=True)
        if up.published:
            raise ApplicationError("the upload is published; results cannot be added", non_retryable=True)

    def is_file(self, upload_id, path):
        return self._files(upload_id).raw_path_is_file(path)

    def listdir(self, upload_id, path):
        return [(os.path.basename(e.path), e.is_file) for e in self._files(upload_id).raw_listdir(path)]

    def copy_out(self, upload_id, path, local):
        with self._files(upload_id).raw_file(path, "rb") as fi, open(local, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)

    def add_files(self, upload_id, files, folder, attempts=120):
        """Add the files (NOMAD processes them); waits while the upload is being processed."""
        from nomad.processing.base import ProcessAlreadyRunning

        up = self._upload(upload_id)
        staged = tempfile.mkdtemp(prefix="inlineholo_add_", dir=self.tmp_root())
        ops = []
        for f in files:
            dst = os.path.join(staged, os.path.basename(f))
            shutil.copy2(f, dst)
            ops.append(dict(op="ADD", path=dst, target_dir=folder, temporary=True))
        for _ in range(attempts):
            try:
                up.reload()
                up.process_upload(file_operations=ops, only_updated_files=True)
                return
            except ProcessAlreadyRunning:
                try:
                    activity.heartbeat("waiting for the processing of the upload")
                except RuntimeError:
                    pass
                time.sleep(5)
        raise ApplicationError("the upload stayed busy; the result could not be added")

    def work_root(self, instance_id):
        from nomad.actions.manager import action_instance_artifacts_dir

        return action_instance_artifacts_dir(instance_id)

    def tmp_root(self):
        from nomad.config import config

        d = os.path.abspath(config.fs.tmp)
        os.makedirs(d, exist_ok=True)
        return d


class LocalBackend(Backend):
    """Uploads are folders ``root/<upload_id>``; working folders ``work/<instance_id>``."""

    def __init__(self, root, work=None, writers=None):
        self.root = os.path.abspath(root)
        self.work = os.path.abspath(work or os.path.join(self.root, "_work"))
        self.writers = writers          # {upload_id: {user ids}} or None (everybody)

    def _p(self, upload_id, path):
        return os.path.join(self.root, upload_id, *[s for s in str(path).split("/") if s])

    def check_access(self, upload_id, user_id):
        if not os.path.isdir(os.path.join(self.root, upload_id)):
            raise ApplicationError(f"upload {upload_id} not found", non_retryable=True)
        if self.writers is not None and user_id not in self.writers.get(upload_id, ()):
            raise ApplicationError("not an author of the upload", non_retryable=True)

    def is_file(self, upload_id, path):
        return os.path.isfile(self._p(upload_id, path))

    def listdir(self, upload_id, path):
        d = self._p(upload_id, path)
        return [(n, os.path.isfile(os.path.join(d, n))) for n in sorted(os.listdir(d))]

    def copy_out(self, upload_id, path, local):
        shutil.copy(self._p(upload_id, path), local)

    def add_files(self, upload_id, files, folder):
        d = self._p(upload_id, folder)
        os.makedirs(d, exist_ok=True)
        for f in files:
            shutil.copy(f, os.path.join(d, os.path.basename(f)))

    def work_root(self, instance_id):
        d = os.path.join(self.work, instance_id)
        os.makedirs(d, exist_ok=True)
        return d


_BACKEND = None


def set_backend(backend):
    global _BACKEND
    _BACKEND = backend


def get_backend() -> Backend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = NomadBackend()
    return _BACKEND
