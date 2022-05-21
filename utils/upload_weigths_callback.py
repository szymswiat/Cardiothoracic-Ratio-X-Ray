from pathlib import Path

from clearml import Task
from pytorch_lightning import Callback
import pytorch_lightning as pl
from tempfile import mkdtemp

from utils.loadable_module import LoadableModule


class UploadWeightsCallback(Callback):

    def __init__(
            self,
            interval: int,
            task: Task
    ):
        self._interval = interval
        self._tmp_dir = Path(mkdtemp())
        self._task = task

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        model: LoadableModule = pl_module.model

        if trainer.current_epoch % self._interval != 0 or trainer.current_epoch == 0:
            return

        w_path = self._tmp_dir / f'weights_{trainer.current_epoch}.pt'

        model.save_state_to_file(w_path)
        self._task.update_output_model(w_path.as_posix(), tags=['interval_weights'])
