import hydra
from clearml import Task
from omegaconf import DictConfig, OmegaConf


@hydra.main('../config', 'train_config')
def train(cfg: DictConfig):
    task_args = dict(auto_connect_frameworks=False,
                     output_uri=True)

    task_args.update(dict(project_name=cfg.project_name,
                          task_name=cfg.task_name))

    if cfg.run_config.offline:
        Task.set_offline(True)

    Task.force_requirements_env_freeze(requirements_file='requirements.txt')
    task = Task.init(**task_args)

    task.set_parameters_as_dict(OmegaConf.to_object(cfg.hparams))

    if cfg.run_config.remote:
        task.execute_remotely(exit_process=True)

    print('Training function reached. Aborting.')


if __name__ == '__main__':
    train()
