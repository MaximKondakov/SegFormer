from transformers import Trainer, SegformerForSemanticSegmentation
from config_training_args import config_train_args
from model_metrics import compute_metrics
from config_category_label_and_color import make_labels


def my_objective(metrics):
    return metrics["eval_mean_iou"]  # metrics["eval_loss"


def model_init():
    """
    model initialize for optuna
    нужно убрать лишнюю инициализацию модели
    :return:
    """
    pretrained_model_name = 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024'
    num_labels, id2label, label2id = make_labels()
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True,
    )

    return model


def optuna_hp_space(trial):
    return {
        "output_dir": trial.suggest_categorical("output_dir", ['optuna_iou']),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-1, 0.6, log=True),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 1e-1, 0.5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [50]),
    }


def parameter_search(train_data, val_data, n_trials):
    trainer = Trainer(
        model_init=model_init,
        args=config_train_args(),
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",  # "minimize"
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        compute_objective=my_objective,
    )
    print(f"Best Params:")
    print(f"best run: {best_run.run_id}")
    print(f"best objective: {best_run.objective}")
    for k, v in best_run.hyperparameters.items():
        print(f"\t{k}: {v}")

    best_lr = float(best_run.hyperparameters['learning_rate'])
    best_weight_decay = float(best_run.hyperparameters['weight_decay'])
    best_warmup_ratio = float(best_run.hyperparameters['warmup_ratio'])
    best_per_device_train_batch_size = int(best_run.hyperparameters['per_device_train_batch_size'])

    return best_lr, best_weight_decay, best_warmup_ratio, best_per_device_train_batch_size


def main():
    pass


if __name__ == "__main__":
    main()
