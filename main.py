import argparse
import torch
from transformers import Trainer
from config_training_args import config_train_args
import dataset_and_aug
import gpu_information
import my_segformer_model
from model_metrics import compute_metrics
from hyperparameter_search_optuna import parameter_search


def get_args():
    """
    continue
    python3 segformer_model/main.py  --epochs 30 --batch_size 8 --model_name nvidia/segformer-b5-finetuned-ade-640-640 --output_dir continue --resume_from_checkpoint True

    optuna
    python3 segformer_model/main.py  --epochs 40 --batch_size 8 --model_name nvidia/segformer-b5-finetuned-cityscapes-1024-1024 --output_dir segformer_best --optuna True

    experiment
    python3 segformer_model/main.py  --epochs 2 --batch_size 1 --model_name nvidia/segformer-b0-finetuned-cityscapes-1024-1024 --output_dir segformer-b0-test
    :return: Parsing input arguments
    """

    parser = argparse.ArgumentParser(description='Great Description To Be Here')
    parser.add_argument('-d', '--data_dir',
                        default='/home/maksim/PycharmProjects/pythonProject/data/cargo_segform_train=0.8val=0.2',
                        type=str, action="store",
                        help="Path to data")
    parser.add_argument('-e', '--epochs', default=2, type=int, action="store",
                        help="Number of epochs")
    parser.add_argument('-l', '--learning_rate', default=6e-5, type=float, action="store",
                        help="Learning rate")
    parser.add_argument('-b', '--batch_size', default=4, type=int, action="store",
                        help="Batch size")
    parser.add_argument('-m', '--model_name', default='nvidia/mit-b0', type=str, action="store",
                        help="Model name")
    parser.add_argument('-o', '--output_dir', default='segformer-result', type=str, action="store",
                        help="Output directory")
    parser.add_argument('-s', '--optuna', default=False, type=bool, action="store",
                        help="Optuna hyperparameter search")
    parser.add_argument('-wd', '--weight_decay', default=0.1, type=float, action="store",
                        help="weight_decay - like l2 regularization")
    parser.add_argument('-wr', '--warmup_ratio', default=0.08, type=float, action="store",
                        help="Ratio of total training steps used for a linear warmup")
    parser.add_argument('-r', '--resume_from_checkpoint', default=False, type=bool, action="store",
                        help="Continue train process")
    args = parser.parse_args()
    return args


def main():
    """
    perform parsing arguments -> create dataset -> initialize training arguments -> model -> trainer -> train
    :return: nothing
    """
    # argparse
    args = get_args()
    # create dataset
    train_dataset, valid_dataset = dataset_and_aug.main(args.data_dir)
    # training args
    if args.optuna:  # hyperparameter search
        n_trials = 10  # number trials for search parameters
        lr, weight_decay, warmup_ratio, batch_size = parameter_search(train_dataset,
                                                                      valid_dataset,
                                                                      n_trials)
        training_args = config_train_args(learning_rate=lr,
                                          epochs=args.epochs,
                                          batch_size=batch_size,
                                          output_dir=args.output_dir,
                                          weight_decay=weight_decay,
                                          warmup_ratio=warmup_ratio,
                                          )
    else:
        training_args = config_train_args(learning_rate=args.learning_rate,
                                          output_dir=args.output_dir,
                                          epochs=args.epochs,
                                          batch_size=args.batch_size,
                                          weight_decay=args.weight_decay,
                                          warmup_ratio=args.warmup_ratio,
                                          )

    # Initialize model
    model = my_segformer_model.get_model(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gpu_information.print_gpu_utilization()
    # initialize hugging-face trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    # train
    if args.resume_from_checkpoint:
        result = trainer.train(
            resume_from_checkpoint="/home/maksim/PycharmProjects/pythonProject/continue/checkpoint-4800")
    else:
        result = trainer.train()
    gpu_information.print_summary(result)


if __name__ == '__main__':
    main()
