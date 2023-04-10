from transformers import TrainingArguments
import multiprocessing


def config_train_args(learning_rate=0.0006, epochs=30, batch_size=8, output_dir='result', weight_decay=0.1,
                      warmup_ratio=0.1):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        save_total_limit=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=400,
        eval_steps=400,
        logging_steps=100,
        eval_accumulation_steps=1,
        # eval_accumulation_steps=3,  # gpu memory reduce
        # gradient_accumulation_steps=2,  # multiplier for batch_size
        bf16=True,  # скорость плохо работает, нужно подумать
        bf16_full_eval=True,
        dataloader_num_workers=8,  # for 100% load - dataloader_num_workers=multiprocessing.cpu_count()
        load_best_model_at_end=True,
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )
    return training_args


def main():
    pass


if __name__ == '__main__':
    main()
