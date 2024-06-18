from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

def preprocess_data(player_df, game_df):
    # Preprocess and merge data
    data = pd.merge(player_df, game_df, on='player_id')
    return data

def train_gpt_model(data):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
    labels = inputs.input_ids.detach().clone()

    dataset = torch.utils.data.TensorDataset(inputs.input_ids, labels)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    return model
