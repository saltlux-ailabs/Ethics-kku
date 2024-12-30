import os
import torch
from datasets import Dataset
import argparse
import wandb


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    Trainer
)


from peft import LoraConfig, get_peft_model

def process_data(example, tokenizer):
    system_prompt = "You are a helpful assistant.."  

    message_full = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{example['instruction']} \n\n{example['input']}"},
        {"role": "assistant", "content": example["output"]}]
    
    message_ahead = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{example['instruction']} \n\n{example['input']}"}]
    
    message_full = tokenizer.apply_chat_template(message_full, tokenize=False)
    message_ahead = tokenizer.apply_chat_template(message_ahead, tokenize=False)

    tokenized_full = tokenizer(message_full, add_special_tokens=False)
    tokenized_ahead = tokenizer(message_ahead, add_special_tokens=False)
    
    input_ids = tokenized_full.input_ids
    attention_mask = tokenized_full.attention_mask
    labels = tokenized_full.input_ids.copy()
    labels[:len(tokenized_ahead.input_ids)] = [-100] * len(tokenized_ahead.input_ids)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--model', type=str, help='type')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epoch', type=int, help='epoch')
    args = parser.parse_args()
    base_model = args.model
    
    if base_model == 'Qwen2':
        base_model_name = "Qwen/Qwen2-7B-Instruct"
    elif base_model == "Qwen2.5":
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    elif base_model == 'Mistral':
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif base_model == "solar":
        base_model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
    elif base_model == "kullm3":
        base_model_name = "nlpai-lab/KULLM3"
    elif base_model == "luxia":
        base_model_name = "./unethical-luxia2-8b-instruct-v1.0"
    else:
        base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    # base_model = "Mistral"

    # cache_dir = "./cache"
    output_dir = "./model_output2"
    new_model_name = f"{base_model}_{args.epoch}_{args.lr}-trained"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,	# 모델 가중치를 4비트로 로드
        bnb_4bit_quant_type="nf4",	# 양자화 유형
        bnb_4bit_compute_dtype=torch.float16,	
        bnb_4bit_use_double_quant=False,	
    )

    # 모델 설정
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", quantization_config=quant_config)
    # base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    # base_model = base_model.to("cuda")
    # base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # 토크나이저 설정    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  
    
    train_dataset = Dataset.from_json('./train_v2.json')
    processed_dataset = train_dataset.map(lambda example: process_data(example, tokenizer))
    

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 로라 설정
    peft_params = LoraConfig(
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        r=16,  # lora_rank
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, peft_params)
    print_trainable_parameters(base_model) 
    
    wandb.init(project="korean-ethics-llm")
    wandb.run.name = base_model_name

    # 학습 파라미터 설정
    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        # save_steps=25,
        logging_steps=10,
        learning_rate=args.lr, 
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        do_train=True,
    )

    # Train on completions only
    # collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    trainer = Trainer(
        model=base_model,
        train_dataset=processed_dataset,
        data_collator=collator,
        args=training_params,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, new_model_name))


    torch.cuda.empty_cache()
