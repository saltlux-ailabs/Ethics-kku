import os
import torch
import json
from datasets import Dataset, load_dataset
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

from utils import extract_model_response, define_output_format, postproc_response


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--model', type=str, default='luxia', help='type')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')
    args = parser.parse_args()
    base_model = args.model
    
    if base_model == 'Qwen2':
        base_model_name = "Qwen/Qwen2-7B-Instruct"
    elif base_model == 'Mistral':
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif base_model == 'solar':
        base_model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
    elif base_model == 'kullm3':
        base_model_name = "nlpai-lab/KULLM3"
    elif base_model == 'luxia':
        base_model_name = "./unethical-luxia2-8b-instruct-v1.0"
    elif base_model == 'bllossom':
        base_model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    else:
        base_model_name = "meta-llama/Llama-3.1-8B-Instruct"

    new_model_name = f"{base_model}_{args.epoch}_{args.lr}-trained"
    # 모델 설정
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,	# 모델 가중치를 4비트로 로드
        bnb_4bit_quant_type="nf4",	# 양자화 유형
        bnb_4bit_compute_dtype=torch.float16,	
        bnb_4bit_use_double_quant=False,	
    )
    
    
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", quantization_config=quant_config)
    # model = PeftModel.from_pretrained(base_model, f"./model_output2/{new_model_name}")
    # model = model.to("cuda")
    # 토크나이저 설정    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # 데이터
    test_dataset = load_dataset("json", data_files={"test": './test_final.json'})

    all_results = []
    cnt_correct, cnt_incorrect = 0, 0
    task_total = {
        "concept": 0,
        "situation": 0,
    }
    type_total = {
        "behavior": 0,
        "conversation": 0
    }
    task_correct = {
        "concept": 0,
        "situation": 0,
    }
    type_correct = {
        "behavior": 0,
        "conversation": 0
    }
    
    #####################
    for ex_id, example in enumerate(tqdm(test_dataset["test"], desc="inference and evaluate results...")):
        answer = example['instances'][0]['output'].strip()
        task = example['task']
        task_type = example['type']
        # inference
        if "KULLM" not in base_model_name:
            system_prompt = "You are a helpful assistant."
        
        elif "KULLM" in base_model_name:
            system_prompt = "당신은 고려대학교 NLP&AI 연구실에서 만든 AI 챗봇입니다.\n당신의 이름은 'KULLM'으로, 한국어로는 '구름'을 뜻합니다.\n당신은 비도덕적이거나, 성적이거나, 불법적이거나 또는 사회 통념적으로 허용되지 않는 발언은 하지 않습니다.\n사용자와 즐겁게 대화하며, 사용자의 응답에 가능한 정확하고 친절하게 응답함으로써 최대한 도와주려고 노력합니다.\n질문이 이상하다면, 어떤 부분이 이상한지 설명합니다. 거짓 정보를 발언하지 않도록 주의합니다."  

        output_format = define_output_format(example)
        message = example['instruction'] + " " + output_format + "\n\n" + example['instances'][0]['input']


        conversation = [
            {"role": "system", "content": system_prompt},  
            {"role": "user", "content": message}
        ]

        # inputs = tokenizer.apply_chat_template(
        #     conversation,
        #     tokenize=True,
        #     add_generation_prompt=False,
        #     return_tensors='pt').to("cuda")
        
        inputs = tokenizer(message, return_tensors='pt').to("cuda")


        outputs = model.generate(inputs['input_ids'], max_new_tokens=32, use_cache=True)

        if "llama" in base_model_name or "bllossom" in base_model_name:
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = extract_model_response(output_text, ex_id,  resp_sp_tok="assistant\n").strip()

        elif "KULLM" in base_model_name:
            output_text = outputs[:, inputs.shape[1]:][0]  # 모델 응답 부분만 추출
            response = tokenizer.decode(output_text, skip_special_tokens=True).strip()
            response = postproc_response(base_model_name, response)
            # 모델 응답만 추출
            # print()
            # print("ex_id: ", ex_id)
            # print("response: ", response)
            # print("answer: ", answer)

        elif "Qwen" in base_model_name:
            output_text = outputs[:, inputs.shape[1]:][0]  # 모델 응답 부분만 추출
            response = tokenizer.decode(output_text, skip_special_tokens=True)   
            response = response.lstrip("assistant\n")
            if ":" in response:
                response = response.split(":")[1].lstrip(" ")         
                
        elif "SOLAR" in base_model_name:
            output_text = outputs[:, inputs.shape[1]:][0]  # 모델 응답 부분만 추출
            response = tokenizer.decode(output_text, skip_special_tokens=True)
            response = response.replace("### Assistant:\n", "")
            
        elif "luxia" in base_model_name:
            output_text = outputs[:, inputs['input_ids'].shape[1]:][0]  # 모델 응답 부분만 추출
            response = tokenizer.decode(output_text, skip_special_tokens=True)
            # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        original_cnt = cnt_correct

        # 평가
        if "classification" in task or task == "sentiment_analysis":
            if "eeve" in base_model_name or "KULLM" in base_model_name or "luxia" in base_model_name:
                if answer in response: 
                    cnt_correct += 1
                else:
                    cnt_incorrect += 1            

            else:
                if response == answer:
                    cnt_correct += 1
                    
                else:
                    cnt_incorrect += 1
            
        elif "multiple-choice" in task:
            if answer in response:
                cnt_correct += 1
            else:
                cnt_incorrect += 1
                
        if 'situation' in task or 'sentiment' in task:
            task_total['situation'] += 1
        elif 'concept' in task:
            task_total['concept'] += 1
            
        if 'behavior' in task_type:
            type_total['behavior'] += 1
        elif 'conversation' in task_type:
            type_total['conversation'] += 1
        
        if original_cnt != cnt_correct:
            if 'situation' in task or 'sentiment' in task:
                task_correct['situation'] += 1
            elif 'concept' in task:
                task_correct['concept'] += 1
                
            if 'behavior' in task_type:
                type_correct['behavior'] += 1
            elif 'conversation' in task_type:
                type_correct['conversation'] += 1


        result = {"ex_id": ex_id, "response": response}
        all_results.append(result)

    torch.cuda.empty_cache()

    # 결과 파일 저장
    # with open(result_path, "w", encoding="utf-8") as wf:
    #     json.dump(all_results, wf, ensure_ascii=False, indent="\t")
    # print()
    # print(result_file, " saved!")

    accuracy = cnt_correct/len(test_dataset["test"])
    task_accuracy = {
        "concept": task_correct['concept'] / task_total['concept'],
        "situation": task_correct['situation'] / task_total['situation'],
    }
    type_accuracy = {
        "behavior": type_correct['behavior'] / type_total['behavior'],
        "conversation": type_correct['conversation'] / type_total['conversation'],
    }

    print()
    print("*****evaluation result*****")
    print("accuracy: ", accuracy)
    print("task_accuracy: ", task_accuracy)
    print("type_accuracy: ", type_accuracy)
    print("cnt_correct: ", cnt_correct)
    print("cnt_incorrect: ", cnt_incorrect)
    print("total: ", len(test_dataset["test"]))
    
    # result_path = f"./results_1/{base_model_name.split('/')[-1]}_1017.json"
    # result_path = f"./result_2/{new_model_name}.json"
    result_path = f"./result_2/bllossom_base.json"
    evaluation_summary = {
        "accuracy": accuracy,
        "cnt_correct": cnt_correct,
        "cnt_incorrect": cnt_incorrect,
        "task_accuracy": task_accuracy,
        "type_accuracy": type_accuracy,
        "total": len(test_dataset["test"])
    }
    with open(result_path, "w", encoding="utf-8") as wf:
        json.dump({"results": all_results, "summary": evaluation_summary}, wf, ensure_ascii=False, indent=4)
    print(result_path, " saved!")