import os
import json
import tqdm
import torch
import random
import bmtrain as bmt
from utils import teacher_forcing
from utils import initialize, setup_model_and_optimizer

INPUT_DIR = "./data/ppl_annotation/tasks"
OUTPUT_FILE = "./data/data/ppl_annotation_examples/res.jsonl"
SPLIT_PATH = "./data/ppl_annotation/splits/split_1.txt"


def valid_json():
    with open(SPLIT_PATH) as f:
        lines = f.readlines()
    valid_tasks = [line[:-1] + ".json" for line in lines]
    return valid_tasks


def test_mean_ppl(args, tokenizer, model, prefix_list, suffix_list):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    model.eval()

    with torch.no_grad():
        input_ids = []
        input_length = []
        index = []

        for s_input_str in prefix_list:
            s_input = tokenizer.encode(s_input_str)
            s_length = len(s_input)

            max_decoder_length = args.max_decoder_length

            s_input_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
            s_input_tokens[:s_length] = torch.tensor(s_input).int()

            s_index = torch.zeros((max_decoder_length,), dtype=torch.int32)
            s_index[s_length - 1] = 1
            s_input_length = torch.tensor(s_length, dtype=torch.long)

            input_ids.append(s_input_tokens)
            input_length.append(s_input_length)
            index.append(s_index)

        input_ids = torch.stack(input_ids).cuda()
        input_length = torch.stack(input_length).cuda()
        index = torch.stack(index).cuda()

        assert len(suffix_list) == len(prefix_list)
        loss = teacher_forcing(model, input_ids, input_length, index, suffix_list, tokenizer, loss_func)
        return loss.item()


def evaluate_a_json(args, tokenizer, model, json_file, test_num=32):
    with open(json_file) as f:
        data = json.load(f)

    template = ""
    template += f"Definition: {data['Definition']}\n\n"

    template += "Positive Examples 1\n"
    template += f"Input: {data['Positive Examples'][0]['input']}\n"
    template += f"Output: {data['Positive Examples'][0]['output']}\n\n"

    template += "Positive Examples 2\n"
    template += f"Input: {data['Positive Examples'][1]['input']}\n"
    template += f"Output: {data['Positive Examples'][1]['output']}\n\n"

    template += "Now complete the following example -\nInput: "

    random.shuffle(data["Instances"])
    if len(data["Instances"]) >= test_num:
        instances = data["Instances"][:test_num]
    else:
        instances = data["Instances"]

    prefix_lst = []
    suffix_lst = []
    for instance in instances:
        prefix_lst.append(template + instance["input"] + "\nOutput: ")
        suffix_lst.append(instance["output"][0] if instance["output"][0] != "" else "null")
    try:
        res = [test_mean_ppl(args, tokenizer, model, prefix_lst[:8], suffix_lst[:8]),
               test_mean_ppl(args, tokenizer, model, prefix_lst[8:16], suffix_lst[8:16]),
               test_mean_ppl(args, tokenizer, model, prefix_lst[16:24], suffix_lst[16:24]),
               test_mean_ppl(args, tokenizer, model, prefix_lst[24:], suffix_lst[24:])]
        return sum(res) / len(res)
    except:
        return None


def main():
    args = initialize()
    print(args)
    tokenizer, the_model, _, _ = setup_model_and_optimizer(args)

    valid_content = valid_json()

    input_json = []
    for file in os.listdir(INPUT_DIR):
        if file.find("of32sample") != -1:
            input_json.append(file)

    # sort_with_names
    task_instruct_dict = {}
    for file in input_json:
        task_name = file[file.find("32sample_") + len("32sample_"):]
        if task_name in valid_content:
            if task_name in task_instruct_dict:
                task_instruct_dict[task_name].append(file)
            else:
                task_instruct_dict[task_name] = [file]

    with open(OUTPUT_FILE, 'r') as g:
        lines = g.readlines()
        data = [json.loads(line)["task"] for line in lines]

    # evaluate & write
    with open(OUTPUT_FILE, 'a') as f:
        for task_name in tqdm.tqdm(task_instruct_dict):
            if task_name not in data:
                res = {"task": task_name, "CEloss": {}}
                for file in task_instruct_dict[task_name]:
                    loss = evaluate_a_json(args, tokenizer, the_model, os.path.join(INPUT_DIR, file))
                    if loss is None:
                        print(f"{file} meet a null!")
                    res["CEloss"][file] = loss
                f.write(f"{json.dumps(res)}\n")
                f.flush()


if __name__ == "__main__":
    main()
