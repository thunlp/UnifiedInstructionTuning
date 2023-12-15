# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractclassmethod
import os
import torch
import re
import json
import random

def is_json_demo(file_name):
    return os.path.basename(file_name).find("jsonl") != -1

def option_shuffle(string):

    def to_block(tuple_lst):
        res = []
        block = []

        for txt in tuple_lst:
            if txt[1].find("A:") == -1: block.append(txt)
            else:
                block.append(txt)
                res.append(block)
                block = []

        return res

    def shuffle_block(involved_blocks):

        res = []
        for block in involved_blocks:

            options = block[:-1]
            answer = block[-1]
            # 原来的答案记录一下
            gold = answer[1][answer[1].find('('): answer[1].find(')')+len(')')]  # (a) like

            # 把这个答案对应的文案记录一下
            gold_txt = ""
            for option in options:
                if option[1].find(gold) != -1:
                    gold_txt = option[1][4:]

            # 把所有的东西shuffle一下
            letters = [option[1][option[1].find('('): option[1].find(')')+len(')')] for option in options]
            txt = [option[1][option[1].find(')') + 1:] for option in options]

            random.shuffle(txt)
            new_options = [letter + new_txt for letter, new_txt in zip(letters, txt)]

            # 找到这个文案现在对应的答案
            for new_option in new_options:
                if new_option.find(gold_txt) != -1:
                    answer[1] = answer[1][:-4] + new_option[new_option.find('('): new_option.find(')')+len(')')] + '.'

            for option, new_option_txt in zip(options, new_options):
                option[1] = new_option_txt
            res.append(options+[answer])

        return res


    lines_involved = []
    # 先把所有的行读进来
    lines = string.split("\n")
    # 然后把所有带有')'的读进来
    for idx in range(len(lines)):
        if lines[idx].find(')') != -1: lines_involved.append([idx, lines[idx]])

    # 然后把他们分分组，如果是A开头的，就截断一下[[],[],[],[],[]]
    blocks_involved = to_block(lines_involved)
    blocks_shuffled = shuffle_block(blocks_involved)

    for block in blocks_shuffled:
        for idx, txt in block:
            lines[idx] = txt

    res = ""
    for line in lines:
        res += line+'\n'

    return res[:-1]


def clear_bookmark(old_str: str):
    return re.sub(r'<<.*>>',"", old_str)

class SuperGLUE(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = []

    # 往data里放一个example
    def make_input(self, tokenizer, template, max_decoder_length, label, rationales=None, solver=None):

        input = tokenizer.encode(template)
        length = len(input)

        # 截断
        if length > max_decoder_length:
            input = input[-max_decoder_length:]
            length = max_decoder_length

        # 补齐
        input_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        # 长度tensor化
        input_length = torch.tensor(length, dtype=torch.long)

        # 初始化target，作用是把input_id的头去掉，而空的部分变成-100(这一部分好像是对特殊任务集使用)
        targets = torch.ones((max_decoder_length,), dtype=torch.long) * -100
        targets[:length-1] = torch.tensor(input[1:]).long()

        # label tensor化
        if type(self) == ADD_Dataset:
            labels = str(label)
        else:
            labels = torch.tensor(label, dtype=torch.long)

        # 初始化index，初始化一个新tensor，使原来输入的最后一位（有内容）为1，而其他为0.
        # 由于时序性的原因，后面的token对你关心的token没有影响。因此默认总是输出最后一个。
        index = torch.zeros((max_decoder_length,), dtype=torch.int32)
        index[length - 1] = 1
        if solver:
            self.data.append({
                "input_ids": input_tokens.cuda(),
                "input_length": input_length.cuda(),
                "targets": targets.cuda(),
                "index": index.cuda(),
                "labels": labels.cuda(),
                "solver": solver,
            })
        elif rationales:
            self.data.append({
                "input_ids": input_tokens.cuda(),
                "input_length": input_length.cuda(),
                "targets": targets.cuda(),
                "index": index.cuda(),
                "labels": labels.cuda(),
                "rationales":rationales,
            })
        else:
            self.data.append({
                "input_ids": input_tokens.cuda(),
                "input_length": input_length.cuda(),
                "targets": targets.cuda(),
                "index": index.cuda(),
                "labels": labels.cuda() if type(self) != ADD_Dataset else labels,
            })

    def read_data(self, dataset, path, split, rank, world_size):

        if split == 'test': return
        if split == 'dev': split = 'dev'
        path = f"{path}/{dataset}/{split}.jsonl"
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines):
                yield json.loads(row)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BoolQ_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length) -> None:
        super().__init__()

        for row in self.read_data("BoolQ", path, split, rank, world_size):
            label = 1 if row["answer"]==True else 0
            text_a = row['passage']
            text_b = row['question']

            template = f'{text_a}. {text_b}?'

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]



class ADD_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length, demo_path=None, shuffle_option=None):

        super().__init__()

        # 读入所有数据，每个row是一个dict
        for row in self.read_data("ADD", path, split, rank, world_size):

            text_a = row['input']
            template = f'Input:\n{text_a}\nTarget:\n'

            label = row['answerKey'] if 'answerKey' in row else None

            case_num = 7
            if is_json_demo(demo_path):
                if demo_path:
                    with open(demo_path) as f:
                        demo = ""
                        lines = f.readlines()
                        for line in random.sample(lines, case_num):
                            datum = json.loads(line)
                            demo += f'Input:\n{datum["input"]}\nTarget:'
                            demo += f'\n{datum["rationale"]}</scratch>'
                            demo += f'\n{datum["answerKey"]}\n'

            else:
                with open(demo_path) as f:
                    demo = f.read()

                    template = demo + template

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):

        # 这里不需要verbalizer
        return []


class CQA_Dataset(SuperGLUE):

    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length, demo_path=None,avoid_r=None,  shuffle_option=False):
        super().__init__()

        for row in self.read_data("CQA", path, split, rank, world_size):


            ###############  title  ###############
            text_a = 'Q: ' + row['question']['stem']
            text_b = '\nAnswer Choices:'
            ###############  title  ###############


            ##############################  label  ##############################
            letter2int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            label = letter2int[row['answerKey']]  # 0 for 'A', 1 for 'B'...or 4 for 'E'
            ##############################  label  ##############################


            ##############################  options  ##############################
            text_c = ""
            choices = row["question"]["choices"]
            if shuffle_option:
                random.shuffle(choices)  # 旭爷技巧开启
            for choice in choices:
                text_c += f'\n({choice["label"].lower()}) {choice["text"]}'
            text_d = "\nA:"
            ##############################  options  ##############################


            ##############################  shuffle  ##############################
            if shuffle_option:
                options = text_c.split('\n')[1:]
                for idx in range(len(options)):
                    if options[idx].find('('+row["answerKey"].lower()+')') != -1: label = idx
                    letter = {0:'a', 1:'b',2:'c',3:'d',4:'e'}[idx]
                    options[idx] = '\n'+f'({letter})'+options[idx][3:]
                text_c = ""
                for option in options: text_c += option
            ##############################  shuffle  ##############################


            template = f'{text_a}{text_b}{text_c}{text_d}'


            ##############################  demo  ##############################
            if demo_path:

                case_num = 7

                # json
                if is_json_demo(demo_path):
                    with open(demo_path) as f:
                        demo = ""
                        lines = f.readlines()
                        for line in random.sample(lines, case_num):
                            datum = json.loads(line)
                            demo += 'Q: ' + datum['question']['stem']
                            demo += '\nAnswer Choices:'
                            choices = datum["question"]["choices"]
                            for idx, choice in enumerate(choices):
                                demo += f'\n({choice["label"].lower()}) {choice["text"]}'
                            demo += f'\nA: {datum["rationale"]}\n\n'

                # txt
                else:
                    with open(demo_path if split.find("hinted") == -1 else demo_path + "_hinted") as f:
                        demo = f.read()

                if shuffle_option:
                    demo = option_shuffle(demo)

                if avoid_r:
                    blocks = demo.split('\n\n')[0:7]
                    blocks = [ex for ex in blocks if ex.find(row['question']['stem']) == -1]
                    demo = ""
                    for ex in blocks: demo += ex + "\n\n"



                template = demo + template
            ##############################  demo  ##############################


            #############################################  output  #############################################
            if "solver" in row.keys():
                self.make_input(tokenizer, template, max_decoder_length, label, solver=row["solver"])
            elif "rationale" in row.keys():
                row["rationale"] = row["rationale"][:-3] + {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}[label] + row["rationale"][-2:]
                cot = row["rationale"]
                self.make_input(tokenizer, template, max_decoder_length, label, rationales=cot)
            else:
                self.make_input(tokenizer, template, max_decoder_length, label)
            #############################################  output  #############################################

    @classmethod
    def get_verbalizer(cls, tokenizer):

        # 这里的0不是说idx的第一个，而是说去idx，因为它还会返回attn_mask和别的东西
        return [tokenizer.encode(" a")[0],
                tokenizer.encode(" b")[0],
                tokenizer.encode(" c")[0],
                tokenizer.encode(" d")[0],
                tokenizer.encode(" e")[0]]
class ASDiv_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()
        for row in self.read_data("ASDiv", path, split, rank, world_size):

            template = "{BODY} {Question} The answer is"
            template = template.replace("{BODY}", row['Body'])
            template = template.replace("{Question}", row['Question'])
            ###############  title  ###############

            rationales = row["Answer"]
            #############################################  output  #############################################
            self.make_input(tokenizer, template, max_encoder_length, label='None', rationales=rationales)
            #############################################  output  #############################################

class ASDiv_CoT_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()
        for row in self.read_data("ASDiv_CoT", path, split, rank, world_size):
            template = """Your task is to give answers to a Calculator API to a piece of text and a math question. Here are some examples of API calls:

Example 1:
Text: At a restaurant each adult meal costs $8 and kids eat free. If a group of 11 people came in and 2 were kids,
Question: how much would it cost for the group to eat?
Answer: (11-2*8)=72
[sep]

Example 2:
Text: A waiter had 47 customers to wait on. If 41 customers left and he got another 20 customers,
Question: how many customers would he have?
Answer: 47-41+20=26
[sep]

Example 3:
Text: Adam had 8 boxes of toys. Each box had 6 toys. Later Adam bought 5 more toys.
Question: How many toys did he have total?
Answer: 8*6+5=53
[sep]

Example 4:
Text: A pet store had 2 white cats, 10 black cats and 3 gray cats.
Question: How many cats did they have total? 
Answer: 2+10+3=15
[sep]

Test Example:
Text: {BODY}
Question: {QUESTION}
Answer:"""
            template = template.replace("{BODY}", row['Body'])
            template = template.replace("{QUESTION}", row['Question'])
            ###############  title  ###############

            rationales = row["Formula"]
            #############################################  output  #############################################
            self.make_input(tokenizer, template, max_encoder_length, label='None', rationales=rationales)
            #############################################  output  #############################################

class ASDiv_Tool_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        super().__init__()
        for row in self.read_data("ASDiv_CoT", path, split, rank, world_size):
            template = """Your task is to add calls to a Calculator API to a piece of text and a math question. The calls should help you answer the math question. You can call the API by writing "[Calculator(expression)]". Here are some examples of API calls:

Example 1:
Text: At a restaurant each adult meal costs $8 and kids eat free. If a group of 11 people came in and 2 were kids,
Question: how much would it cost for the group to eat?
API call: [Calculator((11-2)*8)]

Example 2:
Text: A waiter had 47 customers to wait on. If 41 customers left and he got another 20 customers,
Question: how many customers would he have?
API call: [Calculator(47-41+20)]

Example 3:
Text: Adam had 8 boxes of toys. Each box had 6 toys. Later Adam bought 5 more toys.
Question: How many toys did he have total?
API call: [Calculator(8*6+5)]

Example 4:
Text: A pet store had 2 white cats, 10 black cats and 3 gray cats.
Question: How many cats did they have total? 
API call: [Calculator(2+10+3)]

Test Example:
Text: {BODY}
Question: {QUESTION}
API call:"""
            template = template.replace("{BODY}", row['Body'])
            template = template.replace("{QUESTION}", row['Question'])
            ###############  title  ###############

            rationales = row["APIcall"]

            #############################################  output  #############################################
            self.make_input(tokenizer, template, max_encoder_length, label='None', rationales=rationales)
            #############################################  output  #############################################



class TLDR_Dataset(SuperGLUE):

    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length, demo_path=None,avoid_r=None,  shuffle_option=False):
        super().__init__()

        for row in self.read_data("TLDR", path, split, rank, world_size):

            ###############  title  ###############
            text_a = 'Please write a summary of the following post.\n'
            text_b = f'Title\n{row["title"]}\n'
            text_c = f'Post\n{row["post"]}\n'
            text_d = f'Summary\n'

            ###############  title  ###############
            template = f'{text_a}{text_b}{text_c}{text_d}'
            ###############  title  ###############

            label = row['label']

            #############################################  output  #############################################
            self.make_input(tokenizer, template, max_decoder_length, label=-1, rationales=label)
            #############################################  output  #############################################

    @classmethod
    def get_verbalizer(cls, tokenizer):

        # 这里的0不是说idx的第一个，而是说去idx，因为它还会返回attn_mask和别的东西
        return [tokenizer.encode(" a")[0],
                tokenizer.encode(" b")[0],
                tokenizer.encode(" c")[0],
                tokenizer.encode(" d")[0],
                tokenizer.encode(" e")[0]]

class TLDRC_Dataset(SuperGLUE):

    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length, demo_path=None,avoid_r=None,  shuffle_option=False):
        super().__init__()

        for row in self.read_data("TLDRC", path, split, rank, world_size):

            ###############  title  ###############
            text_a = 'Please give a score of the following summary of a post.\n'
            text_b = f'Title\n{row["info"]["title"]}\n'
            text_c = f'Post\n{row["info"]["post"]}\n'
            text_d = f'Summary\n'
            ###############  title  ###############
            template = f'{text_a}{text_b}{text_c}{text_d}'
            ###############  title  ###############

            label = row['choice']
            rationales = [row["summaries"][0]["text"], row["summaries"][1]["text"]]

            #############################################  output  #############################################
            self.make_input(tokenizer, template, max_decoder_length, label=label, rationales=rationales)
            #############################################  output  #############################################

    @classmethod
    def get_verbalizer(cls, tokenizer):

        # 这里的0不是说idx的第一个，而是说去idx，因为它还会返回attn_mask和别的东西
        return [tokenizer.encode(" a")[0],
                tokenizer.encode(" b")[0],
                tokenizer.encode(" c")[0],
                tokenizer.encode(" d")[0],
                tokenizer.encode(" e")[0]]

class CB_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("CB", path, split, rank, world_size):
            if row["label"] =="contradiction":
                label = 0
            elif row["label"]=="entailment":
                label = 1
            else:
                label = 2
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2?'

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0], tokenizer.encode(" Maybe")[0]]
    

class COPA_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("COPA", path, split, rank, world_size):
            label = row["label"]
            text = row["premise"]
            choice_1 = row["choice1"]
            choice_2 = row["choice2"]
            question = row["question"]

            template = f'Choice 1: {choice_1} Choice 2: {choice_2} The {question} of "{text}" was choice'

            self.make_input(tokenizer, template, max_decoder_length, label)

            if split == 'train': # mirror
                label = label ^ 1
                template = f'Choice 1: {choice_2} Choice 2: {choice_1} The {question} of "{text}" was choice'

                self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" 1")[0], tokenizer.encode(" 2")[0]]


class MultiRC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()
        self.qids = []

        for template, label, qid in self.read_data("MultiRC", path, split, rank, world_size):
            self.make_input(tokenizer, template, max_decoder_length, label)
            self.qids.append(qid)

    def read_data(self, dataset, path, split, rank, world_size):
        if split == 'test': return
        if split == 'dev': split = 'val'
        path = f"{path}/{dataset}/{split}.jsonl"
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            max_id = (len(lines)) // world_size * world_size
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1

        max_id = (cnt) // world_size * world_size
        cnt = 0
        with open(path, encoding='utf8') as f:
            lines = f.readlines()
            for i, row in enumerate(lines[:max_id]):
                row = json.loads(row)
                text = row["passage"]["text"]

                for question_json in row["passage"]["questions"]:
                    question = question_json["question"]
                    for answer_json in question_json["answers"]:
                        cnt += 1
                        if cnt > max_id: break
                        if cnt % world_size != rank: continue
                        answer = answer_json["text"]
                        label = answer_json["label"]

                        template = f'{text} Is answer "{answer}" the answer to the question "{question}"?'

                        qid = f'{row["idx"]}-{question_json["idx"]}'

                        yield (template, label, qid)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class ReCoRD_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("ReCoRD", path, split, rank, world_size):
            label = row["idx"]
            text = row["passage"]["text"]
            
            entities = []
            for entity_json in row['passage']['entities']:
                start = entity_json['start']
                end = entity_json['end']
                entity = text[start:end+1]
                entities.append(entity)

            text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations

            for question_json in row["qas"]:
                question = question_json["query"]
                answers = []
                for answer_json in question_json["answers"]:
                    answer = answer_json["text"]
                    answers.append(answer)

                template = f'{text} Question: {question} Entities: {entities} Which entities can be filled in the placeholder?'

                self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class RTE_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("RTE", path, split, rank, world_size):
            label = 0 if row["label"]=="not_entailment" else 1
            text_a = row["premise"]
            text_b = row["hypothesis"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does sentence 1 entails sentence 2?'

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class WiC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("WiC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text_a = row["sentence1"]
            text_b = row["sentence2"]
            word = row["word"]

            template = f'Sentence 1: {text_a} Sentence 2: {text_b} Does the word {word} in sentence 1 express the same meaning as in sentence 2?'

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]


class WSC_Dataset(SuperGLUE):
    def __init__(self, path, split, rank, world_size, tokenizer, max_decoder_length):
        super().__init__()

        for row in self.read_data("WSC", path, split, rank, world_size):
            label = 1 if row["label"]==True else 0
            text = row["text"]
            
            span_1 = row["target"]["span1_text"]
            span_2 = row["target"]["span2_text"]

            template = f'{text} Does {span_2} refers to {span_1}?'

            self.make_input(tokenizer, template, max_decoder_length, label)

    @classmethod
    def get_verbalizer(cls, tokenizer):
        return [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]
