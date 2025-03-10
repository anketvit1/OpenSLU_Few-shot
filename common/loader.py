'''
Author: Qiguang Chen
Date: 2023-01-11 10:39:26
LastEditors: Qiguang Chen
LastEditTime: 2023-04-08 13:08:40
Description: all class for load data.

'''
import os
import torch
import json
from datasets import load_dataset, Dataset
#dataset = load_dataset("LightChen2333/OpenSLU",'atis', trust_remote_code=True)
from torch.utils.data import DataLoader
import random
from common.utils import InputData
import numpy as np
#from datasets import load_from_disk

# Specify the path to your dataset directory
#dataset_path = "/home/kumara/JointBERT/data/atis"

# Load the dataset from the specified directory
#dataset = load_from_disk(dataset_path)

ABS_PATH=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

import random
import json
from collections import defaultdict
from datasets import Dataset, load_dataset

class DataFactory(object):
    def __init__(self, tokenizer, use_multi_intent=False, to_lower_case=True):
        """_summary_

        Args:
            tokenizer (Tokenizer): _description_
            use_multi_intent (bool, optional): _description_. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.use_multi = use_multi_intent
        self.to_lower_case = to_lower_case
        self.slot_label_dict = None
        self.intent_label_dict = None
        self.slot_label_list = []
        self.intent_label_list = []

    def __is_supported_datasets(self, dataset_name: str) -> bool:
        return dataset_name.lower() in ["atis", "snips", "mix-atis", "mix-snips"]

    def select_few_shot_samplesold(self, dataset, num_samples_per_class=5):
        """
        Selects a few-shot subset by picking 'num_samples_per_class' examples per intent.
        """
        class_samples = defaultdict(list)

        # Group samples by intent label
        for idx, intent in enumerate(dataset["intent"]):  # Assuming 'intent' is the label
            class_samples[intent].append(idx)

        # Select 'num_samples_per_class' randomly from each class
        selected_indices = []
        for intent, indices in class_samples.items():
            selected_indices.extend(random.sample(indices, min(num_samples_per_class, len(indices))))

        return dataset.select(selected_indices)

    def select_few_shot_samples(self, dataset, num_samples_per_class=5):
        """
        Selects a few-shot subset by picking 'num_samples_per_class' examples per intent.
        Also prints the selected samples for debugging.
        """
        class_samples = defaultdict(list)

        # Group samples by intent label
        for idx, intent in enumerate(dataset["intent"]):  # Assuming 'intent' is the label
            class_samples[intent].append(idx)

        # Select 'num_samples_per_class' randomly from each class
        selected_indices = []
        print("\n==== Few-Shot Sample Selection ====")
        for intent, indices in class_samples.items():
            selected_count = min(num_samples_per_class, len(indices))
            selected = random.sample(indices, selected_count)
            selected_indices.extend(selected)

            # Print intent and selected sample count
            print(f"Intent: {intent} | Total samples: {len(indices)} | Selected: {selected_count}")
            # Print actual selected sample texts
            #for idx in selected:
            #    print(f"  → {dataset[idx]['text']}")

        print("====================================\n")
        return dataset.select(selected_indices)

    def load_dataset(self, dataset_config, split="train", few_shot=False, num_shot=5):
        dataset_name = None
        if split not in dataset_config:
            dataset_name = dataset_config.get("dataset_name")
        elif self.__is_supported_datasets(dataset_config[split]):
            dataset_name = dataset_config[split].lower()

        if dataset_name is not None:
            dataset = load_dataset("LightChen2333/OpenSLU", dataset_name, split=split)

            # Shuffle only for training
            if split == "train":
                dataset = dataset.shuffle()

            # Apply few-shot sampling if enabled
            if few_shot:
                dataset = self.select_few_shot_samples(dataset, num_shot)
                print(f"Few-shot dataset size: {len(dataset)}")

            print(f"Number of samples in {split} dataset: {len(dataset)}")
            return dataset
        else:
            # File-based dataset loading
            data_file = dataset_config[split]
            data_dict = {"text": [], "slot": [], "intent": []}

            with open(data_file, encoding="utf-8") as f:
                lines = f.readlines()

            if split == "train":
                selected_lines = random.sample(lines, min(len(lines), 1000))  # Select 1000 samples max
            else:
                selected_lines = lines

            for line in selected_lines:
                row = json.loads(line)
                if len(row["text"]) == len(row["slot"]):
                    data_dict["text"].append(row["text"])
                    data_dict["slot"].append(row["slot"])
                    data_dict["intent"].append(row["intent"])
                else:
                    print("Error Input:", row)

            dataset = Dataset.from_dict(data_dict)

            # Apply few-shot selection
            if few_shot:
                dataset = self.select_few_shot_samples(dataset, num_shot)
                print(f"Few-shot dataset size: {len(dataset)}")

            return dataset

    def load_dataset_old(self, dataset_config, split="train", fraction=0.1):
        #print("Loading dataset with fraction:", fraction)
        dataset_name = None
        if split not in dataset_config:
            dataset_name = dataset_config.get("dataset_name")
        elif self.__is_supported_datasets(dataset_config[split]):
            dataset_name = dataset_config[split].lower()
        if dataset_name is not None:
            #return load_dataset("LightChen2333/OpenSLU", dataset_name, split=split, fraction=fraction)
            if split == "train":
                dataset = load_dataset("LightChen2333/OpenSLU", dataset_name, split=split, fraction=fraction)
                # Shuffle the dataset
                dataset = dataset.shuffle()
        
                # Select a fraction of the dataset if specified
                #if fraction is not None:
                num_samples = int(len(dataset) * 0.03)
                dataset = dataset.select(range(num_samples))
                print(f"Loading {num_samples} samples") 
                print(f"Number of samples in {split} dataset: {len(dataset)}")
                return dataset
            elif split == "validation":
                dataset = load_dataset("LightChen2333/OpenSLU", dataset_name, split=split, fraction=None)
                print(f"Number of samples in {split} dataset: {len(dataset)}")
                return dataset
            else:
                dataset = load_dataset("LightChen2333/OpenSLU", dataset_name, split=split, fraction=None)
                print(f"Number of samples in {split} dataset: {len(dataset)}")  # Print number of samples
                return dataset
                #return load_dataset("LightChen2333/OpenSLU", dataset_name, split=split, fraction=fraction)
        else:
            data_file = dataset_config[split]
            data_dict = {"text": [], "slot": [], "intent": []}
            with open(data_file, encoding="utf-8") as f:
                lines = f.readlines()
                if split == "train":
                    num_lines_to_select = int(len(lines) * 0.03)  # Calculate the number of lines to select
                    selected_lines = random.sample(lines, num_lines_to_select)
                    print(f"Loading {num_lines_to_select} samples out of {len(lines)}")
                else:
                    selected_lines = lines
                    print(f"Loading {len(selected_lines)} samples out of {len(lines)}")
                for line in selected_lines:
                    row = json.loads(line)
                    if len(row["text"]) == len(row["slot"]):
                        data_dict["text"].append(row["text"])
                        data_dict["slot"].append(row["slot"])
                        data_dict["intent"].append(row["intent"])
                    else:
                        print("Error Input: ", row)
            return Dataset.from_dict(data_dict)

    #def load_dataset(self, dataset_config, split="train"):
    #    dataset_name = None
    #    if split not in dataset_config:
    #        dataset_name = dataset_config.get("dataset_name")
    #    elif self.__is_supported_datasets(dataset_config[split]):
    #        dataset_name = dataset_config[split].lower()
    #    if dataset_name is not None:
    #        return load_dataset("LightChen2333/OpenSLU", dataset_name, split=split)
    #    else:
    #        data_file = dataset_config[split]
    #        data_dict = {"text": [], "slot": [], "intent":[]}
    #        with open(data_file, encoding="utf-8") as f:
    #            for line in f:
    #                row = json.loads(line)
    #                if len(row["text"]) == len(row["slot"]):
    #                    data_dict["text"].append(row["text"])
    #                    data_dict["slot"].append(row["slot"])
    #                    data_dict["intent"].append(row["intent"])
    #                else:
    #                    print("Error Input: ", row)
    #        return Dataset.from_dict(data_dict)

    def update_label_names(self, dataset, label_path=None):
        if label_path is not None:
            label = json.load(open(label_path,"r", encoding="utf8"))
            if label.get("slot"):
                self.slot_label_dict = {x:i for i,x in enumerate(label["slot"])}
                self.slot_label_list = label["slot"]
            if label.get("intent"):
                self.intent_label_dict = {x:i for i,x in enumerate(label["intent"])}
                self.intent_label_list = label["intent"]
            if label.get("intent") is None and label.get("slot") is None:
                print("Error!!")
                raise ValueError
        else:
            for slot_label in dataset["slot"]:
                for x in slot_label:
                    if x not in self.slot_label_list:
                        self.slot_label_list.append(x)
            self.slot_label_dict = {key: index for index,
                                key in enumerate(self.slot_label_list)}
            
            for intent_labels in dataset["intent"]:
                if self.use_multi:
                    intent_label = intent_labels.split("#")
                else:
                    intent_label = [intent_labels]
                for x in intent_label:
                    if x not in self.intent_label_list:
                        self.intent_label_list.append(x)
            
            self.intent_label_dict = {key: index for index,
                                    key in enumerate(self.intent_label_list)}
        

    def update_vocabulary(self, dataset):
        if self.tokenizer.name_or_path in ["word_tokenizer"]:
            for data in dataset:
                self.tokenizer.add_instance(data["text"])

    @staticmethod
    def fast_align_data(text, padding_side="right"):
        for i in range(len(text.input_ids)):
            desired_output = []
            for word_id in text.word_ids(i):
                if word_id is not None:
                    start, end = text.word_to_tokens(
                        i, word_id, sequence_index=0 if padding_side == "right" else 1)
                    if start == end - 1:
                        tokens = [start]
                    else:
                        tokens = [start, end - 1]
                    if len(desired_output) == 0 or desired_output[-1] != tokens:
                        desired_output.append(tokens)
            yield desired_output

    def fast_align(self,
                   batch,
                   ignore_index=-100,
                   device="cuda",
                   config=None,
                   enable_label=True,
                   label2tensor=True):
        if self.to_lower_case:
            input_list = [[t.lower() for t in x["text"]] for x in batch]
        else:
            input_list = [x["text"] for x in batch]
        text = self.tokenizer(input_list,
                              return_tensors="pt",
                              padding=True,
                              is_split_into_words=True,
                              truncation=True,
                              **config).to(device)
        if enable_label:
            if label2tensor:

                slot_mask = torch.ones_like(text.input_ids) * ignore_index
                for i, offsets in enumerate(
                        DataFactory.fast_align_data(text, padding_side=self.tokenizer.padding_side)):
                    num = 0
                    assert len(offsets) == len(batch[i]["text"])
                    assert len(offsets) == len(batch[i]["slot"])
                    for off in offsets:
                        slot_mask[i][off[0]
                                     ] = self.slot_label_dict[batch[i]["slot"][num]]
                        num += 1
                slot = slot_mask.clone()
                attentin_id = 0 if self.tokenizer.padding_side == "right" else 1
                for i, slot_batch in enumerate(slot):
                    for j, x in enumerate(slot_batch):
                        if x == ignore_index and text.attention_mask[i][j] == attentin_id and (text.input_ids[i][
                                j] not in self.tokenizer.all_special_ids or text.input_ids[i][j] == self.tokenizer.unk_token_id):
                            slot[i][j] = slot[i][j - 1]
                slot = slot.to(device)
                if not self.use_multi:
                    intent = torch.tensor(
                        [self.intent_label_dict[x["intent"]] for x in batch]).to(device)
                else:
                    one_hot = torch.zeros(
                        (len(batch), len(self.intent_label_list)), dtype=torch.float)
                    for index, b in enumerate(batch):
                        for x in b["intent"].split("#"):
                            one_hot[index][self.intent_label_dict[x]] = 1.
                    intent = one_hot.to(device)
            else:
                slot_mask = None
                slot = [['#' for _ in range(text.input_ids.shape[1])]
                        for _ in range(text.input_ids.shape[0])]
                for i, offsets in enumerate(DataFactory.fast_align_data(text)):
                    num = 0
                    for off in offsets:
                        slot[i][off[0]] = batch[i]["slot"][num]
                        num += 1
                if not self.use_multi:
                    intent = [x["intent"] for x in batch]
                else:
                    intent = [
                        [x for x in b["intent"].split("#")] for b in batch]
            return InputData((text, slot, intent))
        else:
            return InputData((text, None, None))

    def general_align_data(self, split_text_list, raw_text_list, encoded_text):
        for i in range(len(split_text_list)):
            desired_output = []
            jdx = 0
            offset = encoded_text.offset_mapping[i].tolist()
            split_texts = split_text_list[i]
            raw_text = raw_text_list[i]
            last = 0
            temp_offset = []
            for off in offset:
                s, e = off
                if len(temp_offset) > 0 and (e != 0 and last == s):
                    len_1 = off[1] - off[0]
                    len_2 = temp_offset[-1][1] - temp_offset[-1][0]
                    if len_1 > len_2:
                        temp_offset.pop(-1)
                        temp_offset.append([0, 0])
                        temp_offset.append(off)
                    continue
                temp_offset.append(off)
                last = s
            offset = temp_offset
            for split_text in split_texts:
                while jdx < len(offset) and offset[jdx][0] == 0 and offset[jdx][1] == 0:
                    jdx += 1
                if jdx == len(offset):
                    continue
                start_, end_ = offset[jdx]
                tokens = None
                if split_text == raw_text[start_:end_].strip():
                    tokens = [jdx]
                else:
                    # Compute "xxx" -> "xx" "#x"
                    temp_jdx = jdx
                    last_str = raw_text[start_:end_].strip()
                    while last_str != split_text and temp_jdx < len(offset) - 1:
                        temp_jdx += 1
                        last_str += raw_text[offset[temp_jdx]
                                             [0]:offset[temp_jdx][1]].strip()

                    if temp_jdx == jdx:
                        if temp_jdx == len(offset) - 1:
                            break
                        else:
                            raise ValueError("Illegal Input data")
                    elif last_str == split_text:
                        tokens = [jdx, temp_jdx]
                        jdx = temp_jdx
                    else:
                        jdx -= 1
                jdx += 1
                if tokens is not None:
                    desired_output.append(tokens)
            yield desired_output

    def general_align(self,
                      batch,
                      ignore_index=-100,
                      device="cuda",
                      config=None,
                      enable_label=True,
                      label2tensor=True,
                      locale="en-US"):
        if self.to_lower_case:
            raw_data = [" ".join(x["text"]).lower() if locale not in ['ja-JP', 'zh-CN', 'zh-TW'] else "".join(x["text"]) for x in
                    batch]
            input_list = [[t.lower() for t in x["text"]] for x in batch]
        else:
            input_list = [x["text"] for x in batch]
            raw_data = [" ".join(x["text"]) if locale not in ['ja-JP', 'zh-CN', 'zh-TW'] else "".join(x["text"]) for x in
                        batch]
        text = self.tokenizer(raw_data,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              return_offsets_mapping=True,
                              **config).to(device)
        if enable_label:
            if label2tensor:
                slot_mask = torch.ones_like(text.input_ids) * ignore_index
                for i, offsets in enumerate(
                        self.general_align_data(input_list, raw_data, encoded_text=text)):
                    num = 0
                    # if len(offsets) != len(batch[i]["text"]) or len(offsets) != len(batch[i]["slot"]):
                    #     if
                    for off in offsets:
                        slot_mask[i][off[0]] = self.slot_label_dict[batch[i]["slot"][num]]
                        num += 1
                # slot = slot_mask.clone()
                # attentin_id = 0 if self.tokenizer.padding_side == "right" else 1
                # for i, slot_batch in enumerate(slot):
                #     for j, x in enumerate(slot_batch):
                #         if x == ignore_index and text.attention_mask[i][j] == attentin_id and text.input_ids[i][
                #             j] not in self.tokenizer.all_special_ids:
                #             slot[i][j] = slot[i][j - 1]
                slot = slot_mask.to(device)
                if not self.use_multi:
                    intent = torch.tensor(
                        [self.intent_label_dict[x["intent"]] for x in batch]).to(device)
                else:
                    one_hot = torch.zeros(
                        (len(batch), len(self.intent_label_list)), dtype=torch.float)
                    for index, b in enumerate(batch):
                        for x in b["intent"].split("#"):
                            one_hot[index][self.intent_label_dict[x]] = 1.
                    intent = one_hot.to(device)
            else:
                slot_mask = None
                slot = [['#' for _ in range(text.input_ids.shape[1])]
                        for _ in range(text.input_ids.shape[0])]
                for i, offsets in enumerate(self.general_align_data(input_list, raw_data, encoded_text=text)):
                    num = 0
                    for off in offsets:
                        slot[i][off[0]] = batch[i]["slot"][num]
                        num += 1
                if not self.use_multi:
                    intent = [x["intent"] for x in batch]
                else:
                    intent = [
                        [x for x in b["intent"].split("#")] for b in batch]
            return InputData((text, slot, intent))
        else:
            return InputData((text, None, None))

    def batch_fn(self,
                 batch,
                 ignore_index=-100,
                 device="cuda",
                 config=None,
                 align_mode="fast",
                 enable_label=True,
                 label2tensor=True):
        if align_mode == "fast":
            # try:
            return self.fast_align(batch,
                                   ignore_index=ignore_index,
                                   device=device,
                                   config=config,
                                   enable_label=enable_label,
                                   label2tensor=label2tensor)
            # except:
            #     return self.general_align(batch,
            #                               ignore_index=ignore_index,
            #                               device=device,
            #                               config=config,
            #                               enable_label=enable_label,
            #                               label2tensor=label2tensor)
        else:
            return self.general_align(batch,
                                      ignore_index=ignore_index,
                                      device=device,
                                      config=config,
                                      enable_label=enable_label,
                                      label2tensor=label2tensor)

    def get_data_loader(self,
                        dataset,
                        batch_size,
                        shuffle=False,
                        device="cuda",
                        enable_label=True,
                        align_mode="fast",
                        label2tensor=True, **config):
        data_loader = DataLoader(dataset,
                                 shuffle=shuffle,
                                 batch_size=batch_size,
                                 collate_fn=lambda x: self.batch_fn(x,
                                                                    device=device,
                                                                    config=config,
                                                                    enable_label=enable_label,
                                                                    align_mode=align_mode,
                                                                    label2tensor=label2tensor))
        return data_loader
