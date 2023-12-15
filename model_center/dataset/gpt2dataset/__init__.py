from .superglue import *

DATASET = {
    "BoolQ": BoolQ_Dataset,
    "CB": CB_Dataset,
    "COPA": COPA_Dataset,
    "MultiRC": MultiRC_Dataset,
    "ReCoRD": ReCoRD_Dataset,
    "RTE": RTE_Dataset,
    "WiC": WiC_Dataset,
    "WSC": WSC_Dataset,
    "CQA": CQA_Dataset,
    "ADD": ADD_Dataset,
    "TLDR": TLDR_Dataset,
    "TLDRC": TLDRC_Dataset,
    "ASDiv": ASDiv_Dataset,
    "ASDiv_CoT": ASDiv_CoT_Dataset,
    "ASDiv_Tool": ASDiv_Tool_Dataset
}