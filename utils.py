import os
import torch
import bmtrain as bmt

from model_center import get_args
from model_center.model import GPTj
from model_center.tokenizer import GPTjTokenizer


def get_tokenizer(args):
    tokenizer = GPTjTokenizer.from_pretrained(args.model_config)
    return tokenizer


def get_model(args):
    model = GPTj.from_pretrained(args.model_config)
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=args.weight_decay, )
    return optimizer


def get_learning_rate_scheduler(args, optimizer, module_type=None):
    assert module_type in (None, 'backbone', 'plugin')
    the_lr = {'backbone': args.backbone_lr, 'plugin': args.plugin_lr}
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer,
                                             start_lr=the_lr[module_type] if module_type else args.lr,
                                             warmup_iter=args.warmup_iters,
                                             end_iter=args.lr_decay_iters,
                                             num_iter=args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer,
                                                start_lr=the_lr[module_type] if module_type else args.lr,
                                                warmup_iter=args.warmup_iters,
                                                end_iter=-1,
                                                num_iter=args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer,
                                               start_lr=the_lr[module_type] if module_type else args.lr,
                                               warmup_iter=args.warmup_iters,
                                               end_iter=args.lr_decay_iters,
                                               num_iter=args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer,
                                                    start_lr=the_lr[module_type] if module_type else args.lr,
                                                    warmup_iter=args.warmup_iters,
                                                    end_iter=args.lr_decay_iters,
                                                    num_iter=args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer,
                                               start_lr=the_lr[module_type] if module_type else args.lr,
                                               warmup_iter=args.warmup_iters,
                                               end_iter=args.lr_decay_iters,
                                               num_iter=args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def initialize():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    bmt.init_distributed(seed=args.seed, )
    os.makedirs(args.save, exist_ok=True)
    return args


def setup_model_and_optimizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer, module_type="backbone")
    bmt.synchronize()
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler


def update_ids_length_index(input_ids, input_length, index, rationale_ids: list):
    assert input_ids.shape[0] == len(rationale_ids)

    # take each example in the batch seperately
    for idx in range(len(rationale_ids)):
        suffix_id = rationale_ids[idx]  # [343, 454, 333, ..., 234]
        length = len(suffix_id)  # length = len(suffix_id)

        index[idx][input_length[idx]: input_length[
                                          idx] + length - 1] = 1  # index[0][input_length[0]: input_length[0] + length -1] = 1
        input_ids[idx][input_length[idx]: input_length[idx] + length] = torch.tensor(
            suffix_id)  # input_ids[0][input_length[0]: input_length[0] + length] = torch.tensor(suffix_id)
        input_length[idx] += length  # length = 643(original length) + 32 = 675


def teacher_forcing(model, input_ids, input_length, index, rationales, tokenizer, loss_func):
    rationale_ids = tokenizer(rationales)['input_ids']
    update_ids_length_index(input_ids, input_length, index, rationale_ids)

    logits = model(input_ids, input_length, return_logits=True)[0]
    logits = logits[torch.where(index == 1)]

    for idx in range(1, len(rationale_ids)): rationale_ids[0] += rationale_ids[idx]
    targets = torch.tensor(rationale_ids[0]).cuda()
    loss = loss_func(logits, targets)

    return loss


def gptj_add_suffix_prompt(input_ids, input_length, index, suffix_id, ):
    length = len(suffix_id)
    input_length += length

    loc = torch.nonzero(index)
    loc = torch.t(loc)

    suffixed_ids = input_ids.clone()

    for idx in range(len(suffix_id)):
        suffixed_ids[loc[0], loc[1] + 1 + idx] = suffix_id[idx]

    index[loc[0], loc[1]] = 0
    index[loc[0], loc[1] + length] = 1

    return suffixed_ids


def gptj_update_length_index(input_length, index):
    input_length += 1
    loc = torch.nonzero(index)
    loc = torch.t(loc)
    index[loc[0], loc[1]] = 0
    index[loc[0], loc[1] + 1] = 1


def initialize():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    bmt.init_distributed(seed=args.seed, )
    os.makedirs(args.save, exist_ok=True)
    return args


def setup_model_and_optimizer_alone(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer, module_type="backbone")
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    return tokenizer, model, optimizer, lr_scheduler


def setup_model_and_optimizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer, module_type="backbone")
    bmt.synchronize()
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler


def gptj_top_k_generate(data, decode_length, model, top_k, suffix_id=None):
    input_ids = data["input_ids"].clone()  # tokenizer.decode(data["input_ids"][0])
    input_length = data["input_length"].clone()
    index = data["index"].clone()
    soft_max = torch.nn.Softmax(dim=-1)

    if suffix_id is not None:
        suffixed_input_ids = gptj_add_suffix_prompt(input_ids, input_length, index, suffix_id)
    else:
        suffixed_input_ids = input_ids

    for decode_length in range(decode_length):
        logits = model(suffixed_input_ids, input_length, return_logits=True)[0]

        input_length += 1

        loc = torch.nonzero(index)
        loc = torch.t(loc)

        logits = logits[torch.where(index == 1)]

        top_logits, top_index = torch.topk(logits, top_k, -1, sorted=True)
        top_distribution = soft_max(top_logits)
        chosen_index = top_distribution.multinomial(num_samples=1, replacement=True).squeeze()
        logits = top_index[loc[0], chosen_index]

        suffixed_input_ids[loc[0], loc[1] + 1] = logits.int()
        index[loc[0], loc[1]] = 0
        index[loc[0], loc[1] + 1] = 1

    return suffixed_input_ids

def t5_top_k_generate(data, decode_length, model, top_k,):

    enc_input = data["enc_input"]
    enc_length = data["enc_length"]
    dec_input = data["dec_input"]
    dec_length = data["dec_length"]
    index = data["index"]

    soft_max = torch.nn.Softmax(dim=-1)

    for decode_length in range(decode_length):
        res = model(enc_input, enc_length, dec_input, dec_length, output_logits=True)
        logits = res[2][torch.where(index == 1)]
        dec_length += 1
        loc = torch.nonzero(index)
        loc = torch.t(loc)

        top_logits, top_index = torch.topk(logits, top_k, -1, sorted=True)
        top_distribution = soft_max(top_logits)
        chosen_index = top_distribution.multinomial(num_samples=1, replacement=True).squeeze()
        logits = top_index[loc[0], chosen_index]

        dec_input[loc[0], loc[1] + 1] = logits.int()
        index[loc[0], loc[1]] = 0
        index[loc[0], loc[1] + 1] = 1

    return dec_input


def gptj_greedy_generate(the_input_ids, the_input_length, the_index, decode_length, model, top_k=1):
    input_ids = the_input_ids.clone()
    input_length = the_input_length.clone()
    index = the_index.clone()
    soft_max = torch.nn.Softmax(dim=-1)
    for decode_length in range(decode_length):
        logits = model(input_ids, input_length, return_logits=True)[0]

        input_length += 1

        loc = torch.nonzero(index)
        loc = torch.t(loc)

        logits = logits[torch.where(index == 1)]

        top_logits, top_index = torch.topk(logits, top_k, -1, sorted=True)
        top_distribution = soft_max(top_logits)
        chosen_index = top_distribution.multinomial(num_samples=1, replacement=True).squeeze()
        logits = top_index[loc[0], chosen_index]

        input_ids[loc[0], loc[1] + 1] = logits.int()
        index[loc[0], loc[1]] = 0
        index[loc[0], loc[1] + 1] = 1

    return input_ids


def gptj_update_ids_length_index(input_ids, input_length, index, rationale_ids: list):
    assert input_ids.shape[0] == len(rationale_ids)

    # take each example in the batch seperately
    for idx in range(len(rationale_ids)):
        suffix_id = rationale_ids[idx]  # [343, 454, 333, ..., 234]
        length = len(suffix_id)  # length = len(suffix_id)

        index[idx][input_length[idx]: input_length[
                                          idx] + length - 1] = 1  # index[0][input_length[0]: input_length[0] + length -1] = 1
        input_ids[idx][input_length[idx]: input_length[idx] + length] = torch.tensor(
            suffix_id)  # input_ids[0][input_length[0]: input_length[0] + length] = torch.tensor(suffix_id)
        input_length[idx] += length  # length = 643(original length) + 32 = 675


def gptj_teacher_forcing(model, input_ids, input_length, index, rationales, tokenizer, loss_func):
    rationale_ids = tokenizer(rationales)['input_ids']
    gptj_update_ids_length_index(input_ids, input_length, index, rationale_ids)

    logits = model(input_ids, input_length, return_logits=True)[0]
    logits = logits[torch.where(index == 1)]

    for idx in range(1, len(rationale_ids)): rationale_ids[0] += rationale_ids[idx]
    targets = torch.tensor(rationale_ids[0]).cuda()
    loss = loss_func(logits, targets)

    return loss
