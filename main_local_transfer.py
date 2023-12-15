import os
import torch
import bmtrain as bmt
from model_center.dataset import DistributedDataLoader
from model_center.dataset.gpt2dataset import DATASET
from utils import initialize, setup_model_and_optimizer, gptj_top_k_generate, gptj_teacher_forcing


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ["train", 'dev']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer,
                                               args.max_encoder_length, args.max_decoder_length)
    return dataset


def evaluate(args, tokenizer, model, dataset):
    print("---- evalutation ----")
    model.eval()
    dataloader = {"dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=False)}
    with torch.no_grad():
        for split in ['dev']:
            for it, data in enumerate(dataloader[split]):

                input_length = data["input_length"]
                probe_length = args.decode_length
                input_ids = gptj_top_k_generate(data, probe_length, model, 1, )  # tokenizer.decode(input_ids[0])
                targets = data["labels"] + 64  # [0, 1, 2, 3, 4] -> [64, 65, 66, 67, 68]
                with open(f"{args.save}/eval_output.txt", 'w') as f:
                    for index in range(targets.shape[0]):
                        print(f"{tokenizer.decode(input_ids[index][:input_length[index]])}", file=f)
                        print(
                            tokenizer.decode(input_ids[index][input_length[index]: input_length[index] + probe_length]),
                            file=f)


def finetune(args, tokenizer, model, optim_manager, model_optimizer, model_lr_scheduler):
    model.train()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    for epoch in range(args.epochs):
        dataset = prepare_dataset(args, tokenizer, f"{args.base_path}/data", args.dataset_name, bmt.rank(),
                                  bmt.world_size())
        dataloader = {"train": DistributedDataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True),
                      "dev": DistributedDataLoader(dataset['dev'], batch_size=args.batch_size, shuffle=True)}
        for split in ['train']:
            for it, data in enumerate(dataloader[split]):
                input_ids = data["input_ids"]
                input_length = data["input_length"]
                index = data["index"]
                targets = data["rationales"]

                loss = gptj_teacher_forcing(model, input_ids, input_length, index, targets, tokenizer, loss_func)
                optim_manager.zero_grad()
                optim_manager.backward(loss)

                model_grad_norm = optim_manager.clip_grad_norm(model_optimizer.param_groups, args.clip_grad,
                                                               norm_type=2)
                optim_manager.step()

                bmt.print_rank(
                    "epoch {:3d} | Iter: {:6d}/{:6d} | model_lr: {:.4e} | model_grad_norm: {:.4f} | loss: {:.4f}".format(
                        epoch,
                        it,
                        len(dataloader[f"{split}"]),
                        model_lr_scheduler.current_lr,
                        model_grad_norm,
                        loss
                    )
                )

        bmt.synchronize()
        if (epoch + 1) % args.eval_epoch_gap == 0:
            evaluate(args, tokenizer, model, dataset)
        if epoch % args.save_epoch == 0:
            bmt.save(model, os.path.join(args.save, f"epoch_{epoch}.pt"))


def main():
    args = initialize()
    print(args)

    tokenizer, model, model_optimizer, model_lr_scheduler = setup_model_and_optimizer(args)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale)
    optim_manager.add_optimizer(model_optimizer, model_lr_scheduler)

    finetune(args, tokenizer, model, optim_manager, model_optimizer, model_lr_scheduler)
    dataset = prepare_dataset(args, tokenizer, f"{args.base_path}/data", args.dataset_name, bmt.rank(),
                              bmt.world_size())
    evaluate(args, tokenizer, model, dataset)


if __name__ == "__main__":
    main()
