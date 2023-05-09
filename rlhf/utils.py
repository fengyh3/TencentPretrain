import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tencentpretrain.model_builder import build_model
from tencentpretrain.model_loader import load_model, _load_state_dict_into_model
from tencentpretrain.utils import *
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.decoders import *
from tencentpretrain.targets import *
from rlhf.model import Model
import sys
import json
from argparse import Namespace


def load_hyperparam(default_args, config_path):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    default_args_dict = vars(default_args)

    command_line_args_dict = {k: default_args_dict[k] for k in [
        a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)
    ]}
    default_args_dict.update(config_args_dict)
    default_args_dict.update(command_line_args_dict)
    args = Namespace(**default_args_dict)

    return args


def build_model(args, target_name):
    embedding = Embedding(args)
    for embedding_name in args.embedding:
        tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
        embedding.update(tmp_emb, embedding_name)

    encoder = str2encoder[args.encoder](args)

    if args.decoder is not None:
        if args.data_processor == "mt":
            tgt_vocab_size = len(args.tgt_tokenizer.vocab)
        else:
            tgt_vocab_size = len(args.tokenizer.vocab)

        tgt_embedding = Embedding(args)
        for embedding_name in args.tgt_embedding:
            tmp_emb = str2embedding[embedding_name](args, tgt_vocab_size)
            tgt_embedding.update(tmp_emb, embedding_name)

        decoder = str2decoder[args.decoder](args)
    else:
        tgt_embedding = None
        decoder = None

    target = Target()
    for tn in target_name:
        tmp_target = str2target[tn](args, len(args.tokenizer.vocab))
        target.update(tmp_target, tn)

    model = Model(args, target_name, embedding, encoder, tgt_embedding, decoder, target)

    return model


def build_and_load(args, target_name, deepspeed_config, pretrained_model_path, lora_pretrained_model_path=None):
    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=deepspeed_config):
            model_for_training = build_model(args, target_name)
    else:
        model_for_training = build_model(args, target_name)

    # Load or initialize parameters.
    if args.enable_zero3:
        model_for_training = _load_state_dict_into_model(model_for_training, pretrained_model_path, "",
                                                         lora_pretrained_model_path)
    else:
        model_for_training = load_model(model_for_training, pretrained_model_path,
                                        lora_pretrained_model_path)

    return model_for_training


def get_optimizer_grouped_parameters(args, model):
    param_optimizer = list(model.named_parameters())
    if args.use_lora:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if 'lora' in n]}
        ]
        for n, p in list(model.named_parameters()):
            if 'lora' not in n:
                p.requires_grad = False
    else:
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

    return optimizer_grouped_parameters


def get_scheduler(args, optimizer):
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup)
    elif args.scheduler in ["tri_stage"]:
        custom_scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup,
                                                         args.total_steps * args.decay, args.total_steps)
    else:
        custom_scheduler = str2scheduler[args.scheduler](optimizer, args.total_steps * args.warmup, args.total_steps)

    return custom_scheduler


def _zero3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def moving_average(actor, ema, beta=0.992, device=None, zero_stage_3=False):
    with torch.no_grad():
        for param, param_ema in zip(actor.parameters(), ema.parameters()):
            params_to_fetch = _zero3_params_to_fetch([param, param_ema]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
