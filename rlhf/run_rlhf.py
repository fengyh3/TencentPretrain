import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tencentpretrain.utils import str2tokenizer, str2dataloader
from tencentpretrain.opts import *
from tencentpretrain.utils.seed import set_seed
from rlhf.RLHFEngine import RLHFEngine
import deepspeed
import torch.distributed as dist
from rlhf.ppo_trainer import PPOTrainer


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--ppo_dataset_path", type=str, default="dataset.pt",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--unsupervised_dataset_path", type=str, default=None,
                        help="Path of the preprocessed dataset.")

    parser.add_argument("--actor_pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--critic_pretrained_model_path", type=str, default=None,
                        help="Path of the pretrained model.")
    parser.add_argument("--actor_output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--critic_output_model_path", type=str, required=True,
                        help="Path of the output model.")
    parser.add_argument("--actor_config_path", type=str, default="models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")
    parser.add_argument("--critic_config_path", type=str, default="models/bert/base_config.json",
                        help="Config file of model hyper-parameters.")

    # Training and saving options.
    parser.add_argument("--total_steps", type=int, default=100000,
                        help="Total training steps.")
    parser.add_argument("--save_checkpoint_steps", type=int, default=10000,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Specific steps to accumulate gradient.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size. The actual batch_size is [batch_size x world_size x accumulation_steps].")
    parser.add_argument("--instances_buffer_size", type=int, default=25600,
                        help="The buffer size of instances in memory.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")

    # loss options
    parser.add_argument("--kl_ctl", type=float, default=0.02, help="kl loss weight.")
    parser.add_argument("--clip_reward_value", type=float, default=5.0, help=".")
    parser.add_argument("--cliprange", type=float, default=0.2, help=".")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help=".")
    parser.add_argument("--gamma", type=float, default=1.0, help=".")
    parser.add_argument("--lam", type=float, default=0.95, help=".")
    parser.add_argument("--unsup_coef", type=float, default=27.8, help=".")

    # Preprocess options.
    tokenizer_opts(parser)

    # Model options.
    model_opts(parser)
    parser.add_argument("--actor_target", choices=["sp", "lm", "mlm", "bilm", "cls", "clr", "rm", "ppo"], default="mlm",
                        nargs='+', help="The training target of the pretraining model.")
    parser.add_argument("--critic_target", choices=["sp", "lm", "mlm", "bilm", "cls", "clr", "rm", "ppo"], default="mlm",
                        nargs='+', help="The training target of the pretraining model.")
    parser.add_argument("--ppo_data_processor",
                        choices=["bert", "lm", "mlm", "bilm", "albert", "mt", "t5", "cls",
                                 "prefixlm", "gsg", "bart", "cls_mlm", "vit", "vilt", "clip",
                                 "s2t", "beit", "dalle", "alpaca", "reward", "ppo"], default="bert",
                        help="The data processor of the pretraining model.")
    parser.add_argument("--unsupervised_data_processor",
                        choices=["bert", "lm", "mlm", "bilm", "albert", "mt", "t5", "cls",
                                 "prefixlm", "gsg", "bart", "cls_mlm", "vit", "vilt", "clip",
                                 "s2t", "beit", "dalle", "alpaca", "reward", "ppo"], default="bert",
                        help="The data processor of the pretraining model.")
    parser.add_argument("--deep_init", action="store_true",
                        help="Scaling initialization of projection layers by a "
                             "factor of 1/sqrt(2N). Necessary to large models.")

    # Optimizer options.
    optimization_opts(parser)
    parser.add_argument("--actor_learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--critic_learning_rate", type=float, default=2e-5,
                        help="Learning rate.")

    # GPU options.
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (GPUs) for training.")
    parser.add_argument("--gpu_ranks", default=[], nargs='+', type=int, help="List of ranks of each process."
                        " Each process has a unique integer rank whose value is in the interval [0, world_size), and runs in a single GPU.")
    parser.add_argument("--master_ip", default="tcp://localhost:12345", type=str, help="IP-Port of master for training.")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl", type=str, help="Distributed backend.")

    # Deepspeed options.
    deepspeed_opts(parser)
    parser.add_argument("--actor_deepspeed_config", default="models/deepspeed_config.json", type=str,
                        help=".")
    parser.add_argument("--critic_deepspeed_config", default="models/deepspeed_config.json", type=str,
                        help=".")
    parser.add_argument("--ref_deepspeed_config", default="models/deepspeed_config.json", type=str,
                        help=".")
    parser.add_argument("--reward_deepspeed_config", default="models/deepspeed_config.json", type=str,
                        help=".")

    # lora options.
    lora_opts(parser)
    parser.add_argument("--actor_lora_pretrained_model_path", type=str, default=None,
                        help="Path of the lora pretrained model.")
    parser.add_argument("--critic_lora_pretrained_model_path", type=str, default=None,
                        help="Path of the lora pretrained model.")

    # Log options.
    log_opts(parser)

    # unused settings.
    parser.add_argument("--whole_word_masking", action="store_true", help="Whole word masking.")
    parser.add_argument("--span_masking", action="store_true", help="Span masking.")
    parser.add_argument("--span_geo_prob", type=float, default=0.2,
                        help="Hyperparameter of geometric distribution for span masking.")
    parser.add_argument("--span_max_length", type=int, default=10,
                        help="Max length for span masking.")

    args = parser.parse_args()

    # construct lora dict parameters.
    # todo yuhaofeng, actor and critic lora params is different.
    if args.use_lora:
        args.lora_params = {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout
        }
    else:
        args.lora_params = None

    ranks_num = len(args.gpu_ranks)

    if args.world_size > 1:
        args.dist_train = True
    else:
        args.dist_train = False

    set_seed(args.seed)

    # initialize
    deepspeed.init_distributed(dist_backend=args.backend)
    rank = dist.get_rank()
    gpu_id = args.local_rank

    # tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    # data loader
    ppo_train_loader = str2dataloader[args.ppo_data_processor](args, args.ppo_dataset_path, args.batch_size,
                                                               rank, args.world_size, gpu_id, True)
    unsupervised_train_loader = None
    if args.unsupervised_dataset_path is not None:
        unsupervised_train_loader = str2dataloader[args.unsupervised_data_processor](args,
                                                                                     args.unsupervised_dataset_path,
                                                                                     args.batch_size, rank,
                                                                                     args.world_size, gpu_id, True)

    # engine loader
    rlhf_engine = RLHFEngine(args)

    # trainer loader
    trainer = PPOTrainer(args, rlhf_engine, ppo_train_loader, unsupervised_train_loader, gpu_id, rank)
    trainer.train()


if __name__ == "__main__":
    main()
