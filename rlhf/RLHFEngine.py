import deepspeed.ops.adam
from rlhf.utils import *
import deepspeed

'''
manage all models in rlhf training.
'''


class RLHFEngine:
    def __init__(self, args):
        self.args = args
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.ref_model = self.build_reference()
        self.reward_model = self.build_reward()

    def build_actor(self):
        # replace model config parameters
        self.args = load_hyperparam(self.args, self.args.actor_config_path)
        self.args.deepspeed_config = self.args.actor_deepspeed_config
        actor_model = build_and_load(self.args, self.args.actor_target, deepspeed_config=self.args.actor_deepspeed_config,
                                     pretrained_model_path=self.args.actor_pretrained_model_path,
                                     lora_pretrained_model_path=self.args.actor_lora_pretrained_model_path)
        # get optimizer parameters.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.args, actor_model)

        # get optimizer, only use adam
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer_grouped_parameters,
                                                        lr=self.args.actor_learning_rate,
                                                        bias_correction=False)

        scheduler = get_scheduler(self.args, optimizer)
        actor_model, optimizer, _, scheduler = deepspeed.initialize(model=actor_model,
                                                                    model_parameters=optimizer_grouped_parameters,
                                                                    args=self.args,
                                                                    optimizer=optimizer,
                                                                    lr_scheduler=scheduler,
                                                                    dist_init_required=False)
        return actor_model

    def build_critic(self):
        self.args = load_hyperparam(self.args, self.args.critic_config_path)
        self.args.deepspeed_config = self.args.critic_deepspeed_config
        critic_model = build_and_load(self.args, self.args.critic_target, deepspeed_config=self.args.critic_deepspeed_config,
                                      pretrained_model_path=self.args.critic_pretrained_model_path,
                                      lora_pretrained_model_path=self.args.critic_lora_pretrained_model_path)
        # get optimizer parameters.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.args, critic_model)

        # get optimizer, only use adam
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer_grouped_parameters,
                                                        lr=self.args.critic_learning_rate,
                                                        bias_correction=False)

        scheduler = get_scheduler(self.args, optimizer)
        critic_model, optimizer, _, scheduler = deepspeed.initialize(model=critic_model,
                                                                     model_parameters=optimizer_grouped_parameters,
                                                                     args=self.args,
                                                                     optimizer=optimizer,
                                                                     lr_scheduler=scheduler,
                                                                     dist_init_required=False)
        return critic_model

    def build_reference(self):
        self.args = load_hyperparam(self.args, self.args.actor_config_path)
        self.args.deepspeed_config = self.args.ref_deepspeed_config
        ref_model = build_and_load(self.args, self.args.actor_target, deepspeed_config=self.args.ref_deepspeed_config,
                                   pretrained_model_path=self.args.actor_pretrained_model_path,
                                   lora_pretrained_model_path=self.args.actor_lora_pretrained_model_path)

        ref_model, optimizer, _, scheduler = deepspeed.initialize(model=ref_model,
                                                                  args=self.args,
                                                                  dist_init_required=False)
        return ref_model

    def build_reward(self):
        self.args = load_hyperparam(self.args, self.args.critic_config_path)
        self.args.deepspeed_config = self.args.reward_deepspeed_config
        reward_model = build_and_load(self.args, self.args.critic_target, deepspeed_config=self.args.reward_deepspeed_config,
                                      pretrained_model_path=self.args.critic_pretrained_model_path,
                                      lora_pretrained_model_path=self.args.critic_lora_pretrained_model_path)

        reward_model, optimizer, _, scheduler = deepspeed.initialize(model=reward_model,
                                                                     args=self.args,
                                                                     dist_init_required=False)
        return reward_model

    # todo ema
    def build_ema(self):
        pass

    def build_unsupervised(self):
        pass
