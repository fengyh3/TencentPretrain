import time
import torch
import torch.nn.functional as F
from generate import generate
from utils import moving_average


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class PPOTrainer:
    def __init__(self, args, rlhf_engine, ppo_train_loader, unsupervised_train_loader, gpu_id, rank):
        self.args = args
        self.rlhf_engine = rlhf_engine
        self.ppo_train_loader = ppo_train_loader
        self.unsupervised_train_loader = unsupervised_train_loader
        self.gpu_id = gpu_id
        self.rank = rank
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.actor_output_model_path = args.actor_output_model_path
        self.critic_output_model_path = args.critic_output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0
        self.act_loss = 0.0
        self.cri_loss = 0.0
        self.pretrained_loss = 0.0
        self.best_loss = float("inf")

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.logger = args.logger

    def generate_experience(self, prompts):
        # prompts: [0, 0, 0, ... prompts]
        eos_id = self.args.tokenizer.eos_id
        pad_id = self.args.tokenzier.pad_id
        self.set_eval()
        with torch.no_grad():
            # seq -> [0, 0, prompts, ans, 0, 0...]
            seq = generate(self.rlhf_engine.actor_model, self.args, prompts, eos_id, pad_id)

        self.set_train()

        mask = (seq != pad_id).long()
        prompt_len = prompts.shape[1]
        with torch.no_grad():
            actor_output = self.rlhf_engine.actor_model(seq, None, mask, return_logits=True)
            ref_output = self.rlhf_engine.ref_model(seq, None, mask, return_logits=True)
            reward_output = self.rlhf_engine.reward_model(seq, None, mask)[:, prompt_len:]
            seg = mask[:, prompt_len:].sum(dim=1).reshape(-1, 1).long() - 1
            reward_output = torch.gather(reward_output, dim=1, index=seg)

            # critic_output -> token MDP
            critic_output = self.rlhf_engine.critic_model(seq, None, mask).detach()[:, :-1]

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(actor_output[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(ref_output[:, :-1, :], seq[:, 1:]),
            'value': critic_output,
            'rewards': reward_output.squeeze(-1),
            'input_ids': seq,
            'attention_mask': mask
        }

    def train(self):
        ppo_loader_iter = iter(self.ppo_train_loader)
        if self.unsupervised_train_loader is not None:
            unsupervised_loader_iter = iter(self.unsupervised_train_loader)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            ppo_batch = next(ppo_loader_iter)
            if self.unsupervised_train_loader is not None:
                unsupervised_batch = list(next(unsupervised_loader_iter))

            if self.gpu_id is not None:
                ppo_batch = ppo_batch.cuda(self.gpu_id)
                # for i in range(len(ppo_batch)):
                #     if torch.is_tensor(ppo_batch[i]):
                #         ppo_batch[i] = ppo_batch[i].cuda(self.gpu_id)
                if self.unsupervised_train_loader is not None:
                    for i in range(len(unsupervised_batch)):
                        if torch.is_tensor(unsupervised_batch[i]):
                            unsupervised_batch[i] = unsupervised_batch[i].cuda(self.gpu_id)

            # make experience
            experience = self.generate_experience(ppo_batch)
            prompts = experience['prompts']
            log_probs = experience['logprobs']
            ref_log_probs = experience['ref_logprobs']
            reward_scores = experience['rewards']
            values = experience['value']
            seq = experience['input_ids']
            attention_mask = experience['attention_mask']

            # compute reward and advantage
            ans_start = prompts.shape[1] - 1
            action_mask = attention_mask[:, 1:]
            old_values = values
            with torch.no_grad():
                old_rewards = self.compute_reward(ans_start, log_probs, ref_log_probs, reward_scores, action_mask)
                advantages, returns = self.get_advantages_and_returns(ans_start, old_values, old_rewards)

            # compute actor loss
            actor_probs = self.rlhf_engine.actor_model(seq, None, attention_mask, return_logits=True)
            actor_log_probs = gather_log_probs(actor_probs[:, :-1, :], seq[:, 1:])
            actor_loss = self.actor_loss(actor_log_probs[:, ans_start:], log_probs[:, ans_start:],
                                         advantages, action_mask[:, ans_start:])
            self.rlhf_engine.actor_model.backward(actor_loss)
            self.rlhf_engine.actor_model.step()
            self.act_loss += actor_loss.item()

            # compute critic loss
            value = self.rlhf_engine.critic_model(seq, None, attention_mask)[:, :-1]
            critic_loss = self.critic_loss(value[:, ans_start:], old_values[:, ans_start:],
                                           returns, action_mask[:, ans_start:])
            self.rlhf_engine.critic_model.backward(critic_loss)
            self.rlhf_engine.critic_model.step()
            self.cri_loss += critic_loss.item()

            # compute pretrained loss
            if self.unsupervised_train_loader is not None:
                src, tgt, seg = unsupervised_batch
                pretrained_logits = self.rlhf_engine.actor_model(src, None, seg)
                pretrained_loss = self.lm_loss(pretrained_logits, tgt, seg)
                pretrained_loss = self.args.unsup_coef * pretrained_loss
                self.rlhf_engine.actor_model.backward(pretrained_loss)
                self.rlhf_engine.actor_model.step()
                self.pretrained_loss += pretrained_loss.item()

            if self.args.use_ema:
                moving_average(self.rlhf_engine.actor_model, self.rlhf_engine.ema_model,
                               zero_stage_3=self.args.enable_zero3)

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and self.rank == 0)):
                self.logger.info("| {:8d}/{:8d} steps"
                                 "| actor loss {:7.2f}"
                                 "| critic loss {:7.2f}"
                                 "| lm loss: {:7.2f}".format(
                    self.current_step,
                    self.total_steps,
                    self.act_loss / self.report_steps,
                    self.cri_loss / self.report_steps,
                    self.pretrained_loss / self.report_steps))

                self.act_loss, self.cri_loss, self.pretrained_loss = 0.0, 0.0, 0.0

            self.current_step += 1

    def set_eval(self):
        self.rlhf_engine.actor_model.eval()
        self.rlhf_engine.critic_model.eval()
        self.rlhf_engine.ref_model.eval()
        self.rlhf_engine.reward_model.eval()

    def set_train(self):
        self.rlhf_engine.actor_model.train()
        self.rlhf_engine.critic_model.train()

    def compute_reward(self, ans_start, log_probs, ref_log_probs, reward_score, action_mask):
        kl_div = -self.args.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_div
        ends = ans_start + action_mask[:, ans_start:].sum(1)
        rewards_clip = torch.clamp(reward_score, -self.args.clip_reward_value, self.args.clip_reward_value)
        bs = log_probs.shape[0]
        for i in range(bs):
            rewards[i, ans_start:ends[i]][-1] += rewards_clip[i]
        return rewards

    def get_advantages_and_returns(self, ans_start, values, rewards):
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.shape[-1]
        for t in reversed(range(ans_start, length)):
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * next_values - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, ans_start:]
        return advantages.detach(), returns

    def actor_loss(self, log_probs, old_log_probs, advantages, mask):
        log_ratio = (log_probs - old_log_probs) * mask
        ratio = torch.exp(log_ratio)
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.cliprange, 1.0 + self.args.cliprange)
        policy_loss = torch.sum(torch.max(policy_loss1, policy_loss2) * mask) / mask.sum()
        return policy_loss

    def critic_loss(self, values, old_values, returns, mask):
        values_clipped = torch.clamp(values, old_values - self.args.cliprange_value,
                                     old_values + self.args.cliprange_value)
        value_loss1 = (values - returns) ** 2
        value_loss2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.sum(torch.max(value_loss1, value_loss2) * mask) / mask.sum()
        return value_loss

    def lm_loss(self, logits, tgt_lm, seg):
        tgt_lm = tgt_lm.contiguous().view(-1)
        seg = seg.contiguous().view(-1)
        logits = logits.contiguous().view(-1, logits.shape[-1])
        logits = logits[seg > 0, :]
        tgt_lm = tgt_lm[seg > 0]
        loss = torch.nn.NLLLoss()(logits, tgt_lm)

        return loss
