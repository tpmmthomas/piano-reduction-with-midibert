import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import tqdm
import copy

from model import MidiBert, MidiBertSeq2Seq
from modelLM import MidiBertLM, MidiBertSeq2SeqComplete
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class BERTTrainer:
    def __init__(
        self,
        midibert: MidiBert,
        train_dataloader,
        valid_dataloader,
        lr,
        batch,
        max_seq_len,
        mask_percent,
        cpu,
        cuda_devices=None,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        )
        self.midibert = midibert  # save this for ckpt
        self.model = MidiBertLM(midibert).to(self.device)
        self.total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"# total parameters: {self.total_params}")

        if torch.cuda.device_count() > 1 and not cpu:
            logger.info(f"Use {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

        # for tracking and calculating the average loss
        self.losses = []

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        mask_ind = random.sample(self.Lseq, round(self.max_seq_len * self.mask_percent))
        mask80 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
        left = list(set(mask_ind) - set(mask80))
        rand10 = random.sample(left, round(len(mask_ind) * 0.1))
        cur10 = list(set(left) - set(rand10))
        return mask80, rand10, cur10

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(
                self.valid_data, self.max_seq_len, train=False
            )
        return valid_loss, valid_acc

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data)
        total_acc, total_losses = [0] * len(self.midibert.e2w), 0

        for step, ori_seq_batch in enumerate(pbar):
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)  # (batch, seq_len, 4)
            input_ids = copy.deepcopy(ori_seq_batch)
            loss_mask = torch.zeros(batch, max_seq_len)

            for b in range(batch):
                # get index for masking
                mask80, rand10, cur10 = self.get_mask_ind()
                # apply mask, random, remain current token
                for i in mask80:
                    mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                for i in rand10:
                    rand_word = torch.tensor(self.midibert.get_rand_tok()).to(
                        self.device
                    )
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(self.device)

            # avoid attend to pad word
            attn_mask = (
                (input_ids[:, :, 0] != self.midibert.bar_pad_word)
                .float()
                .to(self.device)
            )  # (batch, seq_len)

            y = self.model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            # accuracy of 6 token types: [Bar, Position, Pitch, Duration, Program, Time Signature]
            all_acc = []
            for i in range(len(self.midibert.n_tokens)):
                acc = torch.sum(
                    (ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask
                )
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))
                losses.append(self.compute_loss(y[i], ori_seq_batch[..., i], loss_mask))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)
                self.optim.step()

            losses = list(map(float, losses))
            # total losses in an epoch
            total_losses += total_loss.item()

            # total losses over all epochs
            self.losses.append(total_loss.item())
            if step % 4 == 0:
                all_acc = " ".join([f"{acc.item():.3f}" for acc in all_acc])
                avg_loss = sum(self.losses) / len(self.losses)
                pbar.set_postfix(
                    {
                        "accs": all_acc,
                        "cur loss": total_loss.item(),
                        "avg loss": avg_loss,
                    }
                )

        return round(total_losses / len(training_data), 3), [
            round(x.item() / len(training_data), 3) for x in total_acc
        ]

    def save_checkpoint(self, filename):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(state, filename + ".ckpt")

    def load_checkpoint(self, ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
        skip_list = ["word_emb", "in_linear"]
        state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not any(skip in k for skip in skip_list)
        }
        self.midibert.load_state_dict(state_dict, strict=False)
        # self.optim.load_state_dict(checkpoint["optimizer"])


class BERTSeq2SeqTrainer:
    def __init__(
        self,
        midibert: MidiBertSeq2Seq,
        train_dataloader,
        valid_dataloader,
        lr,
        batch,
        num_epochs,
        max_seq_len,
        cpu,
        cuda_devices=None,
        checkpoint=None,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        )
        self.midibert = midibert
        self.model = MidiBertSeq2SeqComplete(midibert).to(self.device)
        if checkpoint is not None:
            checkpoint = torch.load(f"./result/seq2seq/{checkpoint}/model_best.ckpt")
            for key in list(checkpoint["state_dict"].keys()):
                # rename the states in checkpoint
                checkpoint["state_dict"][key.replace("module.", "")] = checkpoint[
                    "state_dict"
                ].pop(key)
            self.model.load_state_dict(checkpoint["state_dict"])
        self.total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info("# total parameters: {self.total_params}")

        if torch.cuda.device_count() > 1 and not cpu:
            logger.info(f"Use {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        # num_steps_per_epoch = int(len(self.train_data) / batch)
        # warmup_steps = int(num_steps_per_epoch * 0.1)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optim, warmup_steps, num_steps_per_epoch * num_epochs
        # )
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

        # for tracking and calculating the average loss
        self.losses = []

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(
                self.valid_data, self.max_seq_len, train=False
            )
        return valid_loss, valid_acc

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data)
        total_acc, total_losses = [0] * len(self.midibert.e2w), 0

        for step, ori_seq_batch in enumerate(pbar):
            batch = ori_seq_batch[0].shape[0]
            ori_seq_batch_x = ori_seq_batch[0].to(self.device)  # (batch, seq_len, 4)
            ori_seq_batch_y = ori_seq_batch[1].to(self.device)  # (batch, seq_len+1, 4)
            input_ids = copy.deepcopy(ori_seq_batch_x)
            target = copy.deepcopy(ori_seq_batch_y)
            loss_mask = torch.zeros(batch, max_seq_len)
            decoder_input_ids = target[:, :-1, :]
            decoder_target = target[:, 1:, :]
            decoder_target = decoder_target.type(torch.int64)

            # avoid attend to pad word
            attn_mask_encoder = (
                (input_ids[:, :, 0] != self.midibert.bar_pad_word)
                .float()
                .to(self.device)
            )  # (batch, seq_len)

            attn_mask_decoder = (
                (decoder_input_ids[:, :, 0] != self.midibert.bar_pad_word)
                .float()
                .to(self.device)
            )  # (batch, seq_len)

            loss_mask = (
                (decoder_target[:, :, 0] != self.midibert.bar_pad_word)
                .long()
                .to(self.device)
            )

            y = self.model.forward(
                input_ids, decoder_input_ids, attn_mask_encoder, attn_mask_decoder
            )  # list of 4 np.array of len (batch,seq_len,possible_choices)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len,4)

            # accuracy of 6 token types: [Bar, Position, Pitch, Duration, Program, Time Signature]
            all_acc = []
            for i in range(len(self.midibert.n_tokens)):
                acc = torch.sum(
                    (decoder_target[:, :, i] == outputs[:, :, i]).float() * loss_mask
                )
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                y[i] = y[i][:, ...].permute(0, 2, 1)

            # calculate losses
            losses, n_tok = [], []
            # importance = [1, 1, 2, 1, 1, 1]  # hardcoded, be careful
            for i, etype in enumerate(self.midibert.e2w):
                n_tok.append(len(self.midibert.e2w[etype]))
                losses.append(
                    self.compute_loss(y[i], decoder_target[..., i], loss_mask)
                )
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            # total_loss_all = [x * y * z for x, y, z in zip(losses, n_tok, importance)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 2.0)  # Reduced from 3.0 to 2.0
                self.optim.step()
                # self.scheduler.step()
                # print(sum([gp['lr'] for gp in self.optim.param_groups]) / len(self.optim.param_groups))

            # delete stuff
            del ori_seq_batch_x
            del ori_seq_batch_y
            del attn_mask_encoder
            del attn_mask_decoder
            del loss_mask
            torch.cuda.empty_cache()
            losses = list(map(float, losses))
            total_losses += total_loss.item()

            # total losses over all epochs
            self.losses.append(total_loss.item())
            if step % 4 == 0:
                all_acc = " ".join([f"{acc.item():.3f}" for acc in all_acc])
                avg_loss = sum(self.losses) / len(self.losses)
                pbar.set_postfix(
                    {
                        "accs": all_acc,
                        "cur loss": total_loss.item(),
                        "avg loss": avg_loss,
                    }
                )

        return round(total_losses / len(training_data), 3), [
            round(x.item() / len(training_data), 3) for x in total_acc
        ]

    def save_checkpoint(self, filename):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(state, filename + ".ckpt")
