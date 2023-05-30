import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel, EncoderDecoderModel, EncoderDecoderConfig


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration, Program, Time Signature]
        self.n_tokens = []  # [7, 29, 91, 69, 101, 9]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256] * len(self.n_tokens)
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w["Bar"]["Bar <PAD>"]
        self.mask_word_np = np.array(
            [self.e2w[etype]["%s <MASK>" % etype] for etype in self.e2w], dtype=np.long
        )
        self.pad_word_np = np.array(
            [self.e2w[etype]["%s <PAD>" % etype] for etype in self.e2w], dtype=np.long
        )

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)
        # feed to bert
        y = self.bert(
            inputs_embeds=emb_linear,
            attention_mask=attn_mask,
            output_hidden_states=output_hidden_states,
        )
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        return np.array(
            [random.choice(range(num_token)) for num_token in self.n_tokens]
        )


class MidiBertSeq2Seq(nn.Module):
    def __init__(self, config_en, config_de, ckpt, ckpt_s2s, e2w, w2e):
        super().__init__()

        encoder_model = BertModel(config_en)
        decoder_model = BertModel(config_de)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_en, config_de)
        if ckpt is not None and ckpt_s2s is not None:
            print(ckpt, "|||", ckpt_s2s)
            raise Exception(
                "BERT checkpoint and Seq2Seq checkpoint cannot both be provided."
            )

        if ckpt is not None:
            checkpoint = torch.load(f"./result/pretrain/{ckpt}/model_best.ckpt")
            for key in list(checkpoint["state_dict"].keys()):
                # rename the states in checkpoint
                checkpoint["state_dict"][key.replace("bert.", "")] = checkpoint[
                    "state_dict"
                ].pop(key)
            encoder_model.load_state_dict(checkpoint["state_dict"], strict=False)
            decoder_model.load_state_dict(checkpoint["state_dict"], strict=False)
            encoder_model.save_pretrained("./s2s_encoder_model/")
            decoder_model.save_pretrained("./s2s_decoder_model/")
            self.bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
                "./s2s_encoder_model/", "./s2s_decoder_model/", config=config
            )
        else:
            self.bert2bert = EncoderDecoderModel(config=config)
        self.hidden_size = config.encoder.hidden_size
        self.bert2bertConfig = config

        # token types: [Bar, Position, Pitch, Duration, Program, Time Signature]
        self.n_tokens = []  # [7, 29, 91, 192, 101, 9]
        for key in e2w:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256] * len(self.n_tokens)
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w["Bar"]["Bar <PAD>"]
        self.mask_word_np = np.array(
            [self.e2w[etype]["%s <MASK>" % etype] for etype in self.e2w], dtype=np.long
        )
        self.pad_word_np = np.array(
            [self.e2w[etype]["%s <PAD>" % etype] for etype in self.e2w], dtype=np.long
        )
        # TODO: add other token types?

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.e2w):
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.hidden_size)

    def forward(
        self,
        input_ids,
        decoder_ids,
        encoder_attn_mask=None,
        decoder_attn_mask=None,
        output_hidden_states=True,
    ):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.e2w):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # Assume the decoder_ids are already of the correct length
        embs2 = []
        for i, key in enumerate(self.e2w):
            embs2.append(self.word_emb[i](decoder_ids[..., i]))
        embs2 = torch.cat([*embs2], dim=-1)
        emb2_linear = self.in_linear(embs2)

        # feed to bert
        y = self.bert2bert(
            inputs_embeds=emb_linear,
            decoder_inputs_embeds=emb2_linear,
            attention_mask=encoder_attn_mask,
            decoder_attention_mask=decoder_attn_mask,
            output_hidden_states=output_hidden_states,
        )

        return y

    def get_rand_tok(self):
        return np.array(
            [random.choice(range(num_token)) for num_token in self.n_tokens]
        )
