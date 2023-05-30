# modified from MidiBERT/CP/main.py

import argparse
import numpy as np
import pickle
import os

from torch.utils.data import DataLoader
from transformers import BertConfig
from model import MidiBert, MidiBertSeq2Seq
from trainer import BERTTrainer, BERTSeq2SeqTrainer
from midi_dataset import MidiDataset, Seq2SeqDataset
import logging
from sklearn.model_selection import train_test_split
from glob import glob

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="")

    ### path setup ###
    parser.add_argument("--dict_file", type=str, default="../dict/CP.pkl")
    parser.add_argument("--name", type=str, default="MidiBert")

    ### pre-train dataset ###
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--seq2seq_checkpoint", type=str)
    ### parameter setting ###
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument(
        "--mask_percent",
        type=float,
        default=0.15,
        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="all sequences are padded to `max_seq_len`",
    )
    parser.add_argument("--hs", type=int, default=768)  # hidden state
    parser.add_argument(
        "--epochs", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="initial learning rate")
    parser.add_argument("--mode", type=str, default="bert", help="bert or seq2seq")

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=[0, 1], help="CUDA device ids"
    )

    args = parser.parse_args()

    return args


def load_data(datasets, mode="bert"):
    to_concat = []

    if mode == "seq2seq":
        X = np.load(f"{datasets[0]}.npy", allow_pickle=True)
        y = np.load(f"{datasets[0]}_ans.npy", allow_pickle=True)
        logger.info("shape of input {} {}".format(X.shape, y.shape))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        logger.info("shape of train input {} {}".format(X_train.shape, y_train.shape))
        return X_train, y_train, X_test, y_test
    for dataset in datasets:
        data = np.load(f"{dataset}.npy", allow_pickle=True)
        to_concat.append(data)

    training_data = np.vstack(to_concat)

    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data) * 0.85)
    X_train, X_val = training_data[:split], training_data[split:]

    return X_train, X_val


def main():
    args = get_args()

    logger.info("Loading Dictionary")
    with open(args.dict_file, "rb") as f:
        e2w, w2e = pickle.load(f)

    logger.info("Loading Dataset {}".format(args.datasets))
    if args.mode == "seq2seq":
        X_train, Y_train, X_test, Y_test = load_data(args.datasets, args.mode)
        trainset = Seq2SeqDataset(X=X_train, y=Y_train)
        validset = Seq2SeqDataset(X=X_test, y=Y_test)
        logger.info(f"X Training data shape: {X_train.shape}")
        logger.info(f"X Testing data shape: {X_test.shape}")
        logger.info(f"Y Training data shape: {Y_train.shape}")
        logger.info(f"Y Testing data shape: {Y_test.shape}")
    else:
        X_train, X_val = load_data(args.datasets, args.mode)
        trainset = MidiDataset(X=X_train)
        validset = MidiDataset(X=X_val)
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    logger.info("Len of train_loader {}".format(len(train_loader)))
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    logger.info("Len of valid_loader {}".format(len(valid_loader)))

    logger.info("Building BERT model")
    configuration = BertConfig(
        max_position_embeddings=args.max_seq_len,
        position_embedding_type="relative_key_query",
        hidden_size=args.hs,
    )
    if args.mode == "seq2seq":
        config_en = BertConfig(
            max_position_embeddings=args.max_seq_len,
            position_embedding_type="relative_key_query",
            hidden_size=args.hs,
        )
        config_de = BertConfig(
            max_position_embeddings=args.max_seq_len,
            position_embedding_type="relative_key_query",
            hidden_size=args.hs,
        )
        config_de.is_decoder = True
        config_de.add_cross_attention = True

        midibert = MidiBertSeq2Seq(
            config_en, config_de, args.checkpoint, args.seq2seq_checkpoint, e2w, w2e
        )
        logger.info("Creating BERT Trainer")
        trainer = BERTSeq2SeqTrainer(
            midibert,
            train_loader,
            valid_loader,
            args.lr,
            args.batch_size,
            args.epochs,
            args.max_seq_len,
            args.cpu,
            args.cuda_devices,
            args.seq2seq_checkpoint,
        )
        save_dir = "result/seq2seq/" + args.name
    else:
        configuration = BertConfig(
            max_position_embeddings=args.max_seq_len,
            position_embedding_type="relative_key_query",
            hidden_size=args.hs,
        )
        midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

        logger.info("Creating BERT Trainer")
        trainer = BERTTrainer(
            midibert,
            train_loader,
            valid_loader,
            args.lr,
            args.batch_size,
            args.max_seq_len,
            args.mask_percent,
            args.cpu,
            args.cuda_devices,
        )
        save_dir = "result/pretrain/" + args.name

        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)

    logger.info("Training Start")
    os.makedirs(save_dir, exist_ok=True)
    logger.info("   save model at {}".format(save_dir))

    best_acc, best_loss = 0, 99999
    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x * y for (x, y) in zip(valid_acc, midibert.n_tokens)]
        avg_acc = sum(weighted_score) / sum(midibert.n_tokens)

        if avg_acc >= best_acc:
            os.remove(glob(f"{save_dir}/model_best-acc_*"))
            trainer.save_checkpoint(
                f"{save_dir}/model_best-acc_epoch={epoch}_loss={valid_loss}_acc={weighted_score}.ckpt"
            )

        if valid_loss <= best_loss:
            os.remove(glob(f"{save_dir}/model_best-loss_*"))
            trainer.save_checkpoint(
                f"{save_dir}/model_best-loss_epoch={epoch}_loss={valid_loss}_acc={weighted_score}.ckpt"
            )
        trainer.save_checkpoint(f"{save_dir}/model_last.ckpt")

        best_acc = max(avg_acc, best_acc)
        best_loss = min(valid_loss, best_loss)

        logger.info(
            "epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}".format(
                epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc
            )
        )

        with open(os.path.join(save_dir, "log"), "a") as outfile:
            outfile.write(
                "Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n".format(
                    epoch + 1, train_loss, train_acc, valid_loss, valid_acc
                )
            )


if __name__ == "__main__":
    main()
