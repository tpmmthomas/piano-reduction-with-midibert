import numpy as np
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
import pickle


class O2p:
    def __init__(self, dict):
        self.event2word, self.word2event = pickle.load(open(dict, "rb"))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.PAD = np.array(
            [self.event2word[etype]["%s <PAD>" % etype] for etype in self.event2word]
        )
        self.EOS = np.array(
            [self.event2word[etype]["%s <EOS>" % etype] for etype in self.event2word]
        )
        self.BOS = np.array(
            [self.event2word[etype]["%s <BOS>" % etype] for etype in self.event2word]
        )
        self.ABS = np.array(
            [self.event2word[etype]["%s <ABS>" % etype] for etype in self.event2word]
        )
        self.orch_max_len = 128  # parameters
        self.piano_max_len = 128

    def align_token(self, notes, length):  # align the tokens bar by bar
        out = []
        bar = []
        bar_count = 0
        seen_first = False
        tpb = 480
        note_idx = 0

        while note_idx < len(notes):
            note = notes[note_idx]
            if (
                bar_count * 4 * tpb <= note[5] < (bar_count + 1) * tpb * 4
            ):  # within current bar
                bar.append(note[:5])
                if seen_first == True and note[0] == 0:
                    print(note, note_idx, notes)
                assert not (
                    seen_first == True and note[0] == 0
                )  # ASSERT: no two 0(newbar) within the same bar
                if not seen_first:
                    seen_first = True
                    bar[-1][0] = 0
                note_idx += 1
            else:
                # assert(len(bar)>0)
                if len(bar) > 0:  # add <ABS> if it is an empty bar
                    out.append(bar)
                else:
                    out.append([list(self.ABS)])
                bar = []
                bar_count += 1
                seen_first = False

        if len(bar) > 0:  # add <ABS> if it is an empty bar
            out.append(bar)
        else:
            out.append([list(self.ABS)])

        bar = []
        bar_count += 1
        seen_first = False

        assert bar_count == length
        return out

    def generate(self, all_tokens_orch, all_tokens_piano):
        current_bar = -1  # add onset &offset on each token
        tpb = 480
        token_with_on_off_set = []
        orch_tokens = []
        piano_tokens = []
        temp_skyline = []
        for token in all_tokens_orch:
            token = np.array(token)
            if not ((token == self.PAD).all() or (token == self.EOS).all()):
                if token[0] == 0:
                    current_bar += 1
                temp = list(token)
                temp.append(int(current_bar * 4 * tpb + token[1] * tpb / 4))  # onset
                temp.append(
                    int(
                        current_bar * 4 * tpb
                        + token[1] * tpb / 4
                        + (token[3] + 1) * tpb / 8
                    )
                )  # offset
                token_with_on_off_set.append(temp)
        total_bar_orch = current_bar + 1
        token_with_on_off_set = sorted(
            token_with_on_off_set, key=lambda x: (x[5], x[0], x[2], x[4])
        )
        orch = self.align_token(token_with_on_off_set, total_bar_orch)
        current_bar = -1
        token_with_on_off_set = []
        for token in all_tokens_piano:
            token = np.array(token)
            if not ((token == self.PAD).all() or (token == self.EOS).all()):
                if token[0] == 0:
                    current_bar += 1
                temp = list(token)
                temp.append(int(current_bar * 4 * tpb + token[1] * tpb / 4))  # onset
                temp.append(
                    int(
                        current_bar * 4 * tpb
                        + token[1] * tpb / 4
                        + (token[3] + 1) * tpb / 8
                    )
                )  # offset
                token_with_on_off_set.append(temp)
        total_bar_piano = current_bar + 1  # skyline
        token_with_on_off_set = sorted(
            token_with_on_off_set, key=lambda x: (x[5], x[0], x[2])
        )
        piano = self.align_token(token_with_on_off_set, total_bar_piano)
        try:
            assert len(piano) == len(orch)
        except:
            print("error", len(piano), len(orch))
            return [], []

        current_bar = 0
        max_token_len = 0
        temp_orch = []
        temp_piano = [self.BOS]
        while current_bar < total_bar_piano:
            while (
                current_bar < total_bar_piano
                and len(temp_orch) + len(orch[current_bar]) < self.orch_max_len
                and len(temp_piano) + len(piano[current_bar]) < self.piano_max_len
            ):
                temp_orch += orch[current_bar]
                temp_piano += piano[current_bar]
                current_bar += 1
            assert (
                0 < len(temp_orch) < self.orch_max_len
                and 0 < len(temp_piano) < self.piano_max_len
            )  # at least it shld has the ABS token

            temp_piano.append(self.EOS)
            temp_orch = np.array(temp_orch).reshape(-1, 5)
            temp_piano = np.array(temp_piano).reshape(-1, 5)

            while len(temp_orch) < self.piano_max_len:  # add PAD
                temp_orch = np.vstack((temp_orch, self.PAD))
            while len(temp_piano) < self.piano_max_len + 1:
                temp_piano = np.vstack((temp_piano, self.PAD))
            orch_tokens.append(temp_orch)
            piano_tokens.append(temp_piano)
            temp_orch = []
            temp_piano = [self.BOS]
        orch_tokens = np.array(orch_tokens)
        piano_tokens = np.array(piano_tokens)
        return orch_tokens, piano_tokens
