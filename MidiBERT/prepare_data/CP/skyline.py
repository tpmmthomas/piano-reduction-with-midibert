import numpy as np
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
import pickle


class Skyline:
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
        self.skyline_max_len = 90  # parameters
        self.full_max_len = 512

    # ref to ./dict/CP_skyline.pkl
    # PAD = np.array([2, 16, 86, 64])  # --> padding
    # EOS = np.array([4, 18, 88, 66])  # --> End of input segment
    # ABS = np.array([5, 19, 89, 67])  # --> empty bar produced by the skyline algo,
    #    (e.g. skyline pick a long note from the bar ahead)

    def mergeIntervals(self, arr):
        # Sorting based on the increasing order
        # of the start intervals
        arr.sort(key=lambda x: x[0])
        # array to hold the merged intervals
        m = []
        s = -10000
        max = -100000
        for i in range(len(arr)):
            a = arr[i]
            if a[0] > max:
                if i != 0:
                    m.append([s, max])
                max = a[1]
                s = a[0]
            else:
                if a[1] >= max:
                    max = a[1]
        #'max' value gives the last point of
        # that particular interval
        # 's' gives the starting point of that interval
        # 'm' array contains the list of all merged intervals
        if max != -100000 and [s, max] not in m:
            m.append([s, max])
        return m

    def gettop(self, note, intervals):
        note_interval = [note[4], note[5]]  # onset,offset
        overlap_time = 0
        total_time = note[5] - note[4]
        if total_time == 0:
            return 1  # (we do not need this note)
        for interval in intervals:
            maxstart = max(note_interval[0], interval[0])
            minend = min(note_interval[1], interval[1])
            if maxstart < minend:
                overlap_time += minend - maxstart
        return overlap_time / total_time

    def skyline(self, notes):  # revised skyline algorithm by Chai, 2000
        # Performed on a single channel
        accepted_notes = []
        notes = sorted(notes, key=lambda x: x[2], reverse=True)  # sort by pitch
        intervals = []
        for note in notes:
            if self.gettop(note, intervals) <= 0.5:
                accepted_notes.append(note)
                intervals.append([note[4], note[5]])  # onset,offset
                intervals = self.mergeIntervals(intervals)
        return sorted(
            accepted_notes, key=lambda x: (x[4], x[0])
        )  # sort by onset & bar(new)

    def skyline_reverse(self, notes):  # revised skyline algorithm by Chai, 2000
        # Performed on a single channel
        accepted_notes = []
        notes = sorted(notes, key=lambda x: x[2])  # sort by pitch
        intervals = []
        for note in notes:
            if self.gettop(note, intervals) <= 0.8:
                accepted_notes.append(note)
                intervals.append([note[4], note[5]])  # onset,offset
                intervals = self.mergeIntervals(intervals)
        return sorted(
            accepted_notes, key=lambda x: (x[4], x[0])
        )  # sort by onset & bar(new)

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
                bar_count * 4 * tpb <= note[4] < (bar_count + 1) * tpb * 4
            ):  # within current bar
                bar.append(note[:4])
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

    def generate(self, all_tokens):
        current_bar = -1  # add onset &offset on each token
        tpb = 480
        token_with_on_off_set = []
        skyline_tokens = []
        full_tokens = []
        temp_skyline = []
        allsong_skyline_tokens = []
        allsong_full_tokens = []
        for token in all_tokens:
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
        total_bar = current_bar + 1  # skyline
        token_with_on_off_set = sorted(
            token_with_on_off_set, key=lambda x: (x[4], x[0], x[2])
        )
        org = self.align_token(token_with_on_off_set, total_bar)
        sl = self.skyline(token_with_on_off_set) + self.skyline_reverse(
            token_with_on_off_set
        )
        sl = [tuple(x) for x in sl]  # remove duplication
        sl = list(dict.fromkeys(sl))
        sl = [list(x) for x in sl]
        sl = sorted(sl, key=lambda x: (x[4], x[0], x[2]))  # sort by onset & bar(new)
        sl = self.align_token(sl, total_bar)

        current_bar = 0
        max_token_len = 0
        temp_skyline = []
        temp_full = [self.BOS]
        while current_bar < total_bar:
            while (
                current_bar < total_bar
                and len(temp_skyline) + len(sl[current_bar]) < self.skyline_max_len
            ):
                temp_skyline += sl[current_bar]
                temp_full += org[current_bar]
                current_bar += 1
            assert (
                0 < len(temp_skyline) < self.skyline_max_len
                and 0 < len(temp_full) < self.full_max_len
            )  # at least it shld has the ABS token

            temp_full.append(self.EOS)
            temp_skyline = np.array(temp_skyline).reshape(-1, 4)
            temp_full = np.array(temp_full).reshape(-1, 4)

            while len(temp_skyline) < self.full_max_len:  # add PAD
                temp_skyline = np.vstack((temp_skyline, self.PAD))
            if len(temp_full) > max_token_len:
                max_token_len = len(temp_full)
            while len(temp_full) < self.full_max_len + 1:
                temp_full = np.vstack((temp_full, self.PAD))
            skyline_tokens.append(temp_skyline)
            full_tokens.append(temp_full)
            temp_skyline = []
            temp_full = [self.BOS]
        print(max_token_len)
        skyline_tokens = np.array(skyline_tokens)
        full_tokens = np.array(full_tokens)
        return skyline_tokens, full_tokens
