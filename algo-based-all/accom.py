import numpy as np
import random


class Note:
    def __init__(self, s, i, v):
        self.start = s
        self.relative_interval = i
        self.duration = v  # duration


class noteMidi:
    def __init__(self, p, s, e):
        self.pitch = p
        self.onset = s
        self.offset = e


class Accompany:
    def __init__(self, num=4, den=4, notelist=[], classjson=None):
        if classjson is None:
            self.numerator = num
            self.denominator = den
            self.notelist = notelist
        else:
            self.numerator = classjson["numerator"]
            self.denominator = classjson["denominator"]
            self.notelist = classjson["notes"]
            self.rhythm = classjson["rhythm"]

    def export_dict(self):
        tmp = dict()
        tmp["numerator"] = self.numerator
        tmp["denominator"] = self.denominator
        tmp["notes"] = []
        for notes in self.notelist:
            tmp2 = dict()
            tmp2["s"] = notes.start
            tmp2["i"] = notes.relative_interval
            tmp2["d"] = notes.duration
            tmp["notes"].append(tmp2)
        tmp["rhythm"] = self.rhythm
        return tmp

    def add_notes(self, n: Note):
        self.notelist.append(n)

    def calculate_rhythm(self):
        total_duration = self.numerator * (
            0.5 ** (math.log(self.denominator, 2) - 2)
        )  # number of quarter notes
        self.rhythm = [0 for _ in range(24)]
        interval = total_duration / 24
        rhythmtime = [i * interval for i in range(24)]
        for note in self.notelist:
            onset = note.start
            idx, val = self.find_closest(rhythmtime, onset)
            if val < interval / 2:
                self.rhythm[idx] = 1

    def find_closest(self, arr, val):
        newlist = [abs(x - val) for x in arr]
        return np.argmin(newlist), np.min(newlist)


# %%
def notes_bar_processing(notes, begin_tick, tpb, num, den):
    # find min pitch
    min_pitch = 128
    for note in notes:
        if note.pitch < min_pitch:
            min_pitch = note.pitch
    accom = Accompany(num=num, den=den, notelist=[])
    for note in notes:
        start = (note.start - begin_tick) / tpb
        rpitch = note.pitch - min_pitch
        dur = (note.end - note.start) / tpb
        accom.add_notes(Note(start, rpitch, dur))
    return accom


# %% [markdown]
# To reduce complexity
# 1. mappping by time signature
# 2. remove similar or even same entries

# %%
import glob
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
import math

import pickle
from lsh import Accompany, Note

with open("accomdb.pickle", "rb") as f:
    real_database = pickle.load(f)
# %%
# Update on 22/1/2022 :0:06
# LSH of db
from lsh import (
    LSHash,
)  # from https://github.com/kayzhu/LSHash/blob/master/lshash/storage.py

LSHDatabase = dict()
for key in real_database:
    LSHDatabase[key] = LSHash(6, 24)
    for i, entry in enumerate(real_database[key]):
        # format: hash_instance.index(<array to hash>, <(Optional) Tag>)
        LSHDatabase[key].index(entry.rhythm, i)

# %% [markdown]
# # 2. Extract Rhythm of the piece


def main(piece, out, mode="guess", ts_piece="4/4"):
    # %%
    # Find channel with lowest pitch (Now: basically channels using bass clef, any better way?)
    mido_obj = mid_parser.MidiFile(piece)
    minpitch = 129
    chosen_channel = []
    for idx, inst in enumerate(mido_obj.instruments):
        if inst.is_drum:
            continue
        total_pitch = 0
        total_note = 0
        for note in inst.notes:
            total_pitch += note.pitch
            total_note += 1
        avg_pitch = total_pitch / total_note
        # print(idx,avg_pitch)
        if avg_pitch <= 54:
            chosen_channel.append(idx)
    # chosen_channel = [0]#for now
    if chosen_channel == []:
        chosen_channel = [0]

    # %%
    # Aggregate Notes from the selected channels
    final_notelist = []
    for channel in chosen_channel:
        for note in mido_obj.instruments[channel].notes:
            final_notelist.append(note)
    final_notelist = sorted(final_notelist, key=lambda x: x.start)

    class lhMatchInstance:
        def __init__(self, notes, starttick, tpb, num, den, rhythm):
            self.notes = notes
            self.starttick = starttick
            self.tpb = tpb
            self.numerator = num
            self.denominator = den
            self.rhythm = rhythm
            # add intervals
            self.total_duration = (
                self.numerator * (0.5 ** (math.log(self.denominator, 2) - 2)) * tpb
            )
            interval = self.total_duration / numerator
            self.intervals = [
                (self.starttick + i * interval, self.starttick + (i + 1) * interval)
                for i in range(numerator)
            ]
            self.calculated_chordnotes = dict()

        def __str__(self):
            return f"rhythm {self.rhythm} num_notes {len(self.notes)}"

        def chord_notes(self, tick=-1):
            if tick == -1:
                tick = self.starttick + 1
            # choose interval
            chosen_interval = None
            for interval in self.intervals:
                if tick >= interval[0] and tick <= interval[1]:
                    chosen_interval = interval
                    break
            if chosen_interval is None:
                print("Check!")
                print(self.intervals, tick)
                chosen_interval = (self.starttick, self.starttick + self.total_duration)
            if chosen_interval in self.calculated_chordnotes:
                return self.calculated_chordnotes[chosen_interval]
            # consider duration and number
            durations = dict()
            doublings = dict()
            lowest_pitch = 129
            for i in range(12):
                durations[i] = 0
                doublings[i] = 0
            at_least_1 = False
            for note in self.notes:
                if (
                    note.start >= chosen_interval[0]
                    and note.start <= chosen_interval[1]
                ):
                    at_least_1 = True
                    if note.pitch < lowest_pitch:
                        lowest_pitch = note.pitch
                    durations[note.pitch % 12] += note.end - note.start
                    doublings[note.pitch % 12] += 1
            if not at_least_1:
                chosen_interval = (self.starttick, chosen_interval[1])
                for note in self.notes:
                    if (
                        note.start >= chosen_interval[0]
                        and note.start <= chosen_interval[1]
                    ):
                        if note.pitch < lowest_pitch:
                            lowest_pitch = note.pitch
                        durations[note.pitch % 12] += note.end - note.start
                        doublings[note.pitch % 12] += 1
            min_duration = min(durations.values())
            max_duration = max(durations.values())
            min_doubling = min(doublings.values())
            max_doubling = max(doublings.values())
            if max_duration - min_duration == 0:
                max_duration = 1
                min_duration = 0
            if max_doubling - min_doubling == 0:
                max_doubling = 1
                min_doubling = 0
            chord_notelist = [lowest_pitch % 12]
            considerations = []
            for i in range(12):
                score = (durations[i] - min_duration) / (
                    max_duration - min_duration
                ) + (doublings[i] - min_doubling) / (max_doubling - min_doubling)
                considerations.append((i, score))
            considerations = sorted(considerations, key=lambda x: x[1], reverse=True)
            for pitch in considerations[:3]:
                if pitch[0] not in chord_notelist and pitch[1] > 0:
                    chord_notelist.append(pitch[0])
            self.calculated_chordnotes[chosen_interval] = chord_notelist
            return chord_notelist

    def find_closest(arr, val):
        newlist = [abs(x - val) for x in arr]
        return np.argmin(newlist), np.min(newlist)

    def rhythm_processing(notes, begin_tick, tpb, numerator, denominator):
        """
        input:
            notes: list of notes in the miditoolkit.notes class
            bar_length: length of a bar in number of ticks
        returns:
            a 24D vector where each dimension = 1 if the corresponding time has a note onset.
        """
        rhythm_list = [0 for _ in range(24)]
        bar_length = numerator * (0.5 ** (math.log(denominator, 2) - 2))
        interval = bar_length / 24
        rhythm_tick = [i * interval for i in range(24)]
        for note in notes:
            onset = (note.start - begin_tick) / tpb
            idx, val = find_closest(rhythm_tick, onset)
            if val < interval / 2:
                rhythm_list[idx] = 1
        return lhMatchInstance(
            notes, begin_tick, tpb, numerator, denominator, rhythm_list
        )

    # This is to extract rhythm bar by bar
    tschanges = dict()
    for ts in mido_obj.time_signature_changes:
        print(ts)
        tschanges[ts.time] = (ts.numerator, ts.denominator)
    if len(tschanges) == 0:
        tschanges[0] = (4, 4)  # TODO manually add the time signature
    tpb = mido_obj.ticks_per_beat
    numerator = tschanges[0][0]
    denominator = tschanges[0][1]
    add_interval = int(tpb * numerator * (0.5 ** (math.log(denominator, 2) - 2)))
    current_tick = add_interval
    begin_tick = 0
    notelist = []
    tmp_notelist = []
    del tschanges[0]
    song_rhythm = []
    for note in final_notelist:
        if note.start < current_tick:
            if note.end > current_tick:
                tmp_notelist.append(
                    ct.Note(
                        start=current_tick,
                        end=note.end,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
                notelist.append(
                    ct.Note(
                        start=note.start,
                        end=current_tick,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
            else:
                notelist.append(note)
        else:
            if notelist != []:
                song_rhythm.append(
                    rhythm_processing(notelist, begin_tick, tpb, numerator, denominator)
                )
            notelist = []
            begin_tick = current_tick
            if begin_tick in tschanges:
                numerator, denominator = tschanges[begin_tick]
                add_interval = int(
                    tpb * numerator * (0.5 ** (math.log(denominator, 2) - 2))
                )
                del tschanges[begin_tick]
            current_tick += add_interval
            tmp2 = []
            for note2 in tmp_notelist:
                if note2.end > current_tick:
                    tmp2.append(
                        ct.Note(
                            start=current_tick,
                            end=note2.end,
                            pitch=note2.pitch,
                            velocity=note2.velocity,
                        )
                    )
                    notelist.append(
                        ct.Note(
                            start=note2.start,
                            end=current_tick,
                            pitch=note2.pitch,
                            velocity=note2.velocity,
                        )
                    )
                else:
                    notelist.append(note2)
            tmp_notelist = tmp2
            if note.end > current_tick:
                tmp_notelist.append(
                    ct.Note(
                        start=current_tick,
                        end=note.end,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
                notelist.append(
                    ct.Note(
                        start=note.start,
                        end=current_tick,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
            else:
                notelist.append(note)
    if notelist != []:
        song_rhythm.append(
            rhythm_processing(notelist, begin_tick, tpb, numerator, denominator)
        )

    # %% [markdown]
    # # 3. Extract Chord information

    # %%
    # Use the ipervious built thing
    # Since then environment would need to be different, I assume the code is run independently and imported here using csv
    if mode == "csv":
        import pandas as pd

        df = pd.read_csv(
            "../complete_chord_identification/output/orchestra.csv"
        )  # pd.read_csv("../complete_chord_identification/output/twinkle-twinkle-little-star.csv")#
        print(df)
        Cstart_tick = list(df["start_tick"].values)
        Cend_tick = list(df["end_tick"].values)
        Ckey = list(df["key"].values)
        Cchord = list(df["chord"].values)
        # TODO get the chordtonote here
    from chordToNote import ChordToNote
    from skyline import skyline_melody

    # %%
    def similarity(rhythm1, rhythm2):
        # Hamming Distance
        total = 0
        for dim in range(24):
            total += abs(rhythm1[dim] - rhythm2[dim])
        return total

    def match_rhythm(database, rhythm, num, den):
        choices = []

        # Update on 22/1/2022 :0:06
        # lsh query:
        # Input: unhashed array
        # Output: Tuple of Tuple
        #   Format:  ((similar candidates,Tag), similarity)
        key = str(num) + "/" + str(den)
        query = LSHDatabase[key].query(rhythm, distance_func="hamming")
        choices = [(reply[0][1], reply[1]) for reply in query]
        if len(choices) == 0:  # if no suitable candidate
            # loop the data base here
            for i, entry in enumerate(database):
                if entry.numerator == num and entry.denominator == den:
                    entry.calculate_rhythm()
                    score = similarity(rhythm, entry.rhythm)  # smaller the better
                    choices.append((i, score))

        choices = sorted(choices, key=lambda x: x[1])
        min_score = choices[0][1]
        selected_score = []
        for choice in choices:
            if choice[1] <= min_score * 1.05:
                selected_score.append(choice[0])
            else:
                break
        return selected_score

    chord_to_note_store = {}

    def find_chord_notes(start_tick):
        for i in range(len(Cstart_tick)):
            if start_tick >= Cstart_tick[i] and start_tick < Cend_tick[i]:
                keyx = str(Ckey[i]) + str(Cchord[i])
                if keyx in chord_to_note_store:
                    return chord_to_note_store[keyx]
                result = ChordToNote(Ckey[i], Cchord[i])
                chord_to_note_store[keyx] = result
        return ChordToNote(Ckey[-1], Cchord[-1])

    def harmonize_1(accompany, barinfo):
        # usiung csv file of chord identification
        # if barinfo.starttick >= Cend_tick[0]:
        #     del Cend_tick[0]
        #     del Cstart_tick[0]
        #     del Ckey[0]
        #     del Cchord[0]
        # thiskey = Ckey[0]
        # thischord = Cchord[0]
        notepitches = find_chord_notes(barinfo.starttick)
        # print(notepitches)
        # Assume starting at octave 2 first
        root_note = 36 + notepitches[0]
        notelist = []
        for note in accompany.notelist:
            notelist.append(
                noteMidi(
                    note.relative_interval + root_note,
                    note.start * barinfo.tpb + barinfo.starttick,
                    note.start * barinfo.tpb
                    + barinfo.starttick
                    + note.duration * barinfo.tpb,
                )
            )
        # Move non-chord notes back to chord notes
        for i, note in enumerate(notelist):
            notepitches = find_chord_notes(note.onset)
            if note.pitch % 12 not in notepitches:
                j = 1
                chosenpitch = note.pitch
                x = random.choice([1, -1])
                while True:
                    if (chosenpitch - x * j) % 12 in notepitches:
                        chosenpitch = chosenpitch - x * j
                        break
                    if (chosenpitch + x * j) % 12 in notepitches:
                        chosenpitch = chosenpitch + x * j
                        break
                    j += 1
                notelist[i] = noteMidi(chosenpitch, note.onset, note.offset)
        return notelist

    def harmonize_2(accompany, barinfo):
        # calculate weighted notes from LH info
        notepitches = barinfo.chord_notes()
        # print(notepitches)
        root_note = 36 + notepitches[0]
        notelist = []
        for note in accompany.notelist:
            notelist.append(
                noteMidi(
                    note.relative_interval + root_note,
                    note.start * barinfo.tpb + barinfo.starttick,
                    note.start * barinfo.tpb
                    + barinfo.starttick
                    + note.duration * barinfo.tpb,
                )
            )
        # Move non-chord notes back to chord notes
        for i, note in enumerate(notelist):

            notepitches = barinfo.chord_notes(note.onset)
            if note.pitch % 12 not in notepitches:
                j = 1
                chosenpitch = note.pitch
                x = random.choice([1, -1])
                while True:
                    if (chosenpitch - x * j) % 12 in notepitches:
                        chosenpitch = chosenpitch - x * j
                        break
                    if (chosenpitch + x * j) % 12 in notepitches:
                        chosenpitch = chosenpitch + x * j
                        break
                    j += 1
                notelist[i] = noteMidi(chosenpitch, note.onset, note.offset)
        return notelist

    # %%
    """
    for each bar in the selected bass track:
        find a record in db with same time signature and nearest note
        then, do harmonization based on the chord
        then insert the notes to the midi
        then combine with the melody obtained from skyline!! Yeah.
        #Try on self zoked melody first
    """
    lh_notelist = []
    prev_idx = dict()
    for bar in song_rhythm:
        keyx = str(bar.numerator) + "/" + str(bar.denominator)
        if keyx not in prev_idx:
            prev_idx[keyx] = set()
        idxs = match_rhythm(
            real_database[ts_piece], bar.rhythm, bar.numerator, bar.denominator
        )
        idx = -1
        for c in idxs:
            if c in prev_idx[keyx]:
                idx = c
                break
        if idx == -1:
            idx = random.choice(idxs)
        prev_idx[keyx].add(idx)
        if mode == "guess":
            lh_notelist.extend(harmonize_2(real_database[keyx][idx], bar))
        else:
            lh_notelist.extend(harmonize_1(real_database[keyx][idx], bar))
    rh_notelist = skyline_melody(piece)

    mido_out = mid_parser.MidiFile()
    mido_out.ticks_per_beat = tpb
    track1 = ct.Instrument(program=0, is_drum=False, name="righthand")
    track2 = ct.Instrument(program=0, is_drum=False, name="lefthand")
    mido_out.instruments = [track1, track2]
    for note in rh_notelist:
        mido_out.instruments[0].notes.append(
            ct.Note(
                start=int(note.onset),
                end=int(note.offset),
                pitch=note.pitch,
                velocity=50,
            )
        )
    for note in lh_notelist:
        mido_out.instruments[1].notes.append(
            ct.Note(
                start=int(note.onset),
                end=int(note.offset),
                pitch=note.pitch,
                velocity=50,
            )
        )
    mido_out.dump(out)
