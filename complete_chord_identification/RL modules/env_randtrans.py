from gym import Env
from gym.spaces import Discrete, Box
import math
from itertools import combinations
from datetime import datetime
from music21 import *
import numpy as np
from m21preprocess import preprocessing, random_transpose


class SegmentationEnv(Env):
    def __init__(self, pieces):
        # Preprocess the pieces
        self.notes = []
        self.offset = []
        self.beat = []
        self.duration = []
        self.octave = []
        self.is_segment = []
        self.piecelist = []
        self.curnotes = []
        self.curoctave = []
        #         self.beatchanges = []
        for piece in pieces:
            getlist = preprocessing(piece, to_trans=False)
            if getlist is None:
                continue
            xnotes, xoffset, xbeat, xduration, xoctave, xissegment = getlist
            self.piecelist.append(piece)
            self.notes.append(xnotes)
            self.offset.append(xoffset)
            self.beat.append(xbeat)
            self.duration.append(xduration)
            self.octave.append(xoctave)
            self.is_segment.append(xissegment)

        # Actions: Remain segment (0), segment (1)
        self.action_space = Discrete(2)

        # Observations: First dim 12 pitch classes, Second dim Octave (1-7), Value is total duration.
        self.observation_space = Box(
            low=np.zeros((12 * 2 + 1,), dtype=np.float32),
            high=np.ones((12 * 2 + 1,), dtype=np.float32),
        )

        # internal state: check where the time currently is
        self.current_piece = 0
        self.current_noteoffset = 0
        self.notelistfirst = 0
        self.notelistlast = 0
        self.latestbeatfirst = 0
        self.latestbeatlast = 0
        self.nextbeatfirst = 0
        self.nextbeatlast = 0
        self.isCorrectSegment = False
        # self.state = np.zeros((12,7))

        # save segmentation for rendering purposes
        self.determined_offset = []
        print("Total number of pieces", len(self.notes))

    def step(self, action):
        # Calculating reward
        if action == 0:  # do nothing
            is_segment = False
            if not self.isCorrectSegment or self.current_noteoffset == 0:
                reward = 1
            else:
                reward = -1  # max(-self.change_in_roughness()/20,-1)
        else:  # segmentation
            is_segment = True
            if self.current_noteoffset == 0:  # illegal operations
                reward = -1
            else:
                self.determined_offset.append(
                    (self.current_piece, self.current_noteoffset)
                )
                if self.isCorrectSegment:
                    reward = 1
                else:
                    reward = -1
        # determine new obs state
        if is_segment and self.current_noteoffset != 0:
            self.notelistfirst = self.latestbeatfirst
        done = False
        if self.latestbeatlast >= len(
            self.beat[self.current_piece]
        ):  # Finished a piece
            self.current_piece += 1
            done = True
            # if self.current_piece == len(self.notes):
            #     done = True
            # else:
            #     done = False
            #     self.current_noteoffset = 0
            #     self.notelistfirst = 0
            #     self.notelistlast = 0
            #     self.latestbeatfirst = 0
            #     self.latestbeatlast = 0
        if not done:
            self.isCorrectSegment = False
            if self.is_segment[self.current_piece][self.latestbeatlast]:
                self.isCorrectSegment = True
            self.current_noteoffset = self.offset[self.current_piece][
                self.latestbeatlast
            ]
            currentbeat = self.beat[self.current_piece][self.latestbeatlast] // 1
            currentindex = self.latestbeatlast + 1
            self.latestbeatfirst = self.latestbeatlast
            while (
                len(self.beat[self.current_piece]) > currentindex
                and self.beat[self.current_piece][currentindex] // 1 == currentbeat
            ):
                if self.is_segment[self.current_piece][currentindex]:
                    self.isCorrectSegment = True
                currentindex += 1
            self.notelistlast = currentindex
            self.latestbeatlast = currentindex
            self.nextbeatfirst = currentindex
            if len(self.beat[self.current_piece]) >= currentindex:
                self.nextbeatlast = currentindex
            else:
                currentbeat = self.beat[self.current_piece][currentindex] // 1
                currentindex = currentindex + 1
                while (
                    len(self.beat[self.current_piece]) > currentindex
                    and self.beat[self.current_piece][currentindex] // 1 == currentbeat
                ):
                    currentindex += 1
                self.nextbeatlast = currentindex
        info = {}
        return self.staterender(done), reward, done, info

    def render(self):
        # print("Current piece:",self.current_piece)
        print("Current notelist:", self.notelistfirst, self.notelistlast)
        for segment in self.determined_offset:
            print(segment)
        return

    def change_in_roughness(self):
        def roughness(notes):
            """
            Calculate the Roughness of notes according to sum of ideal ratio N+M
            Reference: https://www.researchgate.net/publication/276905584_Measuring_Musical_Consonance_and_Dissonance
            """

            def interval_to_ratio(interval):
                interval_ratio_mapping = {
                    0: 1 + 1,
                    1: 18 + 17,
                    2: 9 + 8,
                    3: 6 + 5,
                    4: 5 + 4,
                    5: 4 + 3,
                    6: 17 + 12,
                    7: 3 + 2,
                    8: 8 + 5,
                    9: 5 + 3,
                    10: 16 + 9,
                    11: 17 + 9,
                }
                interval_pitch_mapping = {
                    1: 0,
                    2: 2,
                    3: 4,
                    4: 5,
                    5: 7,
                    6: 9,
                    7: 11,
                    8: 12,
                }
                ans = interval_pitch_mapping[int(interval[-1])]
                if int(interval[-1]) in [4, 5, 8]:
                    intname = interval[:-1]
                    if intname == "dd":
                        ans -= 2
                    elif intname == "d":
                        ans -= 1
                    elif intname == "A":
                        ans += 1
                    elif intname == "AA":
                        ans += 2
                else:
                    intname = interval[:-1]
                    if intname == "m":
                        ans -= 1
                    elif intname == "d":
                        ans -= 2
                    elif intname == "A":
                        ans += 1
                    elif intname == "AA":
                        ans += 2
                ans = ans % 12
                return interval_ratio_mapping[ans]

            for combo in combinations(notes, 2):
                try:
                    n1 = note.Note(combo[0])
                    n2 = note.Note(combo[1])
                    xinterval = interval.Interval(noteStart=n1, noteEnd=n2)
                    ans += interval_to_ratio(xinterval.semiSimpleName)
                except:
                    continue
            return ans / len(notes) if len(notes) != 0 else 0

        notelist1 = []
        for i in range(self.notelistfirst, self.latestbeatfirst):
            notelist1.append(self.curnotes[i] + str(self.curoctave[i]))
        notelist2 = notelist1.copy()
        for i in range(self.latestbeatfirst, self.latestbeatlast):
            notelist2.append(self.curnotes[i] + str(self.curoctave[i]))
        notelist1 = list(dict.fromkeys(notelist1))
        notelist2 = list(dict.fromkeys(notelist2))
        return roughness(notelist2) - roughness(notelist1) if len(notelist1) != 0 else 0

    def staterender(self, done):
        pitch_to_index = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        octave_weight = [1.25, 1.25, 1.1, 1, 0.9, 0.8, 0.7]
        obsarray = np.zeros((2, 12))
        if done:
            return np.append(obsarray.flatten(), [0])
        for idx in range(self.notelistfirst, self.notelistlast):
            current_note = self.curnotes[idx]
            current_duration = self.duration[self.current_piece][idx]
            current_octave = self.curoctave[idx]
            pitchindex = pitch_to_index[current_note[0]]
            current_note = current_note[1:]
            if current_note == "#":
                pitchindex += 1
            elif current_note == "##":
                pitchindex += 2
            elif current_note == "-":
                pitchindex -= 1
            elif current_note == "--":
                pitchindex -= 2
            pitchindex = pitchindex % 12
            if current_octave < 1 or current_octave > 7:
                continue
            current_octave -= 1
            obsarray[0][pitchindex] += current_duration * octave_weight[current_octave]
            obsarray[0][pitchindex] = min(30, obsarray[0][pitchindex])
        for idx in range(self.nextbeatfirst, self.nextbeatlast):
            current_note = self.curnotes[idx]
            current_duration = self.duration[self.current_piece][idx]
            current_octave = self.curoctave[idx]
            pitchindex = pitch_to_index[current_note[0]]
            current_note = current_note[1:]
            if current_note == "#":
                pitchindex += 1
            elif current_note == "##":
                pitchindex += 2
            elif current_note == "-":
                pitchindex -= 1
            elif current_note == "--":
                pitchindex -= 2
            pitchindex = pitchindex % 12
            if current_octave < 1 or current_octave > 7:
                continue
            current_octave -= 1
            obsarray[1][pitchindex] += current_duration * octave_weight[current_octave]
            obsarray[1][pitchindex] = min(30, obsarray[1][pitchindex])
        obsarray = obsarray / 30
        return np.append(
            obsarray.flatten(), [min(max(self.change_in_roughness() / 20, 1), 0)]
        )

    def reset(self, idx=0):
        self.current_piece = idx
        self.curnotes, self.curoctave = random_transpose(
            self.notes[idx], self.octave[idx]
        )
        self.current_noteoffset = 0
        self.notelistfirst = 0
        self.notelistlast = 0  # exclusive
        self.latestbeatfirst = 0
        self.latestbeatlast = 0  # exclusive
        self.isCorrectSegment = False
        if self.is_segment[self.current_piece][self.latestbeatlast]:
            self.isCorrectSegment = True
        currentbeat = self.beat[self.current_piece][self.latestbeatlast] // 1
        currentindex = self.latestbeatlast + 1
        while (
            len(self.beat[self.current_piece]) > currentindex
            and self.beat[self.current_piece][currentindex] // 1 == currentbeat
        ):
            if self.is_segment[self.current_piece][currentindex]:
                self.isCorrectSegment = True
            currentindex += 1
        self.notelistlast = currentindex
        self.latestbeatlast = currentindex
        self.nextbeatfirst = currentindex
        if len(self.beat[self.current_piece]) >= currentindex:
            self.nextbeatlast = currentindex
        else:
            currentbeat = self.beat[self.current_piece][currentindex] // 1
            currentindex = currentindex + 1
            while (
                len(self.beat[self.current_piece]) > currentindex
                and self.beat[self.current_piece][currentindex] // 1 == currentbeat
            ):
                currentindex += 1
            self.nextbeatlast = currentindex

        return self.staterender(False)
