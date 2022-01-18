from gym import Env
from gym.spaces import Discrete, Box
import math
from itertools import combinations
from datetime import datetime
from music21 import *
import numpy as np

class SegmentationEnv(Env): 
    def __init__(self, pieces):
        #Preprocess the pieces
        self.notes = []
        self.offset = []
        self.beat = []
        self.duration = []
        self.octave = []
        self.is_segment = []
#         self.beatchanges = []
        for piece in pieces:
            xnotes = []
            xoffset = []
            xbeat = []
            xduration = []
            xoctave = []
            xcoroffset = []
            xissegment = []
            firstlyric = True
            print(piece)
            c = converter.parse(piece)
            post = c.flattenParts().flat
            for note in post.notes:
                duration = note.duration.quarterLength
                offset = note.offset
                beat = float(note.beat)
                if note.lyric is not None:
                    if firstlyric:
                        xsegment = False
                        firstlyric = False
                    else:
                        xsegment = True
                else:
                    xsegment = False
                allnotes = list(note.pitches)
                for note1 in allnotes:
                    xnotes.append(note1.name)
                    xoffset.append(offset)
                    xbeat.append(beat)
                    xduration.append(duration)
                    xoctave.append(note1.octave)
                    xissegment.append(xsegment)
            self.notes.append(xnotes)
            self.offset.append(xoffset)
            self.beat.append(xbeat)
            self.duration.append(xduration)
            self.octave.append(xoctave)
            self.is_segment.append(xissegment)
            #             xbeatchange = {}
#             for ts in post.recurse().getElementsByClass(meter.TimeSignature):
#                 assert ts.denominator in [2,4,8]
#                 if ts.denominator == 2:
#                     xbeatchange[ts.offset] = 2
#                 elif ts.denominator == 4:
#                     xbeatchange[ts.offset] = 1
#                 else:
#                     xbeatchange[ts.offset] = 0.5
#             self.beatchanges.append(xbeatchange)

        #Actions: Remain segment (0), segment (1)
        self.action_space = Discrete(2)
        
        #Observations: First dim 12 pitch classes, Second dim Octave (1-7), Value is total duration.
        self.observation_space = Box(
            low=np.zeros((12*7+1,),dtype=np.float32),
            high=np.ones((12*7+1,),dtype=np.float32)
        )
        
        #internal state: check where the time currently is 
        self.current_piece = 0
        self.current_noteoffset = 0
        self.notelistfirst = 0
        self.notelistlast = 0
        self.latestbeatfirst = 0
        self.latestbeatlast = 0
        self.isCorrectSegment = False
        # self.state = np.zeros((12,7))
        
        #save segmentation for rendering purposes
        self.determined_offset = []
        print("Total number of pieces",len(self.notes))

    def step(self, action):
        #Calculating reward
        if action == 0: # do nothing
            is_segment = False
            if not self.isCorrectSegment or self.current_noteoffset == 0:
                reward = 1
            else:
                reward = -1#max(-self.change_in_roughness()/20,-1)
        else: # segmentation
            is_segment = True
            if self.current_noteoffset == 0: #illegal operations
                reward = -1
            else:
                self.determined_offset.append((self.current_piece,self.current_noteoffset))
                if self.isCorrectSegment:
                    reward = 1
                else:
                    reward = -1    
        #determine new obs state
        if is_segment and self.current_noteoffset != 0:
            self.notelistfirst = self.latestbeatfirst
        done = False
        if self.latestbeatlast >= len(self.beat[self.current_piece]): #Finished a piece
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
            self.current_noteoffset = self.offset[self.current_piece][self.latestbeatlast]
            currentbeat = self.beat[self.current_piece][self.latestbeatlast]//1
            currentindex = self.latestbeatlast + 1
            self.latestbeatfirst = self.latestbeatlast
            while len(self.beat[self.current_piece]) > currentindex and self.beat[self.current_piece][currentindex]//1 == currentbeat:
                if self.is_segment[self.current_piece][currentindex]:
                    self.isCorrectSegment = True
                currentindex += 1
            self.notelistlast = currentindex
            self.latestbeatlast = currentindex
        info = {}
        return self.staterender(done), reward, done, info

    def render(self):
        # print("Current piece:",self.current_piece)
        print("Current notelist:",self.notelistfirst,self.notelistlast)
        for segment in self.determined_offset:
            print(segment)
        return
    
    def change_in_roughness(self):
        def roughness(notes):
            '''
            Calculate the Roughness of notes according to sum of ideal ratio N+M
            Reference: https://www.researchgate.net/publication/276905584_Measuring_Musical_Consonance_and_Dissonance
            '''
            def interval_to_ratio(interval):
                interval_ratio_mapping = {
                    0:1+1,
                    1:18+17,
                    2:9+8,
                    3:6+5,
                    4:5+4,
                    5:4+3,
                    6:17+12,
                    7:3+2,
                    8:8+5,
                    9:5+3,
                    10:16+9,
                    11:17+9,
                }
                interval_pitch_mapping = {
                    1:0,
                    2:2,
                    3:4,
                    4:5,
                    5:7,
                    6:9,
                    7:11,
                    8:12
                }
                ans = interval_pitch_mapping[int(interval[-1])]
                if int(interval[-1]) in [4,5,8]:
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
                ans = ans%12
                return interval_ratio_mapping[ans]
            ans = 0
            for combo in combinations(notes,2):
                n1 = note.Note(combo[0])
                n2 = note.Note(combo[1])
                xinterval = interval.Interval(noteStart=n1,noteEnd=n2)
                ans += interval_to_ratio(xinterval.semiSimpleName)
            return ans/len(notes) if len(notes)!= 0 else 0
        notelist1 = []
        for i in range(self.notelistfirst,self.latestbeatfirst):
            notelist1.append(self.notes[self.current_piece][i]+str(self.octave[self.current_piece][i]))
        notelist2 = notelist1.copy()
        for i in range(self.latestbeatfirst,self.latestbeatlast):
            notelist2.append(self.notes[self.current_piece][i]+str(self.octave[self.current_piece][i]))
        notelist1 = list(dict.fromkeys(notelist1))
        notelist2 = list(dict.fromkeys(notelist2))
        return roughness(notelist2)-roughness(notelist1) if len(notelist1) != 0 else 0
    
    def staterender(self,done):
        pitch_to_index = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        obsarray = np.zeros((12,7))
        notelist = []
        if done:
            return np.append(obsarray.flatten(),[0])
        for idx in range(self.notelistfirst,self.notelistlast):
            current_note = self.notes[self.current_piece][idx]
            notelist.append(current_note)
            current_duration = self.duration[self.current_piece][idx]
            current_octave = self.octave[self.current_piece][idx]
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
            obsarray[pitchindex][current_octave] += current_duration
            obsarray[pitchindex][current_octave] = min(30,obsarray[pitchindex][current_octave])
#         print(notelist)
        obsarray = obsarray/30
        return np.append(obsarray.flatten(),[min(max(self.change_in_roughness()/20,1),0)])

    def reset(self,idx = 0):
        self.current_piece = idx
        self.current_noteoffset = 0
        self.notelistfirst = 0
        self.notelistlast = 0 #exclusive
        self.latestbeatfirst = 0
        self.latestbeatlast = 0 #exclusive
        self.isCorrectSegment = False
        if self.is_segment[self.current_piece][self.latestbeatlast]:
            self.isCorrectSegment = True
        currentbeat = self.beat[self.current_piece][self.latestbeatlast]//1
        currentindex = self.latestbeatlast + 1
        while len(self.beat[self.current_piece]) > currentindex and self.beat[self.current_piece][currentindex]//1 == currentbeat:
            if self.is_segment[self.current_piece][currentindex]:
              self.isCorrectSegment = True
            currentindex += 1
        self.notelistlast = currentindex
        self.latestbeatlast = currentindex
        return self.staterender(False)