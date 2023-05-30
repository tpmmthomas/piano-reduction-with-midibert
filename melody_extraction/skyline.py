

from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from mido import MidiFile
import time
import operator
import math


# In[4]:


# from music21 import *
# c = converter.parse(piece) #Also OK, check how to process


# In[5]:


# https://stackoverflow.com/questions/63105201/python-mido-how-to-get-note-starttime-stoptime-track-in-a-list
# https://dev.to/varlen/editing-midi-files-with-python-2m0g (checking how to get back this)


# In[6]:


import numpy as np


class noteMidi:
    def __init__(self, p, s, e):
        self.pitch = p
        self.onset = s
        self.offset = e


def mergeIntervals(arr):
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


def gettop(note, intervals):
    note_interval = [note.onset, note.offset]
    overlap_time = 0
    total_time = note.offset - note.onset
    if total_time == 0:
        return 1  # (we do not need this note)
    for interval in intervals:
        maxstart = max(note_interval[0], interval[0])
        minend = min(note_interval[1], interval[1])
        if maxstart < minend:
            overlap_time += minend - maxstart
    return overlap_time / total_time


def skyline(notes):  # revised skyline algorithm by Chai, 2000
    # Performed on a single channel
    accepted_notes = []
    notes = sorted(notes, key=lambda x: x.pitch, reverse=True)
    intervals = []
    for note in notes:
        if gettop(note, intervals) <= 0.5:
            accepted_notes.append(note)
            intervals.append([note.onset, note.offset])
            intervals = mergeIntervals(intervals)
    return sorted(accepted_notes, key=lambda x: x.onset)


def skyline2(notes):  # revised skyline algorithm by Chai, 2000
    # Performed on a single channel
    accepted_notes = []
    rejected_notes = []
    notes = sorted(notes, key=lambda x: x.pitch, reverse=True)
    intervals = []
    for note in notes:
        if gettop(note, intervals) <= 0.5:
            accepted_notes.append(note)
            intervals.append([note.onset, note.offset])
            intervals = mergeIntervals(intervals)
        else:
            rejected_notes.append(note)
    intervals = []
    for note in rejected_notes:  # allow at most 2 line
        if gettop(note, intervals) <= 0.5:
            accepted_notes.append(note)
            intervals.append([note.onset, note.offset])
            intervals = mergeIntervals(intervals)
    return sorted(accepted_notes, key=lambda x: x.onset)


# In[7]:


# # channel_no = 6
# notelist = []
# for inst in mido_obj.instruments:
#     for notes in inst.notes:
#         notelist.append(noteMidi(notes.pitch,notes.start,notes.end))
# notelist_new = skyline(notelist)
# print(len(notelist),len(notelist_new))


# In[8]:


# Helper function for calculating Eucliedean Distance
# Works with list as well as properly indexed dictionary
def dist(v1, v2):
    assert len(v1) == len(v2)
    tmp = 0
    for dim in range(len(v1)):
        tmp += (v1[dim] - v2[dim]) ** 2
    return math.sqrt(tmp)


def cluster_dist(c1, c2, histo):
    dict1 = dict()
    dict2 = dict()
    for i in range(12):
        dict1[i] = 0
        dict2[i] = 0
    for ind in c1:
        for k in histo[ind]:
            dict1[k] += histo[ind][k]
    for ind in c2:
        for k in histo[ind]:
            dict2[k] += histo[ind][k]
    return dist(dict1, dict2)


# In[21]:


# Implementation of the paper: Melody extraction on MIDI music files (Ozcan etal,2005)
def skyline_melody(piece):
    # piece = "bwv1067/MENUET.mid"
    mido_obj = mid_parser.MidiFile(piece)
    tpb = mido_obj.ticks_per_beat
    all_notes = []  # This will be a 2D list of notes for each channel
    for inst in mido_obj.instruments:
        print(inst)
        # ignore percussion channel
        if inst.is_drum:
            continue
        notelist = []
        for notes in inst.notes:
            notelist.append(noteMidi(notes.pitch, notes.start, notes.end))
        notelist = skyline(notelist)
        if len(notelist) > 0:
            all_notes.append(notelist)
    # Calculate a_i, b_i and x_i (refer to paper P.5)
    pitch_histogram = []
    x = []
    for notelist in all_notes:
        total_pitch = 0
        pitchrange = dict()
        for note1 in notelist:
            total_pitch += note1.pitch
            if note1.pitch in pitchrange:
                pitchrange[note1.pitch] += 1
            else:
                pitchrange[note1.pitch] = 1
        avg_pitch = total_pitch / len(notelist)  # This is a_i
        entropy = 0
        pitchhist = dict()
        for i in range(12):
            pitchhist[i] = 0
        for distinct_pitch in pitchrange:
            probit = pitchrange[distinct_pitch] / len(notelist)
            entropy += probit * math.log(probit, 128)
            pitchhist[distinct_pitch % 12] += pitchrange[distinct_pitch]
        entropy = -entropy  # This is b_i
        x.append(avg_pitch + 128 * entropy)  # This is x_i
        pitch_histogram.append(pitchhist)  # Pitch histogram normalized to 12 pitches
    # Calculate Clustering Threshold
    # First Calculate average histogram h_A
    avg_histogram = dict()
    for i in range(12):
        avg_histogram[i] = 0
    for ph in pitch_histogram:
        for i in range(12):
            avg_histogram[i] += ph[i]
    weighted_avg_histogram = avg_histogram.copy()
    total_pitch = 0
    for i in range(12):
        total_pitch += avg_histogram[i]
        avg_histogram[i] /= len(pitch_histogram)
    # IS THIS CORRECT??????????? (From paper equation 14)
    for i in range(12):
        weighted_avg_histogram[i] = weighted_avg_histogram[i] * (
            weighted_avg_histogram[i] / total_pitch
        )
    # Threshold
    t = dist(avg_histogram, weighted_avg_histogram)  # /2
    print("Threshold", t)
    clusters = [[i] for i in range(len(all_notes))]  # index based to denote channels
    is_changed = True
    # Agglomerative Clustering: just use Naive O(n^2) method since n<=16
    # Here we
    while is_changed:
        is_changed = False
        min_dist = t
        min_c1, min_c2 = None, None
        for c1 in clusters:
            for c2 in clusters:
                if c1 != c2:
                    new_dist = cluster_dist(c1, c2, pitch_histogram)
                    if new_dist < min_dist:
                        min_dist = new_dist
                        min_c1 = c1
                        min_c2 = c2
        if min_c1 is not None:
            is_changed = True
            new_cluster = min_c1.copy()
            new_cluster.extend(min_c2)
            clusters.remove(min_c1)
            clusters.remove(min_c2)
            clusters.append(new_cluster)
    print("Resultant Cluster:", clusters)
    clustered_notelist = []
    # Select melody channels in each cluster and group them together
    for cluster in clusters:
        max_x = 0
        melody_channel = None
        for ind in cluster:
            if x[ind] > max_x:
                max_x = x[ind]
                melody_channel = ind
                clustered_notelist.extend(all_notes[ind])
        # clustered_notelist.extend(all_notes[melody_channel])
        print(f"channel {ind} selected as melody channel.")
    final_notelist = skyline(clustered_notelist)
    # final_notelist2 = skyline2(clustered_notelist)
    return final_notelist  # ,final_notelist2