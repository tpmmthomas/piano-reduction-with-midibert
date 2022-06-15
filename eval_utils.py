from itertools import combinations
import numpy as np

def mergeIntervals(arr):
        # Sorting based on the increasing order 
        # of the start intervals
        arr.sort(key = lambda x: x[0]) 
        # array to hold the merged intervals
        m = []
        s = -10000
        max = -100000
        for i in range(len(arr)):
            a = arr[i]
            if a[0] > max:
                if i != 0:
                    m.append([s,max])
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

def gettop(note,intervals):
    note_interval = [note.start,note.end]
    overlap_time = 0
    total_time = note.end - note.start
    if total_time == 0:
        return 1 #(we do not need this note)
    for interval in intervals:
        maxstart = max(note_interval[0],interval[0])
        minend = min(note_interval[1],interval[1])
        if maxstart < minend:
            overlap_time += minend-maxstart
    return overlap_time/total_time

def skyline(notes,thres): #revised skyline algorithm by Chai, 2000
    #Performed on a single channel
    accepted_notes = []
    notex = sorted(notes, key=lambda x: x.pitch, reverse=True)
    intervals = []
    for note in notex:
        if gettop(note,intervals) <= thres:
            accepted_notes.append(note)
            intervals.append([note.start,note.end])
            intervals = mergeIntervals(intervals)
    return sorted(accepted_notes,key=lambda x: x.start)

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
        return interval_ratio_mapping[interval%12]        
    ans = 0
    count = 0
    for combo in combinations(notes, 2):
        if not is_intersect((combo[0].start,combo[0].end),(combo[1].start,combo[1].end)):
            continue
        interval = abs(combo[0].pitch - combo[1].pitch)
        count += 1
        ans = max(interval_to_ratio(interval) / (18+17),ans)

    return ans

def is_intersect(int1,int2):
    maxstart = max(int1[0],int2[0])
    minend = min(int1[1],int2[1])
    return maxstart < minend

def cosine_similarity(v1,v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))