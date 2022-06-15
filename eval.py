# %%
from postprocessing import postprocess

# %%
import glob

files = glob.glob("./objective_eval/algo_unprocessed/*.mid")
len(files)

# %%
target_dir = "./objective_eval/algo_processed/"
import os

for file in files:
    outpath = target_dir + os.path.basename(file)
    postprocess(file, outpath)

# %%
files = glob.glob("./objective_eval/cyclegan_unprocessed/*.mid")
target_dir = "./objective_eval/cyclegan_processed/"
for file in files:
    outpath = target_dir + os.path.basename(file)
    postprocess(file, outpath)

# %%
files = glob.glob("./objective_eval/bertr2f_unprocessed/*.mid")
target_dir = "./objective_eval/bertr2f_processed/"
for file in files:
    outpath = target_dir + os.path.basename(file)
    postprocess(file, outpath)

# %%
files = glob.glob("./objective_eval/bertreduction_unprocessed/*.mid")
target_dir = "./objective_eval/bertreduction_processed/"
for file in files:
    outpath = target_dir + os.path.basename(file)
    postprocess(file, outpath)

# # %%
# from eval_utils import skyline, roughness, is_intersect
# import numpy as np


# def dissonance_metric(notetotal, tpb):
#     note_skyline = skyline(notetotal, 0)
#     current_min = 0
#     rough, weight = [], []
#     for note in note_skyline:
#         current_min = note.start
#         to_remove = []
#         considered = []
#         for i, note2 in enumerate(notetotal):
#             if note2.end < current_min:
#                 to_remove.append(i)
#             if note2.start > note.end:
#                 break
#             if note != note2 and is_intersect(
#                 (note2.start, note2.end), (note.start, note.end)
#             ):
#                 considered.append(note2)
#         to_remove = sorted(to_remove, reverse=True)
#         for idx in to_remove:
#             notetotal.pop(idx)
#         if len(considered) < 2:
#             continue
#         r = roughness(considered)
#         rough.append(r)
#         weight.append((note.end - note.start) / tpb)
#     rough = np.array(rough)
#     weight = np.array(weight)
#     if len(rough) == 0:
#         dissonance = 0
#     else:
#         dissonance = np.dot(rough, weight) / np.sum(weight)
#     print(f"The amount of dissonance is calculated to be {dissonance:.2f}.")
#     return dissonance


# # %%
# from eval_utils import is_intersect, cosine_similarity


# def tonal_distance(ppath, opath):
#     DEFAULT_RESOLUTION = 24
#     add_step = DEFAULT_RESOLUTION
#     sliding_window = [0, add_step * 2]
#     piano, tpbp = load_midi(ppath)
#     orch, tpbo = load_midi(opath)

#     # update tpb
#     for note in piano:  # seems it knows to update in_place
#         note.start = int(note.start / tpbp * DEFAULT_RESOLUTION)
#         note.end = int(note.end / tpbp * DEFAULT_RESOLUTION)
#     for note in orch:
#         note.start = int(note.start / tpbo * DEFAULT_RESOLUTION)
#         note.end = int(note.end / tpbo * DEFAULT_RESOLUTION)

#     dists = []
#     while len(piano) != 0:
#         considered_piano = np.zeros(shape=(12))
#         considered_orch = np.zeros(shape=(12))
#         to_remove_piano = []
#         to_remove_orch = []
#         hv_note_p = False
#         hv_note_o = False
#         for i, pnote in enumerate(piano):
#             if pnote.end < sliding_window[0]:
#                 to_remove_piano.append(i)
#             if pnote.start > sliding_window[1]:
#                 break
#             if is_intersect(sliding_window, (pnote.start, pnote.end)):
#                 # insert a tuple for consideration (duration,pitch_class)
#                 hv_note_p = True
#                 s = max(sliding_window[0], pnote.start)
#                 e = min(sliding_window[1], pnote.end)
#                 dur = e - s
#                 assert dur >= 0
#                 considered_piano[pnote.pitch % 12] += dur
#         to_remove_piano = sorted(to_remove_piano, reverse=True)
#         for idx in to_remove_piano:
#             piano.pop(idx)
#         for i, onote in enumerate(orch):
#             if onote.end < sliding_window[0]:
#                 to_remove_orch.append(i)
#             if onote.start > sliding_window[1]:
#                 break
#             if is_intersect(sliding_window, (onote.start, onote.end)):
#                 # insert a tuple for consideration (duration,pitch_class)
#                 hv_note_o = True
#                 s = max(sliding_window[0], onote.start)
#                 e = min(sliding_window[1], onote.end)
#                 dur = e - s
#                 assert dur >= 0
#                 considered_orch[onote.pitch % 12] += dur
#         to_remove_orch = sorted(to_remove_orch, reverse=True)
#         for idx in to_remove_orch:
#             orch.pop(idx)
#         if hv_note_p and hv_note_o:
#             sim = cosine_similarity(considered_piano, considered_orch)
#             dists.append(sim)
#         sliding_window[0] += add_step
#         sliding_window[1] += add_step
#     dists = np.array(dists)
#     print(
#         f"Cosine Similarity: Max {np.max(dists):.2f} Min {np.min(dists):.2f} Mean {np.mean(dists):.2f} SD {np.std(dists):.2f}"
#     )
#     return np.mean(dists)


# # %%
# algo_dissonance = []
# cyclegan_dissonance = []
# bertr2f_dissonance = []
# bertreduction_dissonance = []
