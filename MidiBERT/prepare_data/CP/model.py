import numpy as np
import pickle
import utils
from tqdm import tqdm
import logging
from skyline import Skyline
from o2p import O2p

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}


def find_intersect(start, end, intervals):
    def is_overlapping(x1, x2, y1, y2):
        return max(x1, y1) < min(x2, y2)

    for inte in intervals:
        if is_overlapping(start, end, inte[0], inte[1]):
            return True
    return False


class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.dict = dict
        self.event2word, self.word2event = pickle.load(open(dict, "rb"))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [
            self.event2word[etype]["%s <PAD>" % etype] for etype in self.event2word
        ]
        # self.EOS = [
        #     self.event2word[etype]["%s <EOS>" % etype] for etype in self.event2word
        # ]

    def extract_events(self, input_path, task):
        if task == "reduction":
            note_items, tempo_items, pianohist = utils.read_items(
                input_path, is_reduction=True
            )
        else:
            note_items, tempo_items = utils.read_items(input_path)
            pianohist = None
        if len(note_items) == 0:
            return [], None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items

        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events, pianohist

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def prepare_data(self, midi_paths, task, max_len):  # path,None,512
        all_words, all_ys = [], []
        if task == "skyline":
            skyline = Skyline(self.dict)
        if task == "o2p":
            o2p = O2p(self.dict)

        for path in midi_paths:
            # extract events
            try:
                logger.info(path)
                if task == "o2p":
                    orch = path + "/orchestra.mid"
                    piano = path + "/piano.mid"
                    events, histp = self.extract_events(orch, task)
                    events2, histp = self.extract_events(piano, task)
                    if len(events2) == 0:
                        logger.info("skipped")
                        continue
                else:
                    events, histp = self.extract_events(path, task)
                if len(events) == 0:
                    continue
                if task == "reduction" and histp is None:
                    print("skipped_nop")
                    continue
                # events to words
                words, ys = [], []
                i = 0
                for note_tuple in events:
                    nts, to_class = [], -1
                    pitch, interval = None, None
                    for e in note_tuple:
                        if e.name == "genLabel":
                            interval = e.value
                            continue
                        e_text = "{} {}".format(e.name, e.value)
                        nts.append(self.event2word[e.name][e_text])
                        if e.name == "Pitch":
                            to_class = e.Type
                            pitch = e.value
                    words.append(nts)
                    if task == "melody" or task == "velocity":
                        ys.append(to_class + 1)

                    if task == "reduction":
                        if pitch not in histp:
                            ys.append(2)
                        elif find_intersect(interval[0], interval[1], histp[pitch]):
                            ys.append(1)
                        else:
                            ys.append(2)
                words2 = []
                if task == "o2p":
                    for note_tuple in events2:
                        nts, to_class = [], -1
                        pitch, interval = None, None
                        for e in note_tuple:
                            if e.name == "genLabel":
                                interval = e.value
                                continue
                            e_text = "{} {}".format(e.name, e.value)
                            nts.append(self.event2word[e.name][e_text])
                            if e.name == "Pitch":
                                to_class = e.Type
                                pitch = e.value
                        words2.append(nts)

                if task == "custom":
                    slice_words = []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i : i + max_len])
                    if len(slice_words[-1]) < max_len:
                        slice_words[-1] = self.padding(
                            slice_words[-1], max_len, ans=False
                        )
                    return np.array(slice_words), None

                if task == "reduction":
                    ysn = np.array(ys)
                    kept_percentage = np.count_nonzero(ysn == 1) / len(ysn)
                    print(kept_percentage)
                    if kept_percentage < 0.4 or kept_percentage > 0.9:
                        print("skipped")
                        continue
                # slice to chunks so that max length = max_len (default: 512)
                if task == "skyline":
                    slice_words, slice_ys = skyline.generate(words)
                elif task == "o2p":
                    try:
                        assert words[0][0] == 0
                        assert words2[0][0] == 0
                    except:
                        print("CRAZY!!!")
                    slice_words, slice_ys = o2p.generate(words, words2)
                else:
                    slice_words, slice_ys = [], []
                    for i in range(0, len(words), max_len):
                        slice_words.append(words[i : i + max_len])
                        if task == "composer":
                            name = path.split("/")[-2]
                            slice_ys.append(Composer[name])
                        elif task == "emotion":
                            name = path.split("/")[-1].split("_")[0]
                            slice_ys.append(Emotion[name])
                        else:
                            slice_ys.append(ys[i : i + max_len])

                    # padding or drop
                    # drop only when the task is 'composer' and the data length < max_len//2
                    if len(slice_words[-1]) < max_len:
                        if task == "composer" and len(slice_words[-1]) < max_len // 2:
                            slice_words.pop()
                            slice_ys.pop()
                        elif task == "skyline":
                            slice_words[-1] = self.padding(
                                slice_words[-1], max_len, ans=False
                            )
                        else:
                            slice_words[-1] = self.padding(
                                slice_words[-1], max_len, ans=False
                            )

                if (
                    task == "melody" or task == "velocity" or task == "reduction"
                ) and len(slice_ys[-1]) < max_len:
                    slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

                all_words = all_words + list(slice_words)
                all_ys = all_ys + list(slice_ys)
            except Exception as e:
                logger.error(e)

        all_words = np.array(all_words).astype(np.int64)
        all_ys = np.array(all_ys).astype(np.int64)

        return all_words, all_ys
