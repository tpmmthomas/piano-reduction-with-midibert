import pickle

event2word = {
    "Bar": {},
    "Position": {},
    "Pitch": {},
    "Duration": {},
    "Program": {},
    "Time Signature": {},
}
word2event = {
    "Bar": {},
    "Position": {},
    "Pitch": {},
    "Duration": {},
    "Program": {},
    "Time Signature": {},
}


def special_tok(cnt, cls):
    # """event2word[cls][cls+' <SOS>'] = cnt
    # word2event[cls][cnt] = cls+' <SOS>'
    # cnt += 1

    event2word[cls][cls + " <PAD>"] = cnt
    word2event[cls][cnt] = cls + " <PAD>"
    cnt += 1

    event2word[cls][cls + " <MASK>"] = cnt
    word2event[cls][cnt] = cls + " <MASK>"
    cnt += 1

    event2word[cls][cls + " <EOS>"] = cnt
    word2event[cls][cnt] = cls + " <EOS>"
    cnt += 1

    event2word[cls][cls + " <ABS>"] = cnt
    word2event[cls][cnt] = cls + " <ABS>"
    cnt += 1

    event2word[cls][cls + " <BOS>"] = cnt
    word2event[cls][cnt] = cls + " <BOS>"
    cnt += 1


# Bar
cnt, cls = 0, "Bar"
event2word[cls]["Bar New"] = cnt
word2event[cls][cnt] = "Bar New"
cnt += 1

event2word[cls]["Bar Continue"] = cnt
word2event[cls][cnt] = "Bar Continue"
cnt += 1
special_tok(cnt, cls)

# =============================================
# Position
# 16 beats -> 24 beats -> 48 beats
cnt, cls = 0, "Position"
for i in range(1, 49):
    event2word[cls][f"Position {i}/48"] = cnt
    word2event[cls][cnt] = f"Position {i}/48"
    cnt += 1

special_tok(cnt, cls)
# =============================================

# Note On
cnt, cls = 0, "Pitch"
for i in range(22, 108):
    event2word[cls][f"Pitch {i}"] = cnt
    word2event[cls][cnt] = f"Pitch {i}"
    cnt += 1

special_tok(cnt, cls)

# Note Duration
cnt, cls = 0, "Duration"
for i in range(12 * 4 * 4): # MAX 4 bars
    event2word[cls][f"Duration {i}"] = cnt
    word2event[cls][cnt] = f"Duration {i}"
    cnt += 1

special_tok(cnt, cls)

# =============================================
# Program
cnt, cls = 0, "Program"
for i in range(97):  # ignore sound effects and ethnic instruments
    # The value 96 reserved for a special input in the finetuning stage.
    event2word[cls][f"Program {i}"] = cnt
    word2event[cls][cnt] = f"Program {i}"
    cnt += 1

special_tok(cnt, cls)
# =============================================


# =============================================
## Customized Tokens, summer 2022
# Simple Time Signature
supported_time_signature = ["24", "34", "44"]

cnt, cls = 0, "Time Signature"
for i in supported_time_signature:
    event2word[cls][f"Time Signature {i}"] = cnt
    word2event[cls][cnt] = f"Time Signature {i}"
    cnt += 1

special_tok(cnt, cls)
# =============================================


# print(event2word)
# print(word2event)
t = (event2word, word2event)

with open("./CP_program.pkl", "wb") as f:
    pickle.dump(t, f)
