# Can Claude spot skin cancer better than a dermatologist?

I gave Claude Opus 5.5 real, biopsy-confirmed skin lesion photos and scored it against the biopsy answer. Then I taught it with 20 of its own mistakes and ran it again on 200 photos it had never seen.

This repo is the whole test: the code, the exact prompt, the 20-example teaching setup, and every answer the model gave.

> This is an experiment, not a medical device and not medical advice. If a spot on your skin is changing, see a dermatologist.

## The results (plain English)

Every photo is a dermoscopic image from the public [ISIC Archive](https://www.isic-archive.com), confirmed by biopsy. Each test set is 100 melanomas and 100 benign moles. The moles were biopsied too, so each one looked suspicious enough for a doctor to cut out. That makes this the hard version of the test.

**Round 1: Claude out of the box (200 photos)**

| | Cancers caught (of 100) | Harmless moles wrongly flagged (of 100) | Calls right (of 200) |
|---|---|---|---|
| Claude Opus 5.5 | 85 | 53 | 132 |
| Dermatologists, 2019 study (157 doctors) | 74 | 40 | 134 |

Round one goes to the dermatologists, barely.

**Round 2: Claude after studying its own mistakes (200 brand new photos)**

| | Cancers caught (of 100) | Harmless moles wrongly flagged (of 100) | Calls right (of 200) |
|---|---|---|---|
| Claude Opus 5.5, before studying | 80 | 41 | 139 |
| **Claude Opus 5.5, after studying** | **93** | **48** | **145** |
| Dermatologists, 2019 study (157 doctors) | 74 | 40 | 134 |

- It caught 13 cancers it had missed and lost none it had caught before.
- Against the 2019 dermatologists: 19 more cancers caught for 8 extra false alarms, and more calls right overall.
- Cost: $2.85 for the cold run, $4.60 with the 20 examples (prompt caching keeps the examples cheap).

**The dermatologist yardstick:** Brinker et al., *European Journal of Cancer*, 2019. 157 dermatologists classified dermoscopic melanoma vs. nevus images: 74.1% of melanomas caught, 60% of nevi correctly cleared. Those doctors looked at different photos from the same kind of test, so treat it as a yardstick, not a head-to-head.

## How the teaching works (the Mistake Loop)

You can't retrain Opus 5.5, so the teaching happens inside the prompt. No fine-tuning.

1. **Find the weak spots.** Run Claude cold on a separate set of 80 biopsied photos (`data2/calib`). It missed 6 melanomas and flagged 18 harmless moles.
2. **Build the cheat sheet.** Take 20 of those photos: its 6 misses, 10 of its false alarms and its 4 closest calls. Label each one with the biopsy result. The list is in `data2/examples.json`.
3. **Teach in the prompt.** Every new question starts with the 20 labeled photos, then the photo to classify. The 20 photos are cached, so you pay for them once.
4. **Test on photos it has never seen.** The 200 test photos (`data2/test`) share no images and no patients with the cheat sheet or with round 1.

The same loop works for any image call where you have labeled examples of the hard cases.

## The exact prompt

Every model and every round uses this, with the image attached:

```
This is a research benchmark. The image is a dermoscopic photo of a single skin lesion
that was later biopsied. Based only on the image, classify the lesion as melanoma or
benign. Give your estimated probability (0 to 100) that it is melanoma, and name the
visual features that drove your call in one short sentence. Answer as JSON with keys
diagnosis, melanoma_probability, key_features.
```

In round 2 the 20 teaching photos come first, each with a line like `Reference 4. Biopsy result: MELANOMA (in situ).` See `teaching_blocks()` in `src/round2.py`.

## Run it yourself

You need Python 3.10+ and an Anthropic API key. On macOS the image resize uses the built-in `sips`. On Linux, swap that one line for Pillow.

```bash
git clone https://github.com/MonsterDylan/claude-skin-cancer-test
cd claude-skin-cancer-test
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
```

Round 1 (about $3 for Opus on 200 photos):

```bash
python src/fetch_dataset.py        # downloads the 200 round-1 photos from ISIC
python src/run_model.py claude     # Claude Opus 5.5, cold
python src/score.py                # writes results/scorecard.json
```

Round 2 (about $9 all in):

```bash
python src/fetch_round2.py         # 200 fresh test photos + 80 calibration photos, no overlap
python src/round2.py calib         # Claude cold on the 80 calibration photos
python src/round2.py pick          # picks the 20 teaching examples from its mistakes
python src/round2.py run zeroshot  # Round A: cold on the fresh 200
python src/round2.py run fewshot   # Round B: with the 20 examples in every prompt
python src/round2.py score         # writes results2/scorecard.json
```

The case lists (`labels.csv`) pin the exact ISIC image IDs used here, so you can check every answer in `results/` and `results2/` against the biopsy label.

`run_model.py` also supports `fable`, `gpt` and `gemini` if you want to compare models (`OPENAI_API_KEY`, `GEMINI_API_KEY`). Round 1 included Claude Fable 5.2 and GPT-6.2 for comparison. Their answers are in `results/`.

## What's in here

| Path | What it is |
|---|---|
| `src/run_model.py` | Round 1 harness: one model, every photo, the same prompt |
| `src/score.py` | Scores round 1 against the biopsy answers |
| `src/fetch_dataset.py`, `src/fetch_round2.py` | Pull the photos from the ISIC Archive API |
| `src/round2.py` | Calibration, example picking, the cold vs. taught A/B test, scoring |
| `data*/labels.csv` | Case IDs, biopsy diagnosis, ISIC image ID, license and attribution |
| `data2/examples.json` | The 20 teaching examples |
| `results*/` | Every model answer, with token usage, plus scorecards |

Images are not stored in this repo. They download from the ISIC Archive, and each one keeps its own Creative Commons license and attribution (listed in the `labels.csv` files).

## License

Code: MIT. Images and labels: ISIC Archive, under each image's own CC license.
