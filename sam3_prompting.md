# SAM 3 Prompting Options

How prompts work in the SAM 3 video predictor used by this repo, and whether a
text/concept prompt can be combined with click points.

This documents the behavior of the SAM 3 build actually installed in the UI venv
(`git+https://github.com/facebookresearch/sam3.git`, see
`scripts/requirements-sam3-ui.txt`). The production wrapper is
`demo/sam3_wrapper_hf.py`.

## TL;DR

- SAM 3 supports **text/concept**, **point**, **box**, and **mask/exemplar**
  prompts.
- The request dispatcher accepts a `text` field on `add_prompt`, but the model
  forbids combining **text** with **points** in a single prompt. It is text/box
  (semantic) **or** points (tracker instance), never both at once.
- This repo's wrapper currently sends **points only** and never sets `text`.

## Prompt types

Per the official SAM 3 README and project page, the model is "Segment Anything
with Concepts" and accepts:

| Type | What it is | Frame scope | Mechanism in code |
| --- | --- | --- | --- |
| Text / concept | Open-vocabulary phrase, e.g. `"person"`, segments **all** instances | Applies to all frames; **resets state** when added | semantic path |
| Box | `xywh` box prompt | Per-frame | semantic path |
| Point | Positive/negative clicks, refine a tracked instance | Per-frame, tied to an `obj_id` | tracker-instance path |
| Mask / exemplar | Image exemplar / mask conditioning | — | semantic path |

## The key constraint: text and points are mutually exclusive

The shared request dispatcher does expose a `text` field (it is simply unused by
this repo's wrapper):

```python
# sam3/model/sam3_base_predictor.py  (handle_request -> add_prompt)
elif request_type == "add_prompt":
    return self.add_prompt(
        ...,
        text=request.get("text", None),
        points=request.get("points", None),
        ...
    )
```

The official README's own example uses it:

```python
predictor.handle_request(
    {"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": "person"}
)
```

But the underlying model `add_prompt` hard-asserts that points cannot be mixed
with text/box:

```python
# sam3/model/sam3_video_inference.py  (model.add_prompt)
if points is not None:
    # tracker-instance prompts
    assert text_str is None and boxes_xywh is None, (
        "When points are provided, text_str and boxes_xywh must be None.")
    assert obj_id is not None, (
        "When points are provided, obj_id must be provided.")
    return self.add_tracker_new_points(...)
else:
    # SAM3 semantic prompts (text / box)
    return super().add_prompt(..., text_str=text_str, boxes_xywh=...)
```

Why they don't mix in one call:

- **Text** is a *semantic / concept* prompt. It is **not frame-specific** (it
  applies across all frames) and it **resets session state** when added
  (`reset_state` in the semantic path).
- **Points** are *tracker-instance* prompts, attached to a specific `obj_id` on a
  specific frame. This is the path the production wrapper uses today via
  `_add_prompt` (`demo/sam3_wrapper_hf.py`).

## Options for this app

### Option A — keep points only (current behavior)

No change. Artists click positive/negative points on keyframes; masks propagate.
See `track_video_from_dir()` in `demo/sam3_wrapper_hf.py`.

### Option B — add a text/concept (or box) prompt path

Thread a `text` kwarg into `_add_prompt` and send it on a frame with **no**
points. Good for "segment all `<concept>`" without clicking. Single call, simple.

### Option C — text to seed, then points to refine

Possible only as **separate** `add_prompt` calls, not one combined prompt — and
note the text call **resets state**, so ordering and re-prompting must be handled
deliberately. This is more involved than Option B and does not map onto the
current single-pass point flow.

## Notes specific to this repo

- Default version is `sam3` (base), not `sam3.1` (multiplex), per `.videomama-env`
  and the `SAM3VideoTracker` default. Override with `SAM3_MODEL_VERSION=sam3.1`.
- The wrapper's `_add_prompt` only ever passes `points` / `point_labels`; adding
  `text` support is a localized change to `demo/sam3_wrapper_hf.py` plus a UI
  field in `demo/production_frame_app.py`.

## Official references

- Project page: https://ai.meta.com/sam3/
- Paper ("SAM 3: Segment Anything with Concepts"):
  https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- Code: https://github.com/facebookresearch/sam3
- Checkpoints: https://huggingface.co/facebook/sam3 and
  https://huggingface.co/facebook/sam3.1
- Interactive demo: https://segment-anything.com/

Code references in the installed package (UI venv,
`/tmp/videomama-sam3-ui-venv/.../site-packages/sam3/`):

- `model/sam3_base_predictor.py` — `handle_request` / `add_prompt` dispatcher
  (accepts `text`, `points`, `point_labels`, `bounding_boxes`).
- `model/sam3_video_inference.py` — model `add_prompt`; enforces the
  text/box-vs-points mutual exclusivity.
- `model_builder.py` — `build_sam3_predictor(version=...)` routing for `sam3` vs
  `sam3.1`.
