---
name: qwen-image-character
description: Generate a full character design sheet (turnaround views, silhouettes, expression matrix, micro-expressions, head angles, poses, close-up, outfit details, hand gestures, color palette) for one consistent character with Qwen-Image-2.1 on the local GPU job server, optionally locked to reference images, using the bundled client script. Use when the user asks for a character sheet, model sheet, character reference sheet or character design board.
---

# Qwen-Image character sheet

Fills the character sheet template with the user's character settings and renders it at 4:3 (2400×1792) on the
Qwen-Image-2.1 job server, optionally conditioned on reference images of the character.

The client is [scripts/qwen_image.py](scripts/qwen_image.py): Python standard library only, run with `python3`
from a shell. Below, `QI` stands for `python3 <this skill's folder>/scripts/qwen_image.py`. Every command prints
one JSON object. Server URL: `QWEN_IMAGE_SERVER` (default `http://127.0.0.1:8000`). PNGs are saved to
`QWEN_IMAGE_OUTPUT_DIR` (default `./qwen_images`) unless `--out` is given.

First run `QI health`. If it returns an `error`, tell the user to start `qwen_image_server.py` on the GPU machine.

## 1. Collect the settings

| Placeholder | Meaning | If the user did not say |
|---|---|---|
| `style` | Realistic 3D / Stylized 3D / Anime / Semi-realistic / IP Design | Ask, or pick from the request |
| `description` | Appearance: face, hair, build, every garment with colour and material, accessories | Ask; it drives everything |
| `gender` | Male / Female / Neutral | Infer from the description or ask |
| `age` | A number | Ask |
| `body_type` | Slender / Standard / Muscular / Exaggerated Proportions | Standard |
| `style_keywords` | e.g. High-end, Fashion, Trendy, Sci-Fi, Expressive | Pick 2–4 that fit |
| `name`, `role`, `personality` (3–5 words), `theme` (one sentence) | Header fields | `auto-generate` |
| reference images | 0–10 local image paths of the character | None |

Reference images give by far the strongest identity lock and realism. For a photorealistic sheet without references,
suggest first making a photographic portrait with the `qwen-image-t2i` skill and using it as the reference.

## 2. Fill the template

Read [references/character_sheet.txt](references/character_sheet.txt) and replace every `{placeholder}` with plain
text. The rules below matter because the model draws prompt text almost literally:

- `{style}`, `{description}`, `{gender}`, `{age}`, `{body_type}`, `{style_keywords}`: the values as given.
- `{subject}`: `a <age>-year-old <noun>`. Nouns: Male → man, Female → woman, Neutral → androgynous person; under 18:
  boy, girl, androgynous teen. Stating this in the task line is required: with only the `Gender:` line, a female
  character was drawn male.
- `{reference_note}`: with N references: `Use the N attached reference image(s) as the only identity source.`
  With none: `No reference image is attached: design the character from the Basic Settings below, then keep that
  exact character identical in every panel.`
- `{name}`, `{role}`, `{personality}`, `{theme}`: `: <value>` (colon, space, value) right after the label. For
  `auto-generate`, use an empty string, so the label stands alone and the model fills it in. Never write an
  instruction such as "invent a name" as a value; it gets drawn onto the sheet.
- Descriptions should be content, not directions, and should avoid typos (e.g. "Donghua", not "Donghu").

## 3. Render

Write the filled template to a file (plain text, not JSON) and pipe it in:

```bash
QI edit -i ref1.png -i ref2.png --ratio 4:3 < sheet_prompt.txt   # with reference images
QI t2i --ratio 4:3 < sheet_prompt.txt                             # without
```

The reply has the job `id`, queue `position`, and `eta_s`, the estimated seconds until it is ready; a sheet takes
as long as any other 2K render on that GPU (`eta_s` is the estimate to use, not a fixed figure). Then run
`QI wait <id>` (polls up to 100 s) again until `status` is `done`;
`saved_to` has the PNG path. Pass `--seed N` when the user wants to iterate on settings while keeping the same
composition, and reuse it on every rerun.

## 4. Check and report

Open the saved PNG with your image-viewing tool and report honestly. Typical results from testing:

- **Works:** a consistent face, hair and outfit across all panels; the largest region is the main turnaround with a
  height scale; header name, role and theme render legibly; with a reference, the identity matches it closely.
- **Common misses:** small panel labels misspelled; the back view sometimes dropped from the main turnaround; the
  palette sometimes gets text labels; the consistency and quality requirements are sometimes drawn as text panels.
  These are model limits, not settings errors.
- **Too illustrated when realism was wanted:** say so and recommend a photographic reference image (see step 1).
