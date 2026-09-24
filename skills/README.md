# Qwen-Image agent skills

Three portable skills in the standard `SKILL.md` format (YAML frontmatter with `name` and `description`, then
instructions). Any agent that can read skill folders and run shell commands can use them: Claude Code, the Claude
Agent SDK, and other agents that read `SKILL.md`.

| Skill | Use it for |
|---|---|
| [`qwen-image-t2i`](qwen-image-t2i/SKILL.md) | New images from a text request |
| [`qwen-image-edit`](qwen-image-edit/SKILL.md) | Editing and composing 1–10 input images, turnarounds, grids |
| [`qwen-image-character`](qwen-image-character/SKILL.md) | Full character design sheets, optionally from reference images |

Each skill folder is self-contained:

- `SKILL.md`: the workflow.
- `references/`: its prompt template (t2i prompt rewriter, edit prompt enhancer, or character sheet template). The
  agent applies the template itself and sends only the finished prompt, so the skills work with any capable model.
- `scripts/qwen_image.py`: the job-server client. It uses only the Python standard library, so it runs on any
  `python3` with no installs. The copy is identical in all three skills; keep them in sync when editing.

## Client script

The job server is asynchronous: submitting returns a job id at once, with the queue `position` and `eta_s`
(estimated seconds until the image is ready, including jobs queued ahead). The agent then polls and downloads.

```bash
QI=skills/qwen-image-t2i/scripts/qwen_image.py
python3 $QI health                                      # server up? model settings, measured sec_per_step
python3 $QI t2i --json < rewrite.json                   # {"rewritten_prompt", "wh_ratio"} from the t2i template
python3 $QI t2i --ratio 2:3 < prompt.txt                # or a plain-text prompt
python3 $QI edit -i a.png -i b.png --json < rewrite.json   # images in <image1>, <image2> order
python3 $QI status <id>                                 # status, position, eta_s, progress
python3 $QI wait <id> [--timeout 100] [--out x.png]     # poll until done (or timeout), then save the PNG
python3 $QI download <id> [--out x.png]                 # save the PNG of a finished job
python3 $QI cancel <id>
python3 $QI list
```

Every command prints one JSON object. The exit status is 1 for errors (unreachable server, bad request, missing
file) and for failed or cancelled jobs; errors look like `{"error": "..."}`. A `wait` that times out while the
job is still running exits 0 with its current status, so the agent reruns it. Flags such as `--ratio` override
the values in `--json` input.

| Variable | Default | |
|---|---|---|
| `QWEN_IMAGE_SERVER` | `http://127.0.0.1:8000` | Job server URL (another machine works) |
| `QWEN_IMAGE_OUTPUT_DIR` | `./qwen_images` | Where `wait` / `download` save PNGs without `--out` |

## Install

1. Run the job server on the GPU machine: `uv run qwen_image_server.py`.
2. Copy (or symlink) the skill folders where your agent looks for skills:

```bash
# Claude Code, this project only
mkdir -p .claude/skills && ln -s ../../skills/qwen-image-{t2i,edit,character} .claude/skills/
# Claude Code, all projects
cp -r skills/qwen-image-{t2i,edit,character} ~/.claude/skills/
```

For other agents, use their skills directory, or paste `SKILL.md` into the system prompt and keep the skill folder
readable at the paths it links to.
