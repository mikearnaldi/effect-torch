# DiffusionGemma chat template fixture

`chat_template.jinja` is an unchanged, 18,575-byte copy of the official Google
Gemma Engineering Team template from
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/chat_template.jinja).

- Model revision: `f7f5b7f5fa82ffc52addd066915886d497f5517b`
- Template SHA-256: `9aeb7eac68ad87bba7567e9d4597ff203e5609f1b427d9e823437d0142cc61bf`
- Author: Google Gemma Engineering Team, as credited in the template header
- License: Apache-2.0, per the
  [pinned model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/README.md).
  A copy is included in `LICENSE`. The upstream repository has no `NOTICE` file.

`cases.json` contains test contexts and exact strings rendered by Jinja2 3.1.6.
The parent directory's `render-reference.py` regenerates those strings without
rewriting the template. Cases cover generation prompts, thinking channels,
content parts, tool definitions and responses, and missing or null content.
