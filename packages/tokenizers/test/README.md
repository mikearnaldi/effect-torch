# Chat template regression tests

Build the host addon, then use core's existing Vitest dependencies to run the
tokenizer-local suite. Run these commands from the workspace root:

```sh
pnpm --filter @effect-torch/tokenizers build:debug
pnpm --filter @effect-torch/core exec vitest run --config ../tokenizers/test/vitest.config.ts
pnpm exec tsc -p packages/tokenizers/test/tsconfig.json
cargo test -p effect-torch-tokenizers chat_template_tests
```

The native and TypeScript tests share Python-method cases and exact rendered
strings from Jinja2 3.1.6. Regenerate the reference strings with:

```sh
uv run --with jinja2==3.1.6 python packages/tokenizers/test/fixtures/render-reference.py
pnpm exec dprint fmt packages/tokenizers/test/fixtures/diffusiongemma/cases.json
```

The reference environment uses `ImmutableSandboxedEnvironment`, `trim_blocks=True`,
and `lstrip_blocks=True`, as Transformers does. The tests run offline using the
checked-in renders and a tiny tokenizer, with no model weights or Python runtime.
