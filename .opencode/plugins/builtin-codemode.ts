import { Plugin } from "@opencode-ai/plugin"

const builtinTools = [
  "read",
  "grep",
  "glob",
  "shell",
  "edit",
  "write",
  "patch",
  "webfetch",
  "websearch"
] as const

export default Plugin.define({
  id: "builtin-codemode",
  async setup(ctx) {
    await ctx.tool.transform((draft) => {
      for (const id of builtinTools) {
        draft.update(id, (tool) => {
          tool.options = { ...tool.options, codemode: true }
        })
      }
    })
  }
})
