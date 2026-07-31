# Manual QA — cry.wav per stage (#30)

- [ ] Run a single-stage generation (`--description "fire lizard"`) and confirm a `cry.wav` appears in the stage dir alongside `sprite.png`.
- [ ] Run in line mode (`--mode line`) and confirm each of the 3 stage dirs gets its own `cry.wav`.
- [ ] Play the generated `cry.wav` and confirm it's audible (mono, Gen 3-style tone) and varies by stage/type.
- [ ] Force a sprite failure (e.g. break the pipeline) and confirm `cry.wav` is still written for that stage.
- [ ] Confirm a cry failure only prints a `Warning: cry generation failed for <name>` to stderr and the run still finishes with `Done!` and produces sprites/.ini.
