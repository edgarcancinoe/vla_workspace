# Repository Layout

## Principles

- reusable code lives in `src/thesis_vla`
- executable tools live in `apps`
- static configuration lives in `config`
- generated state lives in `runtime`
- documentation lives in `docs`
- tests live in `tests`

## External repos

Upstream code is intentionally kept outside this repo's internal structure and should remain under the sibling `../repos/` directory.

## Runtime policy

Anything generated while running experiments, robot sessions, or debugging should go to `runtime/` instead of polluting source directories.
