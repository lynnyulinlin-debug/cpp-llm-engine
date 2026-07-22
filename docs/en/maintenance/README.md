# Maintenance Docs

This section is for contributors and maintainers.

## Contents

- [tutorial maintenance rules](/maintenance/tutorial_maintenance)
- [writing guidelines](/maintenance/tutorial_writing_guidelines)
- [chapter template](/maintenance/chapter_template)
- [summary template](/maintenance/summary_template)

## How to use it

- Read this when you change structure, add chapters, or adjust bridges.
- Skip it if you only want to learn the tutorial.
- Keep the main line stable and use this section to preserve that stability.

## Checks

- If you change docs or links, run `npm run docs:check-links` first
- Then run `npm run docs:build`
- If the build still fails, check routes and anchors before changing the prose
