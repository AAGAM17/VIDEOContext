# Agent examples

Six realistic requests a developer makes to a coding agent, and what happens when the agent
uses VIDEOContext. Each example lists the prompt, the workflow the skill prescribes, the
exact commands, and output. All outputs are **real**, produced by these commands on the
repository's demo video (`tests/fixtures/demo.mp4`, 62 s: slides, a login page, a terminal
with a failing test), abridged only by cutting lines. Agent answers are written from that
evidence; the one in [ask-temporal-question](ask-temporal-question/) is from an actual
Claude Code run.

| Example | Request |
|---|---|
| [analyze-video](analyze-video/) | "Analyze demo.mp4." |
| [ask-temporal-question](ask-temporal-question/) | "What happened after the error?" |
| [bug-investigation](bug-investigation/) | "Find where the app starts failing." |
| [ui-reconstruction](ui-reconstruction/) | "Recreate the login page in React." |
| [documentation-generation](documentation-generation/) | "Turn this recording into docs." |
| [video-comparison](video-comparison/) | "What changed between these recordings?" |

To reproduce: `cp tests/fixtures/demo.mp4 /tmp/x/ && cd /tmp/x && videocontent analyze demo.mp4`,
then run the commands shown. The agent adds `--agent` for JSON; the human-readable form is
shown where it is shorter.
