# prompt

❯ Use subagents for this task...                                                                                                                                                                                                   
                                                                                                                                                                                                                                   
  Agent 1: inspect the current fallback chain and summarize weaknesses.                                                                                                                                                            
                                                                                                                                                                                                                                   
  Agent 2: inspect test coverage and identify missing cases.                                                                                                                                                                       
                                                                                                                                                                                                                                   
  Agent 3: inspect the README and suggest changes needed to reflect current architecture.                                                                                                                                          
                                                                                                                                                                                                                                   
  Then synthesize the findings into one implementation plan.                                                                                                                                                                       
                                                                                                                                                                                                                                   
  Do not edit code until I approve.                                                                                                                                                                                                

# response

⏺ Spawning all three agents in parallel now.

⏺ Running 3 Explore agents…
   ├ Inspect fallback chain weaknesses
   ├ Inspect test coverage gaps
   ├ Inspect test coverage gaps
   └ Inspect README vs current architecture

⏺ All three agents are running in parallel. I'll synthesize their findings into an implementation plan once they all report back.

✻ Sautéed for 25s · 3 local agents still running

⏺ Agent "Inspect test coverage gaps" completed
  ⎿  ⧉ Selected 1 lines from !adhoc/claude_code_outputs/output1.md in PyCharm

⏺ Test coverage agent is done. Waiting on the other two.

✻ Sautéed for 1s · 2 local agents still running

⏺ Agent "Inspect README vs current architecture" completed
  ⎿  ⧉ Selected 1 lines from !adhoc/claude_code_outputs/output1.md in PyCharm

⏺ Two down, one to go — still waiting on the fallback chain agent.

✻ Cogitated for 2s · 1 local agent still running

⏺ Agent "Inspect fallback chain weaknesses" completed
  ⎿  ⧉ Selected 1 lines from !adhoc/claude_code_outputs/output1.md in PyCharm

## final

⏺ All three are done. Here's the synthesis:

  ---
  Implementation Plan

  Findings Summary
  
  One important note: 
  The documentation and test-coverage agents both reported:
  the src/agents/ directory doesn't exist, 
  but the fallback-chain agent successfully read those files 
  — and your commit history confirms they were created. 
  
  The other two agents appear to have missed them. I'll treat them as present.

  ---
  
### Track 1 — Harden the Fallback Chain
  
  Source: Agent 1. Ordered by severity.

#### 1a. Wrap post-loop tool calls (Critical — crash risk)
  agent.py:70, 75 — tool_evaluate_target_resume() and tool_revise_target_resume() are called with no try/except. 
  A ValueError from malformed LLM JSON propagates uncaught and kills main(). 
  Wrap each in a try/except that logs the error and continues to the render step.

#### 1b. Guard evaluation against missing target_resume_txt (Critical)
  agent.py:69-70 — the check is "resume_evaluation_json" not in artifacts, 
  but never confirms target_resume_txt is present first. 
  The ValueError from tool_evaluate_target_resume() at tools.py:726-728 bubbles up uncaught. 
  Add an explicit guard before the evaluation block.

#### 1c. Fix inconsistent state after revision evaluation failure (High)
  tools.py:834-839 — 
  revised_resume_txt is written to artifacts 
  before _evaluate_resume() is called. 
  If that call fails, 
  state has a revised resume with no evaluation metadata. 
  Either write all revision artifacts atomically 
  or roll back revised_resume_txt on failure.

#### 1d. Handle parse errors in the agent loop (High)
  agent_loop.py:114, 162-164 — 
  a ValueError on JSON parse breaks the loop immediately, 
  bypassing both fallback paths (lines 261 and 294). 
  The fallbacks are only triggered on FinalAction or iteration exhaustion. 
  Catch parse errors explicitly 
  and route them into the same fallback logic.

#### 1e. Reject placeholder error strings as valid output (High)
  agent_loop.py:325-327 — 
  when both fallback attempts fail, 
  a literal error string is written to target_resume_txt. 
  agent.py:88 then happily renders it as a resume. 
  Add a sentinel check (e.g., string starts with "ERROR:")
  before accepting the artifact as a valid draft.

#### 1f. Validate corpus before generation (Medium)
  No check that state.corpus_examples is non-empty 
  before tool_generate_target_resume is called. 
  Add a warning log (or soft error) 
  if generation is attempted with zero corpus examples, 
  so failures aren't silent.

### Track 2 — Fill Test Coverage Gaps
  
Source: Agent 2.

#### 2a. LLM-dependent tools with zero coverage
  - tool_evaluate_target_resume — no test at all
  - tool_generate_target_resume — no test 
  - tool_retrieve_similar_resume_examples — no test
  These should be tested with a FakeLLM 
  - the same way test_agent_loop.py does it.
  
#### 2b. Missing edge cases for existing tools
  - Malformed JSON returned by LLM  
    - verify _parse_action() 
    - error propagates correctly and loop exits gracefully (not silently)
  - overall_score: null in evaluation JSON 
    - verify revision threshold logic doesn't crash 
  - Empty corpus (corpus_examples=[]) 
    - verify generation doesn't silently produce an empty/bad resume
  - Partial revision artifacts 
    - (has revised_resume_txt, missing revision_evaluation_json) 
    - verify tool_render_report produces degraded-but-valid output

#### 2c. Fallback path coverage
  - Max-iteration exhaustion 
    - fallback LLM call succeeds: verify target_resume_txt is a real resume, not a placeholder
  - Max-iteration exhaustion 
    - fallback LLM call also fails: verify placeholder is detected and downstream pipeline handles it without crashing
  - These paths are nominally tested in test_agent_loop.py 
    - but assertions are too weak to catch the placeholder-as-valid-resume bug from 1e

#### 2d. Verify test_agents.py covers the handoff path
  The commit message says 9 tests exist; 
  confirm analyst→writer→critic handoff is tested end-to-end with a FakeLLM, 
  not just each agent in isolation.

  
### Track 3 — Documentation
  
  Source: Agent 3.

#### 3a. README: add a code-structure section
  README currently describes the workflow at a step-by-step level but doesn't map to actual files. Add a short "Code Structure" section that distinguishes Phase 1 (agent_loop.py) from Phase 2 (agent.py post-loop pipeline) and
  lists what each phase owns.

#### 3b. README: add corpus format requirements
  Document that each corpus entry requires a jd.md + resume_variant.txt pair, and that tool_load_resume_corpus reports skipped directories with reasons.

#### 3c. CLAUDE.md: add graceful degradation section
  Currently undocumented at the architecture level. Describe the two fallback paths in agent_loop.py, the placeholder sentinel, and the post-loop safety net.

#### 3d. CLAUDE.md: add multi-agent layer section
  The new src/agents/ layer (analyst/writer/critic) is not mentioned anywhere in CLAUDE.md. Add a section describing each agent's responsibility and how the handoff works, parallel to the existing Phase 1/Phase 2 descriptions.

  
### Suggested Implementation Order
  
  ┌──────────┬─────────┬─────────────────────────────────────────────────────────────┐
  │ Priority │  Track  │                             Work                             │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 1        │ 1a + 1b │ try/except wrappers + target_resume_txt guard in agent.py    │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 2        │ 1c      │ atomic revision artifact writes in tools.py                  │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 3        │ 1d + 1e │ parse error routing + placeholder sentinel in agent_loop.py  │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 4        │ 2a + 2b │ new unit/integration tests using FakeLLM                     │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 5        │ 2c + 2d │ fallback path tests + verify test_agents.py handoff coverage │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 6        │ 1f      │ corpus validation warning                                    │
  ├──────────┼─────────┼──────────────────────────────────────────────────────────────┤
  │ 7        │ 3a–3d   │ documentation updates                                        │
  └──────────┴─────────┴──────────────────────────────────────────────────────────────┘

  Let me know if you want to adjust priorities 
  or scope before I start implementing.

✻ Sautéed for 46s
