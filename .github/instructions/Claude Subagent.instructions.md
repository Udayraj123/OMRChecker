---
description: Claude 4.6 Subagent instructions for analysis and reporting results to the main agent.
applyTo: '**/*'
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---
use subagents for all the analysis & report back to mainagent the results, then mainagent present it to user using #askQuestion tool with the most viable options. Then base on the choosen option, mainagent will use subagents for the implementation. Always report back to mainagent the results of the subagents, and let mainagent decide the next steps based on the results. Do not take any action without reporting back to mainagent first. Break down complex tasks into smaller sub-tasks and assign them to subagents for efficient execution. Always ensure that the subagents are working towards the overall goal defined by the main agent.
