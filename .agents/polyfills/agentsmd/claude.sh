#!/bin/sh

# This project is licensed under the [Blue Oak Model License, Version 1.0.0][1],
# but you may also license it under [Apache License, Version 2.0][2] if you—
# or your legal team—prefer.
# [1]: https://blueoakcouncil.org/license/1.0.0
# [2]: https://www.apache.org/licenses/LICENSE-2.0

cd "$CLAUDE_PROJECT_DIR"
agent_files=$(find . -name "AGENTS.md" -type f)

# Check for global AGENTS.md if global install exists
# Detection: check if agentfill is installed globally by checking for global polyfill directory
has_global_agentsmd=0
if [ -d "$HOME/.agents/polyfills" ] && [ -f "$HOME/AGENTS.md" ]; then
has_global_agentsmd=1
# Add to list if we have project files
if [ -n "$agent_files" ]; then
agent_files="$HOME/AGENTS.md
$agent_files"
else
agent_files="$HOME/AGENTS.md"
fi
fi

# Exit if no AGENTS.md files found
[ -z "$agent_files" ] && exit 0

cat <<end_context
<agentsmd_instructions>
This project uses AGENTS.md files to provide scoped instructions based on the
file or directory being worked on.

This project has the following AGENTS.md files:

<available_agentsmd_files>
$agent_files
</available_agentsmd_files>

NON-NEGOTIABLE: When working with any file or directory within the project:

1. Load ALL AGENTS.md files in the directory hierarchy matching that location
   BEFORE you start working on (reading/writing/etc) the file or directory. You
   do not have to reload AGENTS.md files you have already loaded previously.

2. ALWAYS apply instructions from the AGENTS.md files that match that location.
   When there are conflicting instructions, apply instructions from the
   AGENTS.md file that is CLOSEST (most specific) to that location. More
   specific instructions OVERRIDE more general ones.

   Precedence order (from lowest to highest priority):
   - Global AGENTS.md (~/AGENTS.md) - lowest priority
   - Project root AGENTS.md (./AGENTS.md)
   - Nested AGENTS.md files - highest priority (closest to the file)

   <example>
     Project structure:
       ~/AGENTS.md          (global)
       ./AGENTS.md               (project root)
       subfolder/
         file.txt
         AGENTS.md               (nested)

     When working with "subfolder/file.txt":
       - Instructions from "subfolder/AGENTS.md" take highest precedence
       - Instructions from "./AGENTS.md" override global
       - Instructions from "~/AGENTS.md" apply only if not overridden
   </example>

3. If there is a root ./AGENTS.md file, ALWAYS apply its instructions to ALL
   work within the project, as everything you do is within scope of the project.
   Precedence rules still apply for conflicting instructions.
</agentsmd_instructions>
end_context

# Load global AGENTS.md first (lowest precedence)
if [ "$has_global_agentsmd" -eq 1 ]; then
cat <<-end_global_context

The content of ~/AGENTS.md is as follows:

<agentsmd path="~/AGENTS.md" absolute_path="$HOME/AGENTS.md">
$(cat "$HOME/AGENTS.md")
</agentsmd>
end_global_context
fi

# If there is a root AGENTS.md, load it now (higher precedence than global)
if [ -f "./AGENTS.md" ]; then
cat <<-end_root_context

The content of ./AGENTS.md is as follows:

<agentsmd path="./AGENTS.md" absolute_path="$CLAUDE_PROJECT_DIR/AGENTS.md">
$(cat "./AGENTS.md")
</agentsmd>
end_root_context
fi
