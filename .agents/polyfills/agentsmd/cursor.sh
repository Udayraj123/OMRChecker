#!/bin/sh

# This project is licensed under the [Blue Oak Model License, Version 1.0.0][1],
# but you may also license it under [Apache License, Version 2.0][2] if you—
# or your legal team—prefer.
# [1]: https://blueoakcouncil.org/license/1.0.0
# [2]: https://www.apache.org/licenses/LICENSE-2.0

# Cursor IDE sessionStart hook for AGENTS.md polyfill
# Outputs JSON with additional_context for context injection

PROJECT_DIR="${CURSOR_PROJECT_DIR:-${CLAUDE_PROJECT_DIR:-.}}"
cd "$PROJECT_DIR"
agent_files=$(find . -name "AGENTS.md" -type f)

# Check for global AGENTS.md if global install exists
has_global_agentsmd=0
if [ -d "$HOME/.agents/polyfills" ] && [ -f "$HOME/AGENTS.md" ]; then
has_global_agentsmd=1
if [ -n "$agent_files" ]; then
agent_files="$HOME/AGENTS.md
$agent_files"
else
agent_files="$HOME/AGENTS.md"
fi
fi

# If no AGENTS.md files found, output minimal JSON and exit
if [ -z "$agent_files" ]; then
printf '{"continue":true}\n'
exit 0
fi

# Build the context string
context="<agentsmd_instructions>
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

     When working with \"subfolder/file.txt\":
       - Instructions from \"subfolder/AGENTS.md\" take highest precedence
       - Instructions from \"./AGENTS.md\" override global
       - Instructions from \"~/AGENTS.md\" apply only if not overridden
   </example>

3. If there is a root ./AGENTS.md file, ALWAYS apply its instructions to ALL
   work within the project, as everything you do is within scope of the project.
   Precedence rules still apply for conflicting instructions.
</agentsmd_instructions>"

# Append global AGENTS.md content (lowest precedence)
if [ "$has_global_agentsmd" -eq 1 ]; then
context="$context

The content of ~/AGENTS.md is as follows:

<agentsmd path=\"~/AGENTS.md\" absolute_path=\"$HOME/AGENTS.md\">
$(cat "$HOME/AGENTS.md")
</agentsmd>"
fi

# Append root AGENTS.md content (higher precedence than global)
if [ -f "./AGENTS.md" ]; then
context="$context

The content of ./AGENTS.md is as follows:

<agentsmd path=\"./AGENTS.md\" absolute_path=\"$PROJECT_DIR/AGENTS.md\">
$(cat "./AGENTS.md")
</agentsmd>"
fi

# JSON-encode the context and output the hook response
printf '%s' "$context" | perl -MJSON::PP -0777 -e '
my $text = do { local $/; <STDIN> };
my $json = JSON::PP->new->utf8;
print $json->encode({
additional_context => $text,
continue => JSON::PP::true
});
'
