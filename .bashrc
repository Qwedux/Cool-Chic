export LOG_ROOT=$(pwd)
export WORKSPACE_NAME=$(basename "$LOG_ROOT")
echo "Starting logging for workspace: $WORKSPACE_NAME"
echo "Logs will be stored in: $LOG_ROOT/${WORKSPACE_NAME}.log"

# Log shell commands per workspace
log_command() {
    mkdir -p "$LOG_ROOT"
    export COMMAND_LOG_FILE="$LOG_ROOT/${WORKSPACE_NAME}.log"

    local last_command="$(history 1)"

    if [[ -n "$last_command" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | $PWD | $last_command" >> "$COMMAND_LOG_FILE"
    fi
}

# Hook into Bash PROMPT_COMMAND
export PROMPT_COMMAND="log_command; $PROMPT_COMMAND"
source .venv/bin/activate
