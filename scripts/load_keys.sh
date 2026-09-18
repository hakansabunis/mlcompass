# Export provider keys from the untracked api_keys.txt.
#
#   source scripts/load_keys.sh
#
# The file is `name = "value"` with names that are not the env vars the
# harness reads, so this maps them. It prints which variables were set and
# never their values -- a key echoed into a terminal ends up in scrollback,
# in a transcript, and in whatever captures either.
#
# api_keys.txt is gitignored and has never been committed. Keep it that way:
# the replication package is meant to be public.

_KEYFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/api_keys.txt"

if [ ! -f "$_KEYFILE" ]; then
    echo "load_keys: no api_keys.txt at $_KEYFILE" >&2
    return 1 2>/dev/null || exit 1
fi

_get() {
    # name = "value"  ->  value. Tolerates spaces and either quote style.
    sed -nE "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*[\"']?([^\"']+)[\"']?[[:space:]]*$/\1/p" \
        "$_KEYFILE" | head -1
}

for _pair in "deepseek_api:DEEPSEEK_API_KEY" \
             "chatgpt_api:OPENAI_API_KEY" \
             "mistral_api:MISTRAL_API_KEY"; do
    _src="${_pair%%:*}"
    _dst="${_pair##*:}"
    _val="$(_get "$_src")"
    if [ -n "$_val" ]; then
        export "$_dst=$_val"
        echo "load_keys: $_dst set (${#_val} chars)"
    else
        echo "load_keys: $_dst NOT found in api_keys.txt" >&2
    fi
done

unset _pair _src _dst _val _KEYFILE
unset -f _get
