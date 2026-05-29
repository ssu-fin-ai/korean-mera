#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/streamlit.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "이미 실행 중입니다. (PID: $(cat $PID_FILE))"
    exit 1
fi

cd "$SCRIPT_DIR"
source venv/bin/activate
nohup streamlit run app.py > streamlit.log 2>&1 &
echo $! > "$PID_FILE"
echo "Streamlit 시작됨 (PID: $!)"
echo "로그: tail -f $SCRIPT_DIR/streamlit.log"
