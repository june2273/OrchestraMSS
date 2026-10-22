#!/bin/bash
# 다른 기기에서 접속할 수 있는 임시 공개 링크를 만든다 (Cloudflare Quick Tunnel).
# 앱 서버가 꺼져 있으면 같이 켜고, Ctrl+C 로 종료하면 이 스크립트가 켠 것만 함께 끈다.
cd "$(dirname "$0")"
PORT=8000
METRICS=127.0.0.1:20241
SERVER_PID=""

cleanup() {
  echo
  echo "종료합니다…"
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then
  echo "앱 서버가 이미 실행 중입니다 (포트 $PORT)."
else
  echo "앱 서버를 켭니다… (로그: server.log)"
  .venv/bin/python main.py > server.log 2>&1 &
  SERVER_PID=$!
  until curl -s -o /dev/null "http://127.0.0.1:$PORT/"; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "서버 시작 실패. server.log 를 확인하세요."; exit 1; }
    sleep 1
  done
fi

echo "터널을 엽니다… (로그: tunnel.log)"
cloudflared tunnel --url "http://127.0.0.1:$PORT" --metrics "$METRICS" > tunnel.log 2>&1 &
TUNNEL_PID=$!

HOST=""
for _ in $(seq 1 60); do
  HOST=$(curl -s "http://$METRICS/quicktunnel" | python3 -c "import json,sys; print(json.load(sys.stdin).get('hostname',''))" 2>/dev/null)
  [ -n "$HOST" ] && break
  kill -0 "$TUNNEL_PID" 2>/dev/null || { echo "터널 시작 실패. tunnel.log 를 확인하세요."; cleanup; }
  sleep 1
done
[ -z "$HOST" ] && { echo "링크를 받지 못했습니다. tunnel.log 를 확인하세요."; cleanup; }

# 터널이 실제로 응답할 때까지 대기. 너무 이른 조회가 맥의 DNS 캐시에 "없음"으로 남지 않도록
# 확인은 공용 DNS(1.1.1.1)로 직접 한다.
URL="https://$HOST"
echo "링크가 열릴 때까지 기다립니다…"
for _ in $(seq 1 30); do
  IP=$(dig +short @1.1.1.1 "$HOST" | head -1)
  [ -n "$IP" ] && [ "$(curl -s -o /dev/null -w '%{http_code}' --resolve "$HOST:443:$IP" "$URL/")" = "200" ] && break
  sleep 2
done

printf '%s' "$URL" | pbcopy
echo
echo "=============================================="
echo "  접속 링크 (클립보드에 복사됨):"
echo "  $URL"
echo "=============================================="
echo "이 창을 열어 두는 동안만 접속됩니다. 끝내려면 Ctrl+C."
echo "주의: 링크를 아는 사람은 누구나 업로드·삭제할 수 있습니다."
wait "$TUNNEL_PID"
cleanup
