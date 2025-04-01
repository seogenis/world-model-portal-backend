# Tmux Backend Commands

## Starting Services

### Start uvicorn in a tmux session
```bash
tmux new -s uvicorn
cd /workspace/world-model-portal-backend
uvicorn app.main:app --host 0.0.0.0 --port 8001
```
Detach from session: `Ctrl+b` then `d`

### Start ngrok in a tmux session
```bash
tmux new -s ngrok
cd /workspace/world-model-portal-backend
ngrok http 8001
```
Detach from session: `Ctrl+b` then `d`

## Managing Sessions

### List all tmux sessions
```bash
tmux ls
```

### Attach to a specific session
```bash
tmux attach -t uvicorn
```

### Kill (delete) a session
```bash
tmux kill-session -t uvicorn
```

### Kill all tmux sessions
```bash
tmux kill-server
```

### Enter scroll/copy mode
```
Ctrl+b then [
```

### Navigate in scroll mode
- Use arrow keys or Page Up/Down to scroll
- Use `/` to search, then `n` for next match
- Press `q` to exit scroll mode

### Copy text in scroll mode
1. Enter scroll mode: `Ctrl+b` then `[`
2. Navigate to starting position
3. Press `Space` to start selection
4. Move cursor to select text
5. Press `Enter` to copy selected text
6. Press `Ctrl+b` then `]` to paste