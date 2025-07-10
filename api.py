from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import cv2
from tracking import RealtimePoseTracker
import uvicorn
import time

app = FastAPI()

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数でトラッキング状態を管理
tracker = None
current_actions = {"punch": False, "kick": False}
tracking_thread = None
tracking_active = False
current_frame = None
cap = None

def tracking_worker():
    """バックグラウンドでポーズトラッキングを実行"""
    global tracker, current_actions, tracking_active, current_frame, cap
    
    # カメラを初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return
    
    # ポーズトラッカーを初期化
    tracker = RealtimePoseTracker()
    print("ポーズトラッキングを開始しました")
    
    try:
        while tracking_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 左右反転
            frame = cv2.flip(frame, 1)
            
            # ポーズトラッキングを実行
            processed_frame = tracker.process_frame(frame)
            
            # デバッグ情報を表示
            debug_info = tracker.get_debug_info()
            y_start = 150
            font_size = 0.8
            
            # パンチスコアを表示
            cv2.putText(processed_frame, f"L_Punch:{debug_info['left_punch_score']:.3f}", 
                        (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"R_Punch:{debug_info['right_punch_score']:.3f}", 
                        (10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 2)
            
            # キックスコアを表示
            cv2.putText(processed_frame, f"L_Kick:{debug_info['left_kick_score']:.3f}", 
                        (10, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
            cv2.putText(processed_frame, f"R_Kick:{debug_info['right_kick_score']:.3f}", 
                        (10, y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
            
            # 閾値を表示
            cv2.putText(processed_frame, "Threshold: 0.08", 
                        (10, y_start + 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 2)
            
            # 現在のフレームを保存
            current_frame = processed_frame.copy()
            
            # 検知結果を更新
            actions = tracker.get_detected_actions()
            current_actions.update(actions)
            
            time.sleep(1/30)
    
    except Exception as e:
        print(f"トラッキングエラー: {e}")
    
    finally:
        if cap:
            cap.release()
        print("ポーズトラッキングを終了しました")

def generate_frames():
    """フレームを生成してストリーミング"""
    global current_frame
    
    while tracking_active:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1/30)

@app.on_event("startup")
async def startup_event():
    """APIサーバー起動時にトラッキングを開始"""
    global tracking_thread, tracking_active
    
    tracking_active = True
    tracking_thread = threading.Thread(target=tracking_worker, daemon=True)
    tracking_thread.start()
    print("APIサーバーが起動しました")

@app.on_event("shutdown")
async def shutdown_event():
    """APIサーバー終了時にトラッキングを停止"""
    global tracking_active, tracker, cap
    
    tracking_active = False
    if tracker:
        tracker.cleanup()
    if cap:
        cap.release()
    print("APIサーバーが終了しました")

@app.get("/")
async def read_root():
    """カメラ映像を表示するHTMLページ"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ポーズトラッキング カメラ映像</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
                text-align: center;
            }
            h1 { color: #333; margin-bottom: 20px; }
            .video-container {
                margin: 20px 0;
                border: 3px solid #333;
                border-radius: 10px;
                overflow: hidden;
                display: inline-block;
                background: #000;
            }
            img {
                display: block;
                max-width: 100%;
                height: auto;
            }
            .status {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px auto;
                max-width: 600px;
            }
            .status-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .action-status {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .detected {
                background-color: #ff4444;
                color: white;
                animation: pulse 1s infinite;
            }
            .not-detected {
                background-color: #e0e0e0;
                color: #666;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            .refresh-btn {
                background: #4CAF50;
                border: none;
                color: white;
                padding: 15px 30px;
                font-size: 16px;
                margin: 10px;
                cursor: pointer;
                border-radius: 5px;
            }
            .api-info {
                background: #333;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px auto;
                max-width: 800px;
                text-align: left;
            }
        </style>
        <script>
            async function updateStatus() {
                try {
                    const response = await fetch('/actions');
                    const actions = await response.json();
                    
                    const punchElement = document.getElementById('punch-status');
                    const kickElement = document.getElementById('kick-status');
                    
                    punchElement.textContent = actions.punch ? 'PUNCH DETECTED!' : '待機中...';
                    kickElement.textContent = actions.kick ? 'KICK DETECTED!' : '待機中...';
                    
                    punchElement.className = actions.punch ? 'action-status detected' : 'action-status not-detected';
                    kickElement.className = actions.kick ? 'action-status detected' : 'action-status not-detected';
                } catch (error) {
                    console.error('Status update error:', error);
                }
            }
            
            setInterval(updateStatus, 100);
            window.onload = updateStatus;
        </script>
    </head>
    <body>
        <h1>🎯 ポーズトラッキング リアルタイム映像</h1>
        
        <div class="video-container">
            <img src="/video_feed" alt="カメラ映像" width="640" height="480" />
        </div>
        
        <div class="status">
            <div class="status-card">
                <h3>🥊 パンチ検知</h3>
                <div id="punch-status" class="action-status not-detected">読み込み中...</div>
            </div>
            <div class="status-card">
                <h3>🦵 キック検知</h3>
                <div id="kick-status" class="action-status not-detected">読み込み中...</div>
            </div>
        </div>
        
        <div>
            <button class="refresh-btn" onclick="location.reload()">🔄 ページ更新</button>
            <button class="refresh-btn" onclick="updateStatus()">📊 ステータス更新</button>
        </div>
        
        <div class="api-info">
            <h3>📡 API エンドポイント</h3>
            <p><strong>全動作:</strong> <code>GET /actions</code> → {"punch": true/false, "kick": true/false}</p>
            <p><strong>パンチのみ:</strong> <code>GET /actions/punch</code> → {"punch": true/false}</p>
            <p><strong>キックのみ:</strong> <code>GET /actions/kick</code> → {"kick": true/false}</p>
            <p><strong>システム状態:</strong> <code>GET /status</code></p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    """カメラ映像をストリーミング"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/actions")
async def get_actions():
    """現在の動作検知結果を返す"""
    global current_actions
    return current_actions

@app.get("/status")
async def get_status():
    """システムステータスを返す"""
    global tracker, tracking_active
    
    status = {
        "tracking_active": tracking_active,
        "tracker_initialized": tracker is not None,
        "current_actions": current_actions
    }
    
    if tracker:
        debug_info = tracker.get_debug_info()
        status["debug_info"] = debug_info
    
    return status

@app.get("/actions/punch")
async def get_punch_status():
    """パンチ検知結果を返す"""
    global current_actions
    return {"punch": current_actions.get("punch", False)}

@app.get("/actions/kick")
async def get_kick_status():
    """キック検知結果を返す"""
    global current_actions
    return {"kick": current_actions.get("kick", False)}

if __name__ == "__main__":
    print("ポーズトラッキングAPIサーバーを起動しています...")
    print("ブラウザで以下のURLにアクセスしてカメラ映像を確認:")
    print("  http://localhost:8000/")
    print("\nAPIエンドポイント:")
    print("  http://localhost:8000/actions - 動作検知結果")
    print("  http://localhost:8000/actions/punch - パンチ検知結果")
    print("  http://localhost:8000/actions/kick - キック検知結果")
    print("\nCtrl+C で終了")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")