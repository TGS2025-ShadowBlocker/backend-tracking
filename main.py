from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import threading
import cv2
from tracking import RealtimePoseTracker
import uvicorn
import time

# 映像ストリーミング用
from fastapi.responses import StreamingResponse
import io

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
# 現在のフレームを保存するグローバル変数
current_frame = None
processed_frame = None

def tracking_worker():
    """バックグラウンドでポーズトラッキングを実行"""
    global tracker, current_actions, tracking_active, current_frame, processed_frame
    
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
                print("フレームの読み込みに失敗しました")
                break
            
            # 左右反転
            frame = cv2.flip(frame, 1)
            
            # 生のカメラフレームを保存
            current_frame = frame.copy()
            
            # ポーズトラッキングを実行
            processed = tracker.process_frame(frame)
            processed_frame = processed if processed is not None else frame.copy()
            
            # 検知結果を更新
            actions = tracker.get_detected_actions()
            current_actions.update(actions)
            
            # フレームレートを調整（30FPS）
            time.sleep(1/30)
    
    except Exception as e:
        print(f"トラッキングエラー: {e}")
    
    finally:
        cap.release()
        print("ポーズトラッキングを終了しました")

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
    global tracking_active, tracker
    
    tracking_active = False
    if tracker:
        tracker.cleanup()
    print("APIサーバーが終了しました")

def generate_frames():
    """映像フレームをJPEGストリームとして生成"""
    global current_frame
    while tracking_active:
        if current_frame is not None:
            # フレームをJPEG形式にエンコード
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
                
            # バイナリデータとしてフレームを送信
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # 適切なフレームレート維持のため少し待機
            time.sleep(1/30)

def generate_processed_frames():
    """処理済みの映像フレームをJPEGストリームとして生成"""
    global processed_frame
    while tracking_active:
        if processed_frame is not None:
            # フレームをJPEG形式にエンコード
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            # バイナリデータとしてフレームを送信
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # 適切なフレームレート維持のため少し待機
            time.sleep(1/30)

@app.get("/video")
async def video_feed():
    """カメラの生の映像をストリーミングで提供"""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video/processed")
async def processed_video_feed():
    """ポーズトラッキング処理後の映像をストリーミングで提供"""
    return StreamingResponse(
        generate_processed_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "ポーズトラッキングAPIサーバー",
        "status": "running",
        "endpoints": {
            "/status": "システムステータスを取得",
            "/video": "カメラの生映像を表示",
            "/video/processed": "処理済み映像を表示"
        }
    }

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

if __name__ == "__main__":
    print("ポーズトラッキングAPIサーバーを起動しています...")
    print("アクセス可能なエンドポイント:")
    print("  http://localhost:8000/ - ルート")
    print("  http://localhost:8000/status - システムステータス")
    print("  http://localhost:8000/video - カメラ映像")
    print("  http://localhost:8000/video/processed - 処理済み映像")
    print("\nCtrl+C で終了")
    
    uvicorn.run(app, port=8000, log_level="info")