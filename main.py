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

# セグメンテーション / Unity 送信用
import numpy as np
from PIL import Image
import mediapipe as mp
import socket
import base64
import json

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

# カメラは単一の VideoCapture を共有
cap = None

# 現在のフレームと処理済みフレーム
current_frame = None            # 生映像（左右反転済）
processed_frame = None          # ポーズ処理後フレーム
segmented_frame = None          # セグメンテーション後のフレーム（黒シルエット）
segmented_alpha = None          # セグメンテーションのアルファマスク

# ストリーミングで使う定数
FRAME_BOUNDARY = b'--frame\r\n'
CONTENT_TYPE_BYTES = b'Content-Type: image/jpeg\r\n\r\n'
MEDIA_TYPE_STR = "multipart/x-mixed-replace; boundary=frame"


class PersonSegmentation:
    """MediaPipe による人物セグメンテーションと Unity への送信を扱う小クラス。
    注意: カメラはこのクラスで扱わず、外部で読み取ったフレームを渡して使うようにしています。
    """
    def __init__(self, host='localhost', port=12345, model_selection=1):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )

        # TCP 通信用
        self.server_socket = None
        self.client_socket = None
        self.host = host
        self.port = port
        self.server_thread = None
        self.sending = False
        self.running = False
        self.setup_server()

    def setup_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Unity 用 TCP サーバー待機中... ({self.host}:{self.port})")
        except Exception as e:
            print(f"サーバー作成エラー: {e}")
            self.server_socket = None

    def process_frame(self, frame):
        # frame は左右反転済みの BGR フレームを想定
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask
        if mask is None:
            # セグメントが取れない場合は空画像を返す
            h, w = frame.shape[:2]
            return np.zeros_like(frame), np.zeros((h, w), dtype=np.uint8)

        # 黒シルエットを作成（人物部分を黒で塗る）
        black_silhouette = np.zeros_like(frame)
        black_silhouette[mask > 0.5] = [0, 0, 0]

        alpha_mask = (mask > 0.5).astype(np.uint8) * 255
        return black_silhouette, alpha_mask

    def frame_to_base64(self, frame, alpha_mask):
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_frame[:, :, 3] = alpha_mask
        pil_image = Image.fromarray(rgba_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def send_to_unity(self, image_data):
        try:
            if self.client_socket:
                data = {"type": "image", "data": image_data, "timestamp": time.time()}
                json_data = json.dumps(data)
                message = json_data + "\n"
                self.client_socket.send(message.encode('utf-8'))
                return True
        except Exception as e:
            print(f"送信エラー: {e}")
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None
        return False

    def server_loop(self):
        if not self.server_socket:
            return
        self.running = True
        try:
            while self.running:
                try:
                    self.client_socket, addr = self.server_socket.accept()
                    print(f"Unity 接続成功: {addr}")
                    # 一旦接続ができたら画像送信ループに入る
                    self.sending = True
                    while self.sending and self.running:
                        # segmented_frame と segmented_alpha はグローバルから読む
                        global segmented_frame, segmented_alpha
                        if segmented_frame is not None and segmented_alpha is not None:
                            img_b64 = self.frame_to_base64(segmented_frame, segmented_alpha)
                            if not self.send_to_unity(img_b64):
                                break
                        time.sleep(1/30)
                except Exception:
                    # accept が例外でもループを続ける（クライアント切断など）
                    time.sleep(0.5)
        finally:
            self.running = False

    def start_server_thread(self):
        if self.server_socket and self.server_thread is None:
            self.server_thread = threading.Thread(target=self.server_loop, daemon=True)
            self.server_thread.start()

    def stop(self):
        self.running = False
        self.sending = False
        try:
            if self.client_socket:
                self.client_socket.close()
        except Exception:
            pass
        try:
            if self.server_socket:
                self.server_socket.close()
        except Exception:
            pass


person_seg = PersonSegmentation()


def tracking_worker():
    """バックグラウンドでカメラ読み取り → ポーズトラッキングとセグメンテーションを実行"""
    global tracker, current_actions, tracking_active, current_frame, processed_frame
    global segmented_frame, segmented_alpha, cap

    # ポーズトラッカーを初期化
    tracker = RealtimePoseTracker()
    print("ポーズトラッキングを開始しました")

    try:
        while tracking_active:
            if cap is None:
                time.sleep(0.1)
                continue

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

            # セグメンテーション実行
            seg, alpha = person_seg.process_frame(frame)
            segmented_frame = seg
            segmented_alpha = alpha

            # 検知結果を更新
            actions = tracker.get_detected_actions()
            current_actions.update(actions)

            # フレームレートを調整（30FPS）
            time.sleep(1/30)

    except Exception as e:
        print(f"トラッキングエラー: {e}")

    finally:
        print("ポーズトラッキングを終了しました")


@app.on_event("startup")
async def startup_event():
    """APIサーバー起動時にカメラ/トラッキング/セグメンテーションを開始"""
    global tracking_thread, tracking_active, cap

    # カメラを初期化（単一）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
    else:
        # 任意で解像度指定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracking_active = True
    # トラッキング・セグメンテーション実行スレッド
    tracking_thread = threading.Thread(target=tracking_worker, daemon=True)
    tracking_thread.start()

    # Unity 送信用サーバースレッドを起動
    person_seg.start_server_thread()
    print("APIサーバーが起動しました")


@app.on_event("shutdown")
async def shutdown_event():
    """APIサーバー終了時にトラッキングとサーバーを停止"""
    global tracking_active, tracker, cap

    tracking_active = False
    if tracker:
        try:
            tracker.cleanup()
        except Exception:
            pass

    # person_seg の停止
    person_seg.stop()

    # カメラ解放
    if cap:
        try:
            cap.release()
        except Exception:
            pass

    print("APIサーバーが終了しました")


def _jpeg_bytes_from_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buffer.tobytes()


def generate_frames():
    """映像フレームをJPEGストリームとして生成"""
    global current_frame
    while tracking_active:
        if current_frame is not None:
            buf = _jpeg_bytes_from_frame(current_frame)
            if buf is None:
                continue
            yield (FRAME_BOUNDARY + CONTENT_TYPE_BYTES + buf + b'\r\n')
        time.sleep(1/30)


def generate_processed_frames():
    """処理済みの映像フレームをJPEGストリームとして生成"""
    global processed_frame
    while tracking_active:
        if processed_frame is not None:
            buf = _jpeg_bytes_from_frame(processed_frame)
            if buf is None:
                continue
            yield (FRAME_BOUNDARY + CONTENT_TYPE_BYTES + buf + b'\r\n')
        time.sleep(1/30)


def generate_segmented_frames():
    """セグメンテーション（黒シルエット）フレームをJPEGストリームとして生成"""
    global segmented_frame
    while tracking_active:
        if segmented_frame is not None:
            buf = _jpeg_bytes_from_frame(segmented_frame)
            if buf is None:
                continue
            yield (FRAME_BOUNDARY + CONTENT_TYPE_BYTES + buf + b'\r\n')
        time.sleep(1/30)


@app.get("/video")
async def video_feed():
    """カメラの生の映像をストリーミングで提供"""
    return StreamingResponse(
        generate_frames(),
        media_type=MEDIA_TYPE_STR
    )


@app.get("/video/processed")
async def processed_video_feed():
    """ポーズトラッキング処理後の映像をストリーミングで提供"""
    return StreamingResponse(
        generate_processed_frames(),
        media_type=MEDIA_TYPE_STR
    )


@app.get("/video/segmented")
async def segmented_video_feed():
    """人物セグメンテーション（黒シルエット）の映像をストリーミングで提供"""
    return StreamingResponse(
        generate_segmented_frames(),
        media_type=MEDIA_TYPE_STR
    )


@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "ポーズトラッキング + セグメンテーション API サーバー",
        "status": "running",
        "endpoints": {
            "/status": "システムステータスを取得",
            "/video": "カメラの生映像を表示",
            "/video/processed": "ポーズ処理済映像を表示",
            "/video/segmented": "人物セグメンテーション（黒シルエット）を表示"
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
        try:
            debug_info = tracker.get_debug_info()
            status["debug_info"] = debug_info
        except Exception:
            pass

    return status


if __name__ == "__main__":
    print("ポーズトラッキング + セグメンテーション API サーバーを起動しています...")
    print("アクセス可能なエンドポイント:")
    print("  http://localhost:8000/ - ルート")
    print("  http://localhost:8000/status - システムステータス")
    print("  http://localhost:8000/video - カメラ映像")
    print("  http://localhost:8000/video/processed - 処理済み映像")
    print("  http://localhost:8000/video/segmented - セグメンテーション映像")
    print("\nCtrl+C で終了")

    uvicorn.run(app, port=8000, log_level="info")