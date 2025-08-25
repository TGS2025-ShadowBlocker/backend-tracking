import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import socket
import time
import base64
import json


class PersonSegmentation:
    """Person segmentation processor that can run standalone (with camera/server)
    or be used as a library where frames are provided by an external capture.

    Args:
        use_camera (bool): when True the class opens its own camera capture.
        setup_server (bool): when True it creates a TCP server for Unity connections.
    """
    def __init__(self, use_camera=True, setup_server=True):
        # MediaPipeのセルフィーセグメンテーション初期化
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0: 軽量モデル, 1: 高精度モデル
        )

        # カメラは必要に応じて外部から提供できる
        self.cap = None
        if use_camera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # TCP通信用のソケット（必要な場合のみセットアップ）
        self.server_socket = None
        self.client_socket = None
        self.setup_server_flag = setup_server
        if setup_server:
            self.setup_server()

        self.running = False

    def setup_server(self):
        """Unity通信用のTCPサーバーを設定"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('localhost', 12345))
            self.server_socket.listen(1)
            print("サーバー待機中... (localhost:12345)")
        except Exception as e:
            print(f"サーバーセットアップエラー: {e}")
            self.server_socket = None

    def wait_for_unity_connection(self):
        """Unityからの接続を待機（サーバー有効時のみ）"""
        if not self.server_socket:
            return False
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"Unity接続成功: {addr}")
            return True
        except Exception as e:
            print(f"接続エラー: {e}")
            return False

    def process_frame(self, frame):
        """フレームを処理して人物のみを切り取り、黒塗りにする

        Args:
            frame: BGR OpenCV image

        Returns:
            tuple: (black_silhouette, alpha_mask) where both are numpy arrays.
        """
        if frame is None:
            return None, None

        # フレームを左右反転
        frame = cv2.flip(frame, 1)

        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # セグメンテーション実行
        results = self.selfie_segmentation.process(rgb_frame)

        # マスクの取得（0-1の範囲）
        mask = results.segmentation_mask
        if mask is None:
            # 失敗時は元フレームを返す
            return np.zeros_like(frame), np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # 人物部分を黒塗りにする
        black_silhouette = np.zeros_like(frame)
        black_silhouette[mask > 0.5] = [0, 0, 0]

        # 背景部分を透明にする場合のマスク情報も保存
        alpha_mask = (mask > 0.5).astype(np.uint8) * 255

        return black_silhouette, alpha_mask

    def frame_to_base64(self, frame, alpha_mask):
        """フレームをBase64エンコードしてUnityに送信可能な形式に変換"""
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_frame[:, :, 3] = alpha_mask  # アルファチャンネルを設定

        pil_image = Image.fromarray(rgba_frame)

        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return img_str

    def send_to_unity(self, image_data):
        """画像データをUnityに送信（クライアント接続がある場合のみ）"""
        try:
            if self.client_socket:
                data = {
                    "type": "image",
                    "data": image_data,
                    "timestamp": time.time()
                }
                json_data = json.dumps(data)
                message = json_data + "\n"

                self.client_socket.send(message.encode('utf-8'))
                return True
        except Exception as e:
            print(f"送信エラー: {e}")
        return False

    def start_processing(self):
        """内部キャプチャを用いる場合のメインループ"""
        if not self.cap:
            print("カメラが設定されていません")
            return

        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("フレーム取得失敗")
                continue

            processed_frame, alpha_mask = self.process_frame(frame)

            # 必要であればUnityへ送信
            if self.setup_server_flag and self.client_socket:
                image_data = self.frame_to_base64(processed_frame, alpha_mask)
                if not self.send_to_unity(image_data):
                    print("Unity接続が切断されました")
                    break

            # デバッグ用表示
            cv2.imshow('Processed Frame', processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(1/30)

    def stop(self):
        """処理を停止してリソースを解放"""
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main():
    print("必要なライブラリ:")
    print("pip install opencv-python mediapipe pillow numpy")

    processor = PersonSegmentation()

    try:
        if processor.wait_for_unity_connection():
            print("処理開始...")
            processor.start_processing()
    except KeyboardInterrupt:
        print("処理を中断しました")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
