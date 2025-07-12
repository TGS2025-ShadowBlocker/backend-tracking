import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import socket
import time
import base64
import json

class PersonSegmentation:
    def __init__(self):
        # MediaPipeのセルフィーセグメンテーション初期化
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1  # 0: 軽量モデル, 1: 高精度モデル
        )
        
        # Webカメラのセットアップ
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # TCP通信用のソケット
        self.server_socket = None
        self.client_socket = None
        self.setup_server()
        
        self.running = False
        
    def setup_server(self):
        """Unity通信用のTCPサーバーを設定"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', 12345))
        self.server_socket.listen(1)
        print("サーバー待機中... (localhost:12345)")
        
    def wait_for_unity_connection(self):
        """Unityからの接続を待機"""
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"Unity接続成功: {addr}")
            return True
        except Exception as e:
            print(f"接続エラー: {e}")
            return False
    
    def process_frame(self, frame):
        """フレームを処理して人物のみを切り取り、黒塗りにする"""
        # フレームを左右反転
        frame = cv2.flip(frame, 1)
        
        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # セグメンテーション実行
        results = self.selfie_segmentation.process(rgb_frame)
        
        # マスクの取得（0-1の範囲）
        mask = results.segmentation_mask
        
        # マスクを3チャンネルに拡張
        np.stack([mask] * 3, axis=-1)
        
        # 人物部分を黒塗りにする
        # マスクが1（人物）の部分を黒（0,0,0）に設定
        black_silhouette = np.zeros_like(frame)
        black_silhouette[mask > 0.5] = [0, 0, 0]  # 黒塗り
        
        # 背景部分を透明にする場合のマスク情報も保存
        alpha_mask = (mask > 0.5).astype(np.uint8) * 255
        
        return black_silhouette, alpha_mask
    
    def frame_to_base64(self, frame, alpha_mask):
        """フレームをBase64エンコードしてUnityに送信可能な形式に変換"""
        # PNGとして保存するためにアルファチャンネルを追加
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_frame[:, :, 3] = alpha_mask  # アルファチャンネルを設定
        
        # PIL Imageに変換
        pil_image = Image.fromarray(rgba_frame)
        
        # Base64エンコード
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def send_to_unity(self, image_data):
        """画像データをUnityに送信"""
        try:
            if self.client_socket:
                # JSONフォーマットで送信
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
        """メイン処理ループ"""
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("フレーム取得失敗")
                continue
            
            # フレーム処理
            processed_frame, alpha_mask = self.process_frame(frame)
            
            # Base64に変換
            image_data = self.frame_to_base64(processed_frame, alpha_mask)
            
            # Unityに送信
            if not self.send_to_unity(image_data):
                print("Unity接続が切断されました")
                break
                
            # デバッグ用：処理済みフレームを表示
            cv2.imshow('Processed Frame', processed_frame)
            
            # ESCキーで終了
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
            # フレームレート制御
            time.sleep(1/30)  # 30fps
    
    def stop(self):
        """処理を停止"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        cv2.destroyAllWindows()

def main():
    # 必要なライブラリをインストール
    print("必要なライブラリ:")
    print("pip install opencv-python mediapipe pillow numpy")
    
    processor = PersonSegmentation()
    
    try:
        # Unity接続待機
        if processor.wait_for_unity_connection():
            print("処理開始...")
            processor.start_processing()
    except KeyboardInterrupt:
        print("処理を中断しました")
    finally:
        processor.stop()

if __name__ == "__main__":
    main()