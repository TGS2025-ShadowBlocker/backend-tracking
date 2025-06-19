import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

class ActionDetector:
    def __init__(self, history_size=10):
        """
        動作検知クラス
        
        Args:
            history_size (int): 履歴フレーム数
        """
        self.history_size = history_size
        self.landmark_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
        # MediaPipeのランドマークインデックス
        self.POSE_LANDMARKS = {
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28
        }
        
        # 検知の閾値設定
        self.PUNCH_VELOCITY_THRESHOLD = 0.3  # パンチの速度閾値
        self.KICK_VELOCITY_THRESHOLD = 0.25  # キックの速度閾値
        self.ACCELERATION_THRESHOLD = 0.4   # 加速度の閾値
        
        # アラート状態
        self.last_punch_time = 0
        self.last_kick_time = 0
        self.alert_duration = 1.0  # アラート表示時間（秒）
    
    def update(self, landmarks, timestamp):
        """
        ランドマークデータを更新
        
        Args:
            landmarks: MediaPipeのランドマークデータ
            timestamp: タイムスタンプ
        """
        if landmarks:
            self.landmark_history.append(landmarks)
            self.time_history.append(timestamp)
    
    def calculate_velocity(self, landmark_idx):
        """
        指定されたランドマークの速度を計算
        
        Args:
            landmark_idx (int): ランドマークのインデックス
            
        Returns:
            float: 速度
        """
        if len(self.landmark_history) < 2:
            return 0.0
        
        current = self.landmark_history[-1]
        previous = self.landmark_history[-2]
        
        if len(current) <= landmark_idx or len(previous) <= landmark_idx:
            return 0.0
        
        # 座標の差分を計算
        dx = current[landmark_idx]['x'] - previous[landmark_idx]['x']
        dy = current[landmark_idx]['y'] - previous[landmark_idx]['y']
        
        # 時間の差分
        dt = self.time_history[-1] - self.time_history[-2]
        if dt == 0:
            return 0.0
        
        # 速度を計算
        velocity = math.sqrt(dx**2 + dy**2) / dt
        return velocity
    
    def calculate_acceleration(self, landmark_idx):
        """
        指定されたランドマークの加速度を計算
        
        Args:
            landmark_idx (int): ランドマークのインデックス
            
        Returns:
            float: 加速度
        """
        if len(self.landmark_history) < 3:
            return 0.0
        
        # 直近3フレームの速度を計算
        velocities = []
        for i in range(-2, 0):
            if len(self.landmark_history) + i < 1:
                continue
            
            curr = self.landmark_history[i]
            prev = self.landmark_history[i-1]
            
            if len(curr) <= landmark_idx or len(prev) <= landmark_idx:
                continue
            
            dx = curr[landmark_idx]['x'] - prev[landmark_idx]['x']
            dy = curr[landmark_idx]['y'] - prev[landmark_idx]['y']
            dt = self.time_history[i] - self.time_history[i-1]
            
            if dt > 0:
                velocity = math.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # 加速度を計算
        dv = velocities[-1] - velocities[-2]
        dt = self.time_history[-1] - self.time_history[-2]
        
        if dt == 0:
            return 0.0
        
        acceleration = dv / dt
        return abs(acceleration)
    
    def detect_punch(self):
        """
        パンチ動作を検知
        
        Returns:
            tuple: (is_punch, side) - パンチ検知結果と左右の情報
        """
        current_time = time.time()
        
        # 左手のパンチ検知
        left_wrist_vel = self.calculate_velocity(self.POSE_LANDMARKS['LEFT_WRIST'])
        left_elbow_vel = self.calculate_velocity(self.POSE_LANDMARKS['LEFT_ELBOW'])
        left_wrist_acc = self.calculate_acceleration(self.POSE_LANDMARKS['LEFT_WRIST'])
        
        # 右手のパンチ検知
        right_wrist_vel = self.calculate_velocity(self.POSE_LANDMARKS['RIGHT_WRIST'])
        right_elbow_vel = self.calculate_velocity(self.POSE_LANDMARKS['RIGHT_ELBOW'])
        right_wrist_acc = self.calculate_acceleration(self.POSE_LANDMARKS['RIGHT_WRIST'])
        
        # パンチの条件: 手首の高速移動 + 肘の動き + 高い加速度
        left_punch = (left_wrist_vel > self.PUNCH_VELOCITY_THRESHOLD and 
                     left_elbow_vel > self.PUNCH_VELOCITY_THRESHOLD * 0.7 and
                     left_wrist_acc > self.ACCELERATION_THRESHOLD)
        
        right_punch = (right_wrist_vel > self.PUNCH_VELOCITY_THRESHOLD and 
                      right_elbow_vel > self.PUNCH_VELOCITY_THRESHOLD * 0.7 and
                      right_wrist_acc > self.ACCELERATION_THRESHOLD)
        
        if left_punch or right_punch:
            self.last_punch_time = current_time
            side = "左" if left_punch else "右"
            return True, side
        
        return False, None
    
    def detect_kick(self):
        """
        キック動作を検知
        
        Returns:
            tuple: (is_kick, side) - キック検知結果と左右の情報
        """
        current_time = time.time()
        
        # 左足のキック検知
        left_ankle_vel = self.calculate_velocity(self.POSE_LANDMARKS['LEFT_ANKLE'])
        left_knee_vel = self.calculate_velocity(self.POSE_LANDMARKS['LEFT_KNEE'])
        left_ankle_acc = self.calculate_acceleration(self.POSE_LANDMARKS['LEFT_ANKLE'])
        
        # 右足のキック検知
        right_ankle_vel = self.calculate_velocity(self.POSE_LANDMARKS['RIGHT_ANKLE'])
        right_knee_vel = self.calculate_velocity(self.POSE_LANDMARKS['RIGHT_KNEE'])
        right_ankle_acc = self.calculate_acceleration(self.POSE_LANDMARKS['RIGHT_ANKLE'])
        
        # キックの条件: 足首の高速移動 + 膝の動き + 高い加速度
        left_kick = (left_ankle_vel > self.KICK_VELOCITY_THRESHOLD and 
                    left_knee_vel > self.KICK_VELOCITY_THRESHOLD * 0.8 and
                    left_ankle_acc > self.ACCELERATION_THRESHOLD * 0.8)
        
        right_kick = (right_ankle_vel > self.KICK_VELOCITY_THRESHOLD and 
                     right_knee_vel > self.KICK_VELOCITY_THRESHOLD * 0.8 and
                     right_ankle_acc > self.ACCELERATION_THRESHOLD * 0.8)
        
        if left_kick or right_kick:
            self.last_kick_time = current_time
            side = "左" if left_kick else "右"
            return True, side
        
        return False, None
    
    def is_punch_alert_active(self):
        """パンチアラートが有効かチェック"""
        return time.time() - self.last_punch_time < self.alert_duration
    
    def is_kick_alert_active(self):
        """キックアラートが有効かチェック"""
        return time.time() - self.last_kick_time < self.alert_duration


class RealtimeSkeletonTracker:
    def __init__(self, camera_index=0):
        """
        リアルタイムスケルトン検出クラス
        
        Args:
            camera_index (int): 使用するカメラのインデックス（通常は0）
        """
        # MediaPipeの設定
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Poseモデルの初期化
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 動作検知器を初期化
        self.action_detector = ActionDetector()
        
        # Webカメラの設定
        self.cap = cv2.VideoCapture(camera_index)
        
        # カメラの解像度設定（オプション）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # フレームレート取得
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # デフォルト値
    
    def start_tracking(self):
        """
        リアルタイムスケルトン検出を開始
        """
        if not self.cap.isOpened():
            print("エラー: カメラが開けませんでした。")
            return
        
        print("リアルタイムスケルトン検出を開始します。")
        print("'q'キーを押すと終了します。")
        print("'s'キーを押すとスクリーンショットを保存します。")
        
        screenshot_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break
            
            # フレームを水平反転（鏡像効果）
            frame = cv2.flip(frame, 1)
            
            # フレームをRGBに変換（MediaPipeはRGBを使用）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ポーズ検出
            results = self.pose.process(rgb_frame)
            
            # スケルトンを描画
            if results.pose_landmarks:
                # ランドマークと接続線を描画
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 検出されたランドマーク数を表示
                landmark_count = len(results.pose_landmarks.landmark)
                cv2.putText(frame, f"Landmarks: {landmark_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # ポーズが検出されなかった場合
                cv2.putText(frame, "No pose detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS表示
            cv2.putText(frame, f"FPS: {int(self.fps)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # フレームを表示
            cv2.imshow('Realtime Skeleton Tracking', frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します。")
                break
            elif key == ord('s'):
                # スクリーンショット保存
                screenshot_filename = f"skeleton_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_filename, frame)
                print(f"スクリーンショットを保存しました: {screenshot_filename}")
                screenshot_count += 1
        
        # リソースを解放
        self.cleanup()
    
    def start_tracking_skeleton_only(self):
        """
        スケルトンのみを表示（背景は黒）
        """
        if not self.cap.isOpened():
            print("エラー: カメラが開けませんでした。")
            return
        
        print("スケルトンのみの表示モードを開始します。")
        print("'q'キーを押すと終了します。")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break
            
            # フレームを水平反転
            frame = cv2.flip(frame, 1)
            
            # 黒い背景を作成
            h, w, c = frame.shape
            black_background = np.zeros((h, w, c), dtype=np.uint8)
            
            # フレームをRGBに変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ポーズ検出
            results = self.pose.process(rgb_frame)
            
            # スケルトンを黒い背景に描画
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    black_background,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # フレームを表示
            cv2.imshow('Skeleton Only', black_background)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def get_landmark_coordinates(self, results):
        """
        ランドマークの座標を取得
        
        Args:
            results: MediaPipeの検出結果
            
        Returns:
            list: ランドマークの座標リスト
        """
        if not results.pose_landmarks:
            return []
        
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return landmarks
    
    def cleanup(self):
        """
        リソースを解放
        """
        self.cap.release()
        cv2.destroyAllWindows()
        print("カメラを終了しました。")
    
    def start_tracking_with_action_detection(self):
        """
        動作検知機能付きリアルタイムスケルトン検出を開始
        """
        if not self.cap.isOpened():
            print("エラー: カメラが開けませんでした。")
            return
        
        print("動作検知機能付きスケルトン検出を開始します。")
        print("'q'キーを押すと終了します。")
        print("'s'キーを押すとスクリーンショットを保存します。")
        print("パンチやキックを検知するとアラートが表示されます。")
        
        screenshot_count = 0
        punch_count = 0
        kick_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break
            
            # フレームを水平反転（鏡像効果）
            frame = cv2.flip(frame, 1)
            
            # フレームをRGBに変換（MediaPipeはRGBを使用）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ポーズ検出
            results = self.pose.process(rgb_frame)
            
            # スケルトンを描画
            if results.pose_landmarks:
                # ランドマークと接続線を描画
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # ランドマークデータを動作検知器に更新
                landmarks = self.get_landmark_coordinates(results)
                timestamp = time.time()
                self.action_detector.update(landmarks, timestamp)
                
                # 動作検知
                is_punch, punch_side = self.action_detector.detect_punch()
                is_kick, kick_side = self.action_detector.detect_kick()
                
                if is_punch:
                    punch_count += 1
                    print(f"パンチ検知! ({punch_side}手) - 総計: {punch_count}")
                
                if is_kick:
                    kick_count += 1
                    print(f"キック検知! ({kick_side}足) - 総計: {kick_count}")
                
                # 検出されたランドマーク数を表示
                landmark_count = len(results.pose_landmarks.landmark)
                cv2.putText(frame, f"Landmarks: {landmark_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # ポーズが検出されなかった場合
                cv2.putText(frame, "No pose detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # FPS表示
            cv2.putText(frame, f"FPS: {int(self.fps)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 動作カウンター表示
            cv2.putText(frame, f"Punches: {punch_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Kicks: {kick_count}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # アラート表示
            if self.action_detector.is_punch_alert_active():
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                cv2.putText(frame, "PUNCH DETECTED!", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            if self.action_detector.is_kick_alert_active():
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 165, 255), 10)
                cv2.putText(frame, "KICK DETECTED!", 
                           (frame.shape[1]//2 - 140, frame.shape[0]//2 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            
            # フレームを表示
            cv2.imshow('Action Detection - Skeleton Tracking', frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します。")
                break
            elif key == ord('s'):
                # スクリーンショット保存
                screenshot_filename = f"action_detection_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(screenshot_filename, frame)
                print(f"スクリーンショットを保存しました: {screenshot_filename}")
                screenshot_count += 1
        
        # 結果表示
        print(f"\n=== 検知結果 ===")
        print(f"パンチ検知回数: {punch_count}")
        print(f"キック検知回数: {kick_count}")
        
        # リソースを解放
        self.cleanup()

def main():
    """
    メイン関数 - 使用例
    """
    print("リアルタイムスケルトン検出")
    print("1: 通常モード（カメラ映像 + スケルトン）")
    print("2: スケルトンのみモード")
    print("3: 動作検知モード（パンチ・キック検知）")
    
    try:
        choice = input("モードを選択してください (1, 2, or 3): ")
        
        # スケルトン検出器を初期化
        tracker = RealtimeSkeletonTracker()
        
        if choice == "1":
            tracker.start_tracking()
        elif choice == "2":
            tracker.start_tracking_skeleton_only()
        elif choice == "3":
            tracker.start_tracking_with_action_detection()
        else:
            print("無効な選択です。動作検知モードで開始します。")
            tracker.start_tracking_with_action_detection()
            
    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()