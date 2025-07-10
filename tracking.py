import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import math

class SimpleActionDetector:
    def __init__(self, velocity_threshold=0.03, acceleration_threshold=0.02):
        """
        シンプルな動作検知器の初期化
        """
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
        # 関節の位置履歴を保存
        self.left_wrist_history = deque(maxlen=10)
        self.right_wrist_history = deque(maxlen=10)
        self.left_ankle_history = deque(maxlen=10)
        self.right_ankle_history = deque(maxlen=10)
        
        # 動作検知状態
        self.punch_detected = False
        self.kick_detected = False
        self.action_cooldown = 0
        
        # デバッグ情報
        self.debug_info = {
            'left_wrist_velocity': 0.0,
            'right_wrist_velocity': 0.0,
            'left_ankle_velocity': 0.0,
            'right_ankle_velocity': 0.0,
            'left_wrist_acceleration': 0.0,
            'right_wrist_acceleration': 0.0,
            'left_ankle_acceleration': 0.0,
            'right_ankle_acceleration': 0.0,
            'left_punch_score': 0.0,
            'right_punch_score': 0.0,
            'left_kick_score': 0.0,
            'right_kick_score': 0.0
        }
    
    def calculate_velocity(self, history):
        """位置履歴から速度を計算"""
        if len(history) < 2:
            return 0.0
        
        current_pos = history[-1]
        previous_pos = history[-2]
        
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        
        return math.sqrt(dx*dx + dy*dy)
    
    def calculate_acceleration(self, history):
        """位置履歴から加速度を計算"""
        if len(history) < 3:
            return 0.0
        
        v1 = self.calculate_velocity(deque([history[-3], history[-2]], maxlen=2))
        v2 = self.calculate_velocity(deque([history[-2], history[-1]], maxlen=2))
        
        return abs(v2 - v1)
    
    def update(self, landmarks):
        """動作検知を更新"""
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return
        
        # 関節の位置を更新
        if len(landmarks) > 28:  # MediaPipeの関節数確認
            # 関節の位置を取得
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            self.left_wrist_history.append([left_wrist['x'], left_wrist['y']])
            self.right_wrist_history.append([right_wrist['x'], right_wrist['y']])
            self.left_ankle_history.append([left_ankle['x'], left_ankle['y']])
            self.right_ankle_history.append([right_ankle['x'], right_ankle['y']])
            
            # 速度と加速度を計算
            left_wrist_vel = self.calculate_velocity(self.left_wrist_history)
            right_wrist_vel = self.calculate_velocity(self.right_wrist_history)
            left_ankle_vel = self.calculate_velocity(self.left_ankle_history)
            right_ankle_vel = self.calculate_velocity(self.right_ankle_history)
            
            left_wrist_acc = self.calculate_acceleration(self.left_wrist_history)
            right_wrist_acc = self.calculate_acceleration(self.right_wrist_history)
            left_ankle_acc = self.calculate_acceleration(self.left_ankle_history)
            right_ankle_acc = self.calculate_acceleration(self.right_ankle_history)
            
            # デバッグ情報を更新
            self.debug_info.update({
                'left_wrist_velocity': left_wrist_vel,
                'right_wrist_velocity': right_wrist_vel,
                'left_ankle_velocity': left_ankle_vel,
                'right_ankle_velocity': right_ankle_vel,
                'left_wrist_acceleration': left_wrist_acc,
                'right_wrist_acceleration': right_wrist_acc,
                'left_ankle_acceleration': left_ankle_acc,
                'right_ankle_acceleration': right_ankle_acc
            })
            
            # パンチ検知（方向性と姿勢を考慮）
            left_punch_score = self.calculate_punch_score(
                left_wrist_vel, left_wrist_acc, left_wrist, left_shoulder
            )
            right_punch_score = self.calculate_punch_score(
                right_wrist_vel, right_wrist_acc, right_wrist, right_shoulder
            )
            
            # キック検知（方向性と姿勢を考慮）
            left_kick_score = self.calculate_kick_score(
                left_ankle_vel, left_ankle_acc, left_ankle, left_hip
            )
            right_kick_score = self.calculate_kick_score(
                right_ankle_vel, right_ankle_acc, right_ankle, right_hip
            )
            
            # スコアを更新
            self.debug_info['left_punch_score'] = left_punch_score
            self.debug_info['right_punch_score'] = right_punch_score
            self.debug_info['left_kick_score'] = left_kick_score
            self.debug_info['right_kick_score'] = right_kick_score
            
            # 動作検知判定（互いに排他的）
            punch_threshold = 0.08
            kick_threshold = 0.08
            
            max_punch_score = max(left_punch_score, right_punch_score)
            max_kick_score = max(left_kick_score, right_kick_score)
            
            # キックが検知された場合、パンチは無効
            if max_kick_score > kick_threshold:
                self.kick_detected = True
                self.punch_detected = False
                self.action_cooldown = 15
                print(f"キック検知! L:{left_kick_score:.3f}, R:{right_kick_score:.3f}")
            elif max_punch_score > punch_threshold:
                self.punch_detected = True
                self.kick_detected = False
                self.action_cooldown = 15
                print(f"パンチ検知! L:{left_punch_score:.3f}, R:{right_punch_score:.3f}")
            else:
                self.punch_detected = False
                self.kick_detected = False
    
    def calculate_punch_score(self, velocity, acceleration, wrist, shoulder):
        """パンチスコアを計算（方向性を考慮）"""
        base_score = velocity + acceleration
        
        # 前方向への動きを重視
        if len(self.left_wrist_history) >= 2:
            # 手首が前方（画面の中央方向）に動いているかチェック
            direction_factor = 1.0
            if abs(wrist['x'] - 0.5) < abs(shoulder['x'] - 0.5):  # 手首が体の中心に近い場合
                direction_factor = 1.5  # 前方向への動きを強化
            
            # 手首が肩より前に出ている場合
            if abs(wrist['x'] - shoulder['x']) > 0.1:
                direction_factor *= 1.3
            
            base_score *= direction_factor
        
        return base_score
    
    def calculate_kick_score(self, velocity, acceleration, ankle, hip):
        """キックスコアを計算（上方向の動きを重視）"""
        base_score = velocity + acceleration
        
        # 上方向への動きを重視
        # 足首が腰より上に上がっている場合
        if ankle['y'] < hip['y']:
            base_score *= 2.0  # 上方向への動きを強化
        
        # 明確な上下動を検出
        if velocity > 0.02:  # 明確な動きがある場合
            base_score *= 1.5
        
        return base_score

    def get_detected_actions(self):
        """検知された動作を返す"""
        return {
            'punch': self.punch_detected,
            'kick': self.kick_detected
        }
    
    def get_debug_info(self):
        """デバッグ情報を返す"""
        return self.debug_info

class RealtimePoseTracker:
    def __init__(self):
        # MediaPipeの設定
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Poseモデルの初期化（リアルタイム用に最適化）
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # フレームカウンター
        self.frame_count = 0
        self.skeleton_data = []
        
        # シンプルな動作検知器を初期化
        self.action_detector = SimpleActionDetector()
    
    def process_frame(self, frame):
        """
        単一フレームを処理してポーズトラッキングを実行
        """
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
            
            # データを保存（オプション）
            frame_data = self._extract_landmark_data(results.pose_landmarks)
            self.skeleton_data.append({
                'frame': self.frame_count,
                'landmarks': frame_data
            })
            
            # 動作検知を更新
            self.action_detector.update(frame_data)
            
            # 検知された動作を画面に表示
            self._draw_action_detection(frame)
        
        self.frame_count += 1
        return frame
    
    def _draw_action_detection(self, frame):
        """検知された動作を画面に描画"""
        actions = self.action_detector.get_detected_actions()
        
        y_offset = 100
        if actions['punch']:
            cv2.putText(frame, "PUNCH DETECTED!", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            y_offset += 40
        
        if actions['kick']:
            cv2.putText(frame, "KICK DETECTED!", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    def get_detected_actions(self):
        """現在検知されている動作を返す"""
        return self.action_detector.get_detected_actions()
    
    def get_debug_info(self):
        """デバッグ情報を返す"""
        return self.action_detector.get_debug_info()
    
    def _extract_landmark_data(self, landmarks):
        """
        ランドマークデータを抽出
        """
        landmark_data = []
        for landmark in landmarks.landmark:
            landmark_data.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmark_data
    
    def get_skeleton_data(self):
        """
        収集したスケルトンデータを返す
        """
        return self.skeleton_data
    
    def save_skeleton_data_csv(self, csv_path):
        """
        収集したスケルトンデータをCSVファイルに保存
        """
        import pandas as pd
        
        if not self.skeleton_data:
            print("保存するデータがありません")
            return
        
        # データをフラット化
        rows = []
        for frame_data in self.skeleton_data:
            row = {'frame': frame_data['frame']}
            for landmark_idx, landmark in enumerate(frame_data['landmarks']):
                row[f'landmark_{landmark_idx}_x'] = landmark['x']
                row[f'landmark_{landmark_idx}_y'] = landmark['y']
                row[f'landmark_{landmark_idx}_z'] = landmark['z']
                row[f'landmark_{landmark_idx}_visibility'] = landmark['visibility']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"スケルトンデータを保存: {csv_path}")
    
    def reset_data(self):
        """
        収集したデータをリセット
        """
        self.skeleton_data = []
        self.frame_count = 0
    
    def cleanup(self):
        """
        リソースのクリーンアップ
        """
        if hasattr(self, 'pose'):
            self.pose.close()
