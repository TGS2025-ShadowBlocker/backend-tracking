import mediapipe as mp
import cv2
import numpy as np

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
        
        self.frame_count += 1
        return frame
    
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


if __name__ == "__main__":
    # tracking.pyを単体で実行した場合のテスト用コード
    import datetime
    
    print("RealtimePoseTrackerのテストを開始します")
    print("Webカメラを使用してポーズトラッキングをテストします")
    print("'q'キーで終了, 's'キーでデータ保存")
    
    # カメラを初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        exit()
    
    # ポーズトラッカーを初期化
    tracker = RealtimePoseTracker()
    is_saving = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました")
                break
            
            # 左右反転
            frame = cv2.flip(frame, 1)
            
            # ポーズトラッキングを実行
            processed_frame = tracker.process_frame(frame)
            
            # 保存状態を表示
            if is_saving:
                cv2.putText(processed_frame, "Saving Data...", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # フレーム数を表示
            frame_text = f"Frame: {tracker.frame_count}"
            cv2.putText(processed_frame, frame_text, (10, processed_frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 画面に表示
            cv2.imshow('Pose Tracking Test', processed_frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                is_saving = not is_saving
                if is_saving:
                    tracker.reset_data()
                    print("データ保存を開始しました")
                else:
                    # データを保存
                    if len(tracker.skeleton_data) > 0:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"test_pose_data_{timestamp}.csv"
                        tracker.save_skeleton_data_csv(filename)
                        print(f"データを保存しました: {filename}")
                    else:
                        print("保存するデータがありません")
    
    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了します")
    
    finally:
        # リソースをクリーンアップ
        cap.release()
        cv2.destroyAllWindows()
        tracker.cleanup()
        print("テストを終了しました")