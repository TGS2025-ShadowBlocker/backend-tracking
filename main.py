import cv2
import mediapipe as mp
import numpy as np

class VideoSkeletonExtractor:
    def __init__(self):
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
    
    def process_video(self, input_path, output_path):
        """
        動画を処理してスケルトンを描画
        """
        # 動画ファイルを開く
        cap = cv2.VideoCapture(input_path)
        
        # 動画の情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 出力動画の設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"処理開始: {total_frames}フレーム")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
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
            
            # フレームを出力動画に追加
            out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # 30フレームごとに進捗表示
                print(f"処理済み: {frame_count}/{total_frames} フレーム")
        
        # リソースを解放
        cap.release()
        out.release()
        
        print(f"処理完了: {output_path}")
    
    def process_video_realtime_preview(self, input_path):
        """
        リアルタイムプレビュー付きで動画を処理
        """
        cap = cv2.VideoCapture(input_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームをRGBに変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ポーズ検出
            results = self.pose.process(rgb_frame)
            
            # スケルトンを描画
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # フレームを表示
            cv2.imshow('Skeleton Detection', frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def extract_skeleton_data(self, input_path):
        """
        スケルトンデータ（座標）を抽出してリストで返す
        """
        cap = cv2.VideoCapture(input_path)
        skeleton_data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームをRGBに変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ポーズ検出
            results = self.pose.process(rgb_frame)
            
            frame_data = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    frame_data.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            
            skeleton_data.append(frame_data)
        
        cap.release()
        return skeleton_data
    
    def save_skeleton_data_csv(self, input_path, csv_path):
        """
        スケルトンデータをCSVファイルに保存
        """
        import pandas as pd
        
        skeleton_data = self.extract_skeleton_data(input_path)
        
        # データをフラット化
        rows = []
        for frame_idx, frame_data in enumerate(skeleton_data):
            row = {'frame': frame_idx}
            for landmark_idx, landmark in enumerate(frame_data):
                row[f'landmark_{landmark_idx}_x'] = landmark['x']
                row[f'landmark_{landmark_idx}_y'] = landmark['y']
                row[f'landmark_{landmark_idx}_z'] = landmark['z']
                row[f'landmark_{landmark_idx}_visibility'] = landmark['visibility']
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"スケルトンデータを保存: {csv_path}")

# 使用例
def main():
    # スケルトン抽出器を初期化
    extractor = VideoSkeletonExtractor()
    
    # 動画ファイルのパス
    input_video = "input_video.mp4"  # 入力動画ファイル
    output_video = "output_skeleton.mp4"  # 出力動画ファイル
    csv_output = "skeleton_data.csv"  # CSVファイル
    
    try:
        # 方法1: スケルトン付き動画を作成
        print("スケルトン動画を作成中...")
        extractor.process_video(input_video, output_video)
        
        # 方法2: リアルタイムプレビュー（オプション）
        # print("リアルタイムプレビューを開始...")
        # extractor.process_video_realtime_preview(input_video)
        
        # 方法3: スケルトンデータをCSVに保存
        print("スケルトンデータをCSVに保存中...")
        extractor.save_skeleton_data_csv(input_video, csv_output)
        
    except FileNotFoundError:
        print(f"エラー: 動画ファイル '{input_video}' が見つかりません")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()
