import cv2
from tracking import RealtimePoseTracker

class WebCamCapture:
    def __init__(self):
        # ウェブカメラのキャプチャを初期化
        self.cap = cv2.VideoCapture(0)

    def start_capture(self):
        """
        ウェブカメラのキャプチャを開始
        """
        print("'q'キーで終了")
        # キャプチャがオープンしている間続ける
        while(self.cap.isOpened()):
            # フレームを読み込む
            ret, frame = self.cap.read()
            if ret == True:
                # 左右反転
                frame = cv2.flip(frame, 1)

                # フレームを表示
                cv2.imshow('Webcam Live', frame)

                # 'q'キーが押されたらループから抜ける
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # キャプチャをリリースし、ウィンドウを閉じる
        self.cleanup()

    def cleanup(self):
        """
        リソースのクリーンアップ
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        デストラクタでリソースをクリーンアップ
        """
        if hasattr(self, 'cap'):
            self.cleanup()


class WebCamPoseTracker(WebCamCapture):
    def __init__(self):
        super().__init__()
        # ポーズトラッカーを初期化
        self.pose_tracker = RealtimePoseTracker()
        
    def start_pose_tracking(self, save_data=False):
        """
        ポーズトラッキング付きでWebカメラキャプチャを開始
        """
        print("トラッキングを開始")
        print("'q'キーで終了")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 左右反転
                frame = cv2.flip(frame, 1)
                
                # ポーズトラッキングを実行
                processed_frame = self.pose_tracker.process_frame(frame)
                
                # FPS情報を表示
                fps_text = f"Frame: {self.pose_tracker.frame_count}"
                cv2.putText(processed_frame, fps_text, (10, processed_frame.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # フレームを表示
                cv2.imshow('Pose Tracking', processed_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                break
        
        # データ保存の確認
        if save_data and len(self.pose_tracker.skeleton_data) > 0:
            self.save_tracking_data()
        
        self.cleanup()
    
    def save_tracking_data(self):
        """
        トラッキングデータをCSVファイルに保存
        """
        if len(self.pose_tracker.skeleton_data) > 0:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_pose_data_{timestamp}.csv"
            self.pose_tracker.save_skeleton_data_csv(filename)
        else:
            print("保存するデータがありません")
    
    def cleanup(self):
        """
        リソースのクリーンアップ
        """
        super().cleanup()
        if hasattr(self, 'pose_tracker'):
            self.pose_tracker.cleanup()


if __name__ == "__main__":
    # ポーズトラッキング付きWebカメラを実行
    pose_webcam = WebCamPoseTracker()
    pose_webcam.start_pose_tracking(save_data=False)
