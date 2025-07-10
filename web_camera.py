import cv2
from tracking import RealtimePoseTracker
import datetime

def main():
    print("ポーズトラッキング（ウィンドウ表示）を開始します")
    print("Webカメラを使用してポーズトラッキングとキック・パンチ検知をテストします")
    print("'q'キーで終了, 's'キーでデータ保存, 'SPACE'キーで一時停止/再開")

    # カメラを初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return

    # ウィンドウサイズを設定
    cv2.namedWindow('Pose Tracking - Camera View', cv2.WINDOW_RESIZABLE)

    # ポーズトラッカーを初期化
    tracker = RealtimePoseTracker()
    is_saving = False
    is_paused = False

    try:
        while True:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("フレームの読み込みに失敗しました")
                    break
                
                # 左右反転
                frame = cv2.flip(frame, 1)
                
                # ポーズトラッキングを実行
                processed_frame = tracker.process_frame(frame)
                
                # 状態表示
                if is_saving:
                    cv2.putText(processed_frame, "Saving Data...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if is_paused:
                    cv2.putText(processed_frame, "PAUSED", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # フレーム数を表示
                frame_text = f"Frame: {tracker.frame_count}"
                cv2.putText(processed_frame, frame_text, (10, processed_frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
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
                
                # 速度を表示
                cv2.putText(processed_frame, f"L_Wrist V:{debug_info['left_wrist_velocity']:.3f}", 
                            (10, y_start + 120), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"R_Wrist V:{debug_info['right_wrist_velocity']:.3f}", 
                            (10, y_start + 150), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"L_Ankle V:{debug_info['left_ankle_velocity']:.3f}", 
                            (10, y_start + 180), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"R_Ankle V:{debug_info['right_ankle_velocity']:.3f}", 
                            (10, y_start + 210), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                
                # 閾値を表示
                cv2.putText(processed_frame, "Threshold: 0.08", 
                            (10, y_start + 240), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 2)
                
                # 操作説明を表示
                cv2.putText(processed_frame, "Press 'q' to quit, 's' to save, SPACE to pause", 
                            (10, processed_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # ウィンドウに表示
                cv2.imshow('Pose Tracking - Camera View', processed_frame)
            
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
                        filename = f"pose_data_{timestamp}.csv"
                        tracker.save_skeleton_data_csv(filename)
                        print(f"データを保存しました: {filename}")
                    else:
                        print("保存するデータがありません")
            elif key == ord(' '):  # スペースキーで一時停止/再開
                is_paused = not is_paused
                if is_paused:
                    print("一時停止中...")
                else:
                    print("再開しました")

    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了します")

    finally:
        # リソースをクリーンアップ
        cap.release()
        cv2.destroyAllWindows()
        tracker.cleanup()
        print("ポーズトラッキングを終了しました")

if __name__ == "__main__":
    main()
