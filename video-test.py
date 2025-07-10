import cv2
import os
import datetime
from tracking import RealtimePoseTracker

def print_video_info(fps, frame_count, duration):
    """動画情報を表示する関数"""
    print("動画情報:")
    print(f"  FPS: {fps:.2f}")
    print(f"  総フレーム数: {frame_count}")
    print(f"  再生時間: {duration:.2f}秒")

def draw_video_info(frame, current_frame, frame_count, current_time, duration):
    """動画の進捗情報を画面に描画する関数"""
    progress = (current_frame / frame_count) * 100
    frame_text = f"Frame: {current_frame}/{frame_count} | Progress: {progress:.1f}%"
    cv2.putText(frame, frame_text, (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    time_text = f"Time: {current_time:.2f}s / {duration:.2f}s"
    cv2.putText(frame, time_text, (10, frame.shape[0] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_status_info(frame, is_saving, is_paused):
    """保存状態と一時停止状態を画面に描画する関数"""
    if is_saving:
        cv2.putText(frame, "Saving Data...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if is_paused:
        cv2.putText(frame, "PAUSED", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def draw_debug_info(frame, debug_info):
    """デバッグ情報を画面に描画する関数"""
    y_start = 150
    font_size = 0.5
    
    # 手首の速度と加速度を表示
    cv2.putText(frame, f"L_Wrist V:{debug_info['left_wrist_velocity']:.3f} A:{debug_info['left_wrist_acceleration']:.3f}", 
                (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
    cv2.putText(frame, f"R_Wrist V:{debug_info['right_wrist_velocity']:.3f} A:{debug_info['right_wrist_acceleration']:.3f}", 
                (10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
    
    # 足首の速度と加速度を表示
    cv2.putText(frame, f"L_Ankle V:{debug_info['left_ankle_velocity']:.3f} A:{debug_info['left_ankle_acceleration']:.3f}", 
                (10, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
    cv2.putText(frame, f"R_Ankle V:{debug_info['right_ankle_velocity']:.3f} A:{debug_info['right_ankle_acceleration']:.3f}", 
                (10, y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
    
    # 閾値を表示
    cv2.putText(frame, "Threshold V:0.02 A:0.01", 
                (10, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 1)

def handle_save_data(tracker, is_saving):
    """データ保存の処理を行う関数"""
    if is_saving:
        tracker.reset_data()
        print("データ保存を開始しました")
        return False
    else:
        # データを保存
        if len(tracker.skeleton_data) > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_pose_data_{timestamp}.csv"
            tracker.save_skeleton_data_csv(filename)
            print(f"データを保存しました: {filename}")
        else:
            print("保存するデータがありません")
        return True

def handle_key_input(key, tracker, is_saving, is_paused, cap):
    """キー入力処理を行う関数"""
    if key == ord('q'):
        return False, is_saving, is_paused  # 終了
    elif key == ord('s'):
        is_saving = handle_save_data(tracker, is_saving)
    elif key == ord(' '):  # スペースキーで一時停止/再開
        is_paused = not is_paused
        status = "一時停止中..." if is_paused else "再開しました"
        print(status)
    elif key == ord('r'):  # 'r'キーで動画を最初から再生
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracker.reset_data()
        print("動画を最初から再生します")
    
    return True, is_saving, is_paused  # 継続

def process_video_frame(cap, tracker, fps, frame_count, duration, is_saving, is_paused):
    """動画フレームを処理する関数"""
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了しました")
        return None
    
    # ポーズトラッキングを実行
    processed_frame = tracker.process_frame(frame)
    
    # 状態情報を描画
    draw_status_info(processed_frame, is_saving, is_paused)
    
    # 進捗情報を描画
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps if fps > 0 else 0
    draw_video_info(processed_frame, current_frame, frame_count, current_time, duration)
    
    # デバッグ情報を描画
    debug_info = tracker.get_debug_info()
    draw_debug_info(processed_frame, debug_info)
    
    return processed_frame

def initialize_video_capture(video_path):
    """動画キャプチャを初期化する関数"""
    if not os.path.exists(video_path):
        print(f"エラー: 動画ファイル '{video_path}' が見つかりません")
        return None, None, None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした")
        return None, None, None, None
    
    # 動画の情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    return cap, fps, frame_count, duration

def test_video_tracking(video_path):
    """動画ファイルでポーズトラッキングをテストする関数"""
    print(f"動画ファイル '{video_path}' を使用してポーズトラッキングをテストします")
    print("'q'キーで終了, 's'キーでデータ保存, 'SPACE'キーで一時停止/再開, 'r'キーで最初から再生")
    
    # 動画キャプチャを初期化
    cap, fps, frame_count, duration = initialize_video_capture(video_path)
    if cap is None:
        return False
    
    print_video_info(fps, frame_count, duration)
    
    # ポーズトラッカーを初期化
    tracker = RealtimePoseTracker()
    is_saving = False
    is_paused = False
    
    # 動画の再生速度を調整（30FPSに正規化）
    wait_time = max(1, int(1000 / 30))
    
    try:
        while True:
            if not is_paused:
                processed_frame = process_video_frame(cap, tracker, fps, frame_count, duration, is_saving, is_paused)
                if processed_frame is None:
                    break
                
                # 画面に表示
                cv2.imshow('Video Pose Tracking Test', processed_frame)
            
            # キー入力処理
            key = cv2.waitKey(wait_time) & 0xFF
            continue_loop, is_saving, is_paused = handle_key_input(key, tracker, is_saving, is_paused, cap)
            if not continue_loop:
                break
    
    except KeyboardInterrupt:
        print("\nキーボード割り込みで終了します")
    
    finally:
        # リソースをクリーンアップ
        cap.release()
        cv2.destroyAllWindows()
        tracker.cleanup()
        print("テストを終了しました")
    
    return True

if __name__ == "__main__":
    # 動画ファイルのパス
    video_path = "video/input_video.mp4"
    
    # 動画トラッキングをテスト
    test_video_tracking(video_path)